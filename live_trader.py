import os
import sys
import json
import gc
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# sys.path.insert(0, str(Path(__file__).parent / "pip_libs"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import tushare as ts

warnings.filterwarnings('ignore')

from ml_strategy.oamv_filter import OAMVHysteresisFilter
from ml_strategy.feature_engine import FeatureDiscretizer
from ml_strategy.triple_barrier import TripleBarrierLabeler
from ml_strategy.catboost_predictor import CatBoostPredictor
from ml_strategy.panic_breaker import MarketPanicCircuitBreaker
from ml_strategy.ssa_denoiser import SSADenoiser
from ml_strategy.chebykan_predictor import ChebyKANTrainer
from ml_strategy.rade_ensemble import RADEEnsemble
from ml_strategy.drift_detector import ADDMDriftDetector
from ml_strategy.sterile_cleaner import SterileDataCleaner
from ml_strategy.disagreement_features import DisagreementFeatureBuilder
from ml_strategy.cost_aware_optimizer import CostAwarePortfolioOptimizer
from ml_strategy.path_signature import PathSignatureBuilder
from ml_strategy.llm_analyzer import LLMStockAnalyzer
from auto_trader import AutoTrader

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
SERVERCHAN_KEYS = [k.strip() for k in os.getenv('SERVERCHAN_KEY', '').split(',') if k.strip()]
pro = None
if TUSHARE_TOKEN:
    try:
        pro = ts.pro_api(TUSHARE_TOKEN)
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api(timeout=120)
    except Exception as e:
        print(f"Tushare init failed: {e}")
        sys.exit(1)

OAMV_UPPER = 4.0
OAMV_LOWER = -2.3
CATBOOST_BUY_THRESHOLD = 0.65
MAX_PORTFOLIO_STOCKS = 3
COMMISSION_RATE = 0.0003
STAMP_DUTY_RATE = 0.001
SLIPPAGE_RATE = 0.0005
FRICTION_COST_PCT = (COMMISSION_RATE * 2 + STAMP_DUTY_RATE + SLIPPAGE_RATE * 2) * 100
MIN_AMPLITUDE_MULT = 1.5
MIN_PRICE = 3.0
MIN_HOLD_DAYS = 3
COOLDOWN_DAYS = 5
TRAILING_STOP_PCT = 8.0
ATR_STOP_MULT = 1.5
PANIC_BREADTH_THRESHOLD = 0.85
PANIC_LIMIT_DOWN_THRESHOLD = 150
POSITION_SIZE_PCT = 0.25
IMPACT_MODEL = 'sqrt'
IMPACT_COEFFICIENT = 0.4
SPREAD_HALF = 0.001
MAX_SLIPPAGE_PCT = 2.0
TRAIN_WINDOW_MONTHS = 12
STOCK_LIMIT = 500
PORTFOLIO_DIR = Path(__file__).parent / "portfolio_state"
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_FILE = PORTFOLIO_DIR / "portfolio.json"
TRADE_LOG_FILE = PORTFOLIO_DIR / "trade_log.json"
SIGNAL_DIR = PORTFOLIO_DIR / "signals"
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)


def is_st_stock(name: str) -> bool:
    for kw in ['ST', '*ST', 'S*ST', 'SST']:
        if name.startswith(kw):
            return True
    return False


def get_stock_daily(ts_code, start_date='20240101', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        col_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'vol': 'Volume', 'trade_date': 'Date'}
        for old, new in col_map.items():
            if old in df.columns:
                df[new] = df[old]
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)
        return df
    except Exception:
        return None


def get_index_daily(ts_code='000001.SH', start_date='20240101', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    try:
        df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        col_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'vol': 'Volume', 'amount': 'amount',
                   'trade_date': 'Date'}
        for old, new in col_map.items():
            if old in df.columns:
                df[new] = df[old]
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)
        return df
    except Exception:
        return None


def compute_indicators(df):
    df = df.copy()
    if 'Close' not in df.columns:
        return None
    df['white_line'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['white_line'] = df['white_line'].ewm(span=10, adjust=False).mean()
    df['ma14'] = df['Close'].rolling(window=14).mean()
    df['ma28'] = df['Close'].rolling(window=28).mean()
    df['ma57'] = df['Close'].rolling(window=57).mean()
    df['ma114'] = df['Close'].rolling(window=114).mean()
    df['yellow_line'] = (df['ma14'] + df['ma28'] + df['ma57'] + df['ma114']) / 4
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(window=14, min_periods=1).mean()
    df['Vol_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema_fast - ema_slow
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    low_list = df['Low'].rolling(window=9, min_periods=1).min()
    high_list = df['High'].rolling(window=9, min_periods=1).max()
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)
    df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df['D_val'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D_val']
    df['amplitude_20'] = ((df['High'] - df['Low']) / df['Close'] * 100).rolling(window=20).mean()
    return df


INDUSTRY_MAP = {
    '银行': '801780.SI', '房地产': '801180.SI', '保险': '801790.SI',
    '证券': '801790.SI', '多元金融': '801790.SI',
    '食品饮料': '801120.SI', '家用电器': '801110.SI', '纺织服饰': '801130.SI',
    '汽车': '801880.SI', '商贸零售': '801200.SI', '社会服务': '801210.SI',
    '医药生物': '801150.SI', '农林牧渔': '801010.SI',
    '电子': '801080.SI', '计算机': '801750.SI', '通信': '801770.SI', '传媒': '801760.SI',
    '电力设备': '801730.SI', '机械设备': '801890.SI', '国防军工': '801740.SI',
    '基础化工': '801030.SI', '石油石化': '801960.SI', '钢铁': '801040.SI',
    '有色金属': '801050.SI', '煤炭': '801950.SI', '建筑材料': '801710.SI',
    '建筑装饰': '801720.SI', '环保': '801970.SI', '公用事业': '801160.SI',
    '交通运输': '801170.SI', '综合': '801230.SI',
    '轻工制造': '801140.SI', '美容护理': '801980.SI',
}

INDUSTRY_ETF_FALLBACK = {
    '银行': '512800.SH', '房地产': '512200.SH', '保险': '512070.SH',
    '证券': '512880.SH', '多元金融': '512880.SH',
    '食品饮料': '515170.SH', '家用电器': '159996.SZ', '纺织服饰': '159993.SZ',
    '汽车': '516110.SH', '商贸零售': '516180.SH', '社会服务': '159766.SZ',
    '医药生物': '512010.SH', '农林牧渔': '159825.SZ',
    '电子': '159997.SZ', '计算机': '512720.SH', '通信': '515880.SH', '传媒': '159805.SZ',
    '电力设备': '159611.SZ', '机械设备': '159886.SZ', '国防军工': '512660.SH',
    '基础化工': '159870.SZ', '石油石化': '159861.SZ', '钢铁': '159861.SZ',
    '有色金属': '512400.SH', '煤炭': '515220.SH', '建筑材料': '159745.SZ',
    '建筑装饰': '159745.SZ', '环保': '159861.SZ', '公用事业': '159825.SZ',
    '交通运输': '159662.SZ', '综合': '159993.SZ',
    '轻工制造': '159993.SZ', '美容护理': '159993.SZ',
}


def compute_industry_j(industry_df):
    if industry_df is None or len(industry_df) < 20:
        return None
    low_9 = industry_df['Low'].rolling(window=9, min_periods=9).min()
    high_9 = industry_df['High'].rolling(window=9, min_periods=9).max()
    denom = high_9 - low_9
    rsv = pd.Series(np.where(denom == 0, 50, (industry_df['Close'] - low_9) / denom * 100),
                     index=industry_df.index, dtype=float)
    rsv = rsv.fillna(50)
    k = pd.Series(np.nan, index=industry_df.index, dtype=float)
    d = pd.Series(np.nan, index=industry_df.index, dtype=float)
    k.iloc[0] = 50.0
    d.iloc[0] = 50.0
    for i in range(1, len(industry_df)):
        k.iloc[i] = 2.0 / 3.0 * k.iloc[i - 1] + 1.0 / 3.0 * rsv.iloc[i]
        d.iloc[i] = 2.0 / 3.0 * d.iloc[i - 1] + 1.0 / 3.0 * k.iloc[i]
    j = 3.0 * k - 2.0 * d
    return j


def get_industry_daily(ts_code, start_date='20240101', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    try:
        if ts_code.endswith('.SI'):
            try:
                df = pro.sw_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df is not None and not df.empty:
                    df = df.sort_values('trade_date').reset_index(drop=True)
                    col_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                               'close': 'Close', 'vol': 'Volume', 'trade_date': 'Date'}
                    for old, new in col_map.items():
                        if old in df.columns:
                            df[new] = df[old]
                    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
                    df.set_index('Date', inplace=True)
                    return df
            except Exception:
                pass
        if ts_code.endswith('.SH') or ts_code.endswith('.SZ'):
            df = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        else:
            df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        col_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'vol': 'Volume', 'trade_date': 'Date'}
        for old, new in col_map.items():
            if old in df.columns:
                df[new] = df[old]
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)
        return df
    except Exception:
        return None


class LivePortfolio:
    def __init__(self, initial_cash=1000000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict = {}
        self.trade_history: List[Dict] = []
        self.recent_sells: Dict[str, str] = {}

    def save(self):
        data = {
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'positions': self.positions,
            'trade_history': self.trade_history,
            'recent_sells': self.recent_sells,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load(cls) -> 'LivePortfolio':
        if not PORTFOLIO_FILE.exists():
            return cls()
        try:
            with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            portfolio = cls(data.get('initial_cash', 1000000))
            portfolio.cash = data.get('cash', portfolio.initial_cash)
            portfolio.positions = data.get('positions', {})
            portfolio.trade_history = data.get('trade_history', [])
            portfolio.recent_sells = data.get('recent_sells', {})
            return portfolio
        except Exception:
            return cls()

    def get_position_count(self) -> int:
        return len(self.positions)

    def can_buy(self) -> bool:
        return self.get_position_count() < MAX_PORTFOLIO_STOCKS and self.cash > 1000

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        total = self.cash
        for ts_code, pos in self.positions.items():
            price = current_prices.get(ts_code, pos.get('entry_price', 0))
            total += pos.get('shares', 0) * price
        return total

    def add_position(self, ts_code, name, price, prob, atr, date_str, mvo_weight=None, **kwargs):
        target_value = self.initial_cash * POSITION_SIZE_PCT
        target_value = min(target_value, self.cash * 0.9)
        shares = int(target_value / price / 100) * 100
        if shares <= 0:
            return False
        amount = shares * price
        commission = max(amount * COMMISSION_RATE, 5.0)
        slippage = amount * kwargs.get('impact_slippage', SLIPPAGE_RATE)
        total_cost = amount + commission + slippage
        if total_cost > self.cash:
            return False
        self.cash -= total_cost
        self.positions[ts_code] = {
            'name': name,
            'entry_date': date_str,
            'entry_price': float(price),
            'shares': shares,
            'entry_prob': float(prob),
            'entry_atr': float(atr),
            'peak_price': float(price),
            'mvo_weight': float(mvo_weight) if mvo_weight is not None else None,
        }
        return True

    def remove_position(self, ts_code, price, date_str, reason=''):
        if ts_code not in self.positions:
            return None
        pos = self.positions[ts_code]
        amount = pos['shares'] * price
        commission = max(amount * COMMISSION_RATE, 5.0)
        stamp_duty = amount * STAMP_DUTY_RATE
        slippage = amount * SLIPPAGE_RATE
        net_proceeds = amount - commission - stamp_duty - slippage
        self.cash += net_proceeds
        profit_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
        profit_abs = pos['shares'] * (price - pos['entry_price'])
        hold_days = (pd.Timestamp(date_str) - pd.Timestamp(pos['entry_date'])).days
        trade_record = {
            'code': ts_code,
            'name': pos['name'],
            'entry_date': pos['entry_date'],
            'exit_date': date_str,
            'entry_price': pos['entry_price'],
            'exit_price': float(price),
            'shares': pos['shares'],
            'profit_pct': float(profit_pct),
            'profit_abs': float(profit_abs),
            'hold_days': hold_days,
            'exit_reason': reason,
            'entry_prob': pos['entry_prob'],
        }
        self.trade_history.append(trade_record)
        self.recent_sells[ts_code] = date_str
        del self.positions[ts_code]
        return trade_record


def send_serverchan(title, desp):
    if not SERVERCHAN_KEYS:
        return False
    import requests
    success_count = 0
    for key in SERVERCHAN_KEYS:
        url = f"https://sctapi.ftqq.com/{key}.send"
        data = {"title": title, "desp": desp}
        try:
            resp = requests.post(url, data=data, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                if result.get('code') == 0:
                    success_count += 1
        except Exception:
            pass
    if success_count > 0:
        print(f"  Server酱推送成功: {success_count}/{len(SERVERCHAN_KEYS)}")
        return True
    return False


def run_live_trader():
    print("=" * 90)
    print("v9.0 实盘交易系统 - PathSignatures + AlphaGlass + ProteuS + 四大共识过滤")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    portfolio = LivePortfolio.load()
    print(f"\n  初始资金: {portfolio.initial_cash:,.0f}")
    print(f"  当前现金: {portfolio.cash:,.2f}")
    print(f"  当前持仓: {portfolio.get_position_count()}/{MAX_PORTFOLIO_STOCKS}")
    if portfolio.positions:
        for code, pos in portfolio.positions.items():
            mvo_str = f"  MVO={pos['mvo_weight']:.2f}" if pos.get('mvo_weight') else ""
            print(f"    {code} {pos['name']}: {pos['shares']}股 @ {pos['entry_price']:.2f}{mvo_str}")

    print("\n" + "=" * 90)
    print("[Step 1] 四层风控检查")
    print("-" * 90)

    print("  加载行业数据...")
    industry_j_cache = {}
    for ind_name, sw_code in INDUSTRY_MAP.items():
        try:
            ind_df = get_industry_daily(sw_code)
            if ind_df is not None and len(ind_df) > 20:
                ind_j = compute_industry_j(ind_df)
                if ind_j is not None:
                    industry_j_cache[ind_name] = ind_j
                    continue
        except Exception:
            pass
        etf_code = INDUSTRY_ETF_FALLBACK.get(ind_name)
        if etf_code:
            try:
                ind_df = get_industry_daily(etf_code)
                if ind_df is not None and len(ind_df) > 20:
                    ind_j = compute_industry_j(ind_df)
                    if ind_j is not None:
                        industry_j_cache[ind_name] = ind_j
            except Exception:
                pass
    print(f"  行业: {len(industry_j_cache)}")

    ssa_denoiser = SSADenoiser(window_length=10, n_signal_groups=2)
    sig_builder = PathSignatureBuilder(truncation_level=2, path_dims=3, path_length=5, lead_lag=False)
    sterile_cleaner = SterileDataCleaner()
    disagreement_builder = DisagreementFeatureBuilder(ssa_window=10, ssa_signal_groups=2)
    portfolio_optimizer = CostAwarePortfolioOptimizer(
        n_scenarios=500, block_size=5, lookback_days=200,
        risk_aversion=0.5, cost_aversion=0.5, max_weight=0.25,
        min_weight=0.0, total_max_weight=0.75,
        impact_coefficient=0.4, spread_half=0.001,
        total_capital=portfolio.initial_cash
    )
    drift_detector = ADDMDriftDetector(ar_order=3, ph_threshold=2.0, ph_delta=0.01, use_vol_filter=True)

    print("  加载股票数据 (with SSA denoising)...")
    index_df = get_index_daily('000001.SH', '20240101')
    if index_df is None:
        print("  上证指数加载失败！")
        return
    index_df = compute_indicators(index_df)

    stock_basic = pro.stock_basic(exchange='', list_status='L',
                                   fields='ts_code,symbol,name,industry,list_date')
    a_stocks = stock_basic[
        (stock_basic['ts_code'].str.endswith('.SH')) |
        (stock_basic['ts_code'].str.endswith('.SZ'))
    ]
    a_stocks = a_stocks[~a_stocks['name'].apply(is_st_stock)]
    a_stocks = a_stocks.head(STOCK_LIMIT)

    all_stock_data = {}
    all_featured_data = {}
    discretizer = FeatureDiscretizer()
    total = len(a_stocks)

    for idx, (_, row) in enumerate(a_stocks.iterrows(), 1):
        ts_code = row['ts_code']
        name = row['name']
        industry = row.get('industry', '')
        if idx % 50 == 0:
            print(f"  进度: {idx}/{total} ({idx/total*100:.1f}%)")
        try:
            df = get_stock_daily(ts_code, '20240101')
            if df is None or len(df) < 200:
                continue
            if df['Close'].iloc[-1] < MIN_PRICE:
                continue
            df = compute_indicators(df)
            if df is None:
                continue
            all_stock_data[ts_code] = {'data': df, 'name': name, 'industry': industry}
            featured_df = discretizer.transform(df)
            ind_j = industry_j_cache.get(industry)
            featured_df = discretizer.add_market_context(featured_df, index_df, None, ind_j)
            featured_df = ssa_denoiser.denoise_features(featured_df)
            all_featured_data[ts_code] = featured_df
        except Exception:
            continue
        if idx % 100 == 0:
            gc.collect()

    print(f"  加载: {len(all_stock_data)} 只")

    print("  初始化0AMV过滤器...")
    oamv_filter = OAMVHysteresisFilter(
        upper_threshold=OAMV_UPPER, lower_threshold=OAMV_LOWER,
        cost_ma_period=34, weekly_ema_period=5, weekly_use_ema=True,
    )
    oamv_filter.fit(index_df, all_stock_data=all_stock_data)

    oamv_state_df = oamv_filter.get_state_df()
    latest_date = oamv_state_df.index[-1]
    latest_state = int(oamv_state_df.iloc[-1]['oamv_state'])
    latest_x = float(oamv_state_df.iloc[-1]['oamv_x'])
    daily_ok = oamv_filter.is_trading_allowed(latest_date, require_weekly=False)
    weekly_ok = oamv_filter.is_trading_allowed(latest_date, require_weekly=True)

    print(f"  日期: {latest_date.strftime('%Y-%m-%d')}")
    print(f"  Layer 1 - 日线0AMV: {'BULL' if daily_ok else 'BEAR'} (X={latest_x:+.2f}%)")
    print(f"  Layer 1 - 周线EMA-5: {'BULL' if weekly_ok else 'BEAR'}")

    print("  初始化熔断器...")
    panic_breaker = MarketPanicCircuitBreaker(
        breadth_threshold=PANIC_BREADTH_THRESHOLD,
        limit_down_threshold=PANIC_LIMIT_DOWN_THRESHOLD, ma_period=20
    )
    panic_breaker.compute_market_breadth(all_stock_data)
    market_state = panic_breaker.get_market_state(latest_date)
    is_panic = market_state == 'panic'
    state_display = {'normal': '正常', 'warning': '⚠️预警', 'panic': '触发!'}.get(market_state, '正常')
    print(f"  Layer 2 - 熔断器: {state_display}")

    if 'ATR14' in index_df.columns and latest_date in index_df.index:
        market_atr = float(index_df.loc[latest_date, 'ATR14'])
        market_close = float(index_df.loc[latest_date, 'Close'])
        market_vol = market_atr / market_close if market_close > 0 else 0.02
        drift_detector.set_market_volatility(market_vol)

    signals = []
    sell_signals = []
    buy_signals = []

    print("\n" + "=" * 90)
    print("[Step 2] 持仓风控检查")
    print("-" * 90)

    for ts_code, pos in list(portfolio.positions.items()):
        try:
            if ts_code not in all_stock_data:
                df = get_stock_daily(ts_code, '20240101')
                if df is None or len(df) == 0:
                    continue
                df = compute_indicators(df)
                if df is None:
                    continue
            else:
                df = all_stock_data[ts_code]['data']

            current_price = float(df['Close'].iloc[-1])
            current_date_str = df.index[-1].strftime('%Y-%m-%d')
            latest_row = df.iloc[-1]

            if current_price > pos['peak_price']:
                pos['peak_price'] = current_price

            hold_days = (pd.Timestamp(current_date_str) - pd.Timestamp(pos['entry_date'])).days

            sell_reason = None

            if is_panic:
                sell_reason = 'panic_circuit_breaker'
            elif not daily_ok:
                sell_reason = 'oamv_daily_bear'

            if sell_reason is None and hold_days >= MIN_HOLD_DAYS:
                yellow_line = latest_row.get('yellow_line')
                if yellow_line is not None and not pd.isna(yellow_line):
                    if current_price < yellow_line:
                        sell_reason = 'close < yellow_line'

                if sell_reason is None and pos['peak_price'] > 0:
                    drawdown_pct = (pos['peak_price'] - current_price) / pos['peak_price'] * 100
                    if drawdown_pct >= TRAILING_STOP_PCT:
                        sell_reason = f'{TRAILING_STOP_PCT:.0f}% trailing stop (DD={drawdown_pct:.1f}%)'

                if sell_reason is None and pos.get('entry_atr', 0) > 0:
                    drawdown_atr = (pos['peak_price'] - current_price) / pos['entry_atr']
                    if drawdown_atr >= ATR_STOP_MULT:
                        sell_reason = f'{ATR_STOP_MULT}x ATR stop (DD={drawdown_atr:.1f}x)'

            profit_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100

            if sell_reason:
                sell_signals.append({
                    'code': ts_code,
                    'name': pos['name'],
                    'action': 'SELL',
                    'price': current_price,
                    'reason': sell_reason,
                    'profit_pct': profit_pct,
                    'hold_days': hold_days,
                    'entry_price': pos['entry_price'],
                })
                print(f"  {ts_code} {pos['name']}: 卖出信号! 原因={sell_reason}, "
                      f"收益={profit_pct:+.2f}%, 持仓{hold_days}天")
            else:
                dd = ((pos['peak_price'] - current_price) / pos['peak_price'] * 100) if pos['peak_price'] > 0 else 0
                print(f"  {ts_code} {pos['name']}: 持有中, 收益={profit_pct:+.2f}%, "
                      f"持仓{hold_days}天, 峰值回撤={dd:.1f}%")

        except Exception as e:
            print(f"  {ts_code}: 检查异常 {e}")

    print("\n" + "=" * 90)
    print("[Step 3] 训练RADE集成模型 + 选股扫描")
    print("-" * 90)

    if not daily_ok:
        print("  日线0AMV = BEAR，跳过选股")
    elif is_panic:
        print("  熔断触发，跳过选股")
    elif market_state == 'warning':
        print("  市场预警，禁止新买入")
    elif not weekly_ok:
        print("  周线EMA-5 = BEAR，禁止新开仓")
    elif not portfolio.can_buy():
        print(f"  持仓已满 ({portfolio.get_position_count()}/{MAX_PORTFOLIO_STOCKS})，跳过选股")
    else:
        from dateutil.relativedelta import relativedelta
        train_end = latest_date.strftime('%Y%m%d')
        train_start_dt = latest_date - relativedelta(months=TRAIN_WINDOW_MONTHS)
        train_start = train_start_dt.strftime('%Y%m%d')
        print(f"  训练区间: {train_start} ~ {train_end}")

        oamv_state_dict = oamv_filter.get_state_dict()
        all_samples = []
        labeler = TripleBarrierLabeler()
        labeler.max_hold_days = 10
        feature_cols = CatBoostPredictor.FEATURE_COLS
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)

        for ts_code in list(all_stock_data.keys()):
            stock_info = all_stock_data[ts_code]
            df = stock_info['data']
            featured_df = all_featured_data[ts_code]
            try:
                oamv_stock = pd.Series(0, index=df.index, dtype=int)
                for d, state in oamv_state_dict.items():
                    if d in oamv_stock.index:
                        oamv_stock.loc[d] = state
                train_mask = (df.index >= train_start_ts) & (df.index < train_end_ts)
                state_mask = oamv_stock == 1
                combined_mask = state_mask & train_mask
                candidate_indices = [i for i in range(len(df)) if combined_mask.iloc[i]]
                if len(candidate_indices) == 0:
                    continue
                atr14 = df['ATR14'] if 'ATR14' in df.columns else pd.Series(np.nan, index=df.index)
                labels = labeler.label_all(df, candidate_indices, atr14)
                for lab in labels:
                    li = lab['entry_idx']
                    if li >= len(featured_df):
                        continue
                    feat_row = featured_df.iloc[li]
                    available_cols = [c for c in feature_cols if c in feat_row.index]
                    if any(pd.isna(feat_row.get(c)) for c in available_cols):
                        continue
                    sample = {c: feat_row[c] for c in available_cols}
                    sample['label'] = lab['label']
                    all_samples.append(sample)
            except Exception:
                continue

        if len(all_samples) < 50:
            print("  训练样本不足，跳过选股")
        else:
            samples_df = pd.DataFrame(all_samples)
            available_cols = [c for c in feature_cols if c in samples_df.columns]
            X_all = samples_df[available_cols].dropna()
            y_all = samples_df.loc[X_all.index, 'label'].values
            split_idx = int(len(X_all) * 0.8)
            X_train = X_all.iloc[:split_idx]
            y_train = y_all[:split_idx]
            X_val = X_all.iloc[split_idx:]
            y_val = y_all[split_idx:]

            catboost_model = CatBoostPredictor(buy_threshold=CATBOOST_BUY_THRESHOLD, l2_leaf_reg=8, max_depth=4)
            catboost_model.train(X_train, y_train, X_val, y_val)
            val_probs_cb = catboost_model.predict_proba(X_val)
            val_preds = val_probs_cb >= CATBOOST_BUY_THRESHOLD
            if val_preds.sum() > 0:
                precision = y_val[val_preds].mean()
                print(f"  CatBoost Precision@0.65: {precision:.2%}")

            kan_trainer = None
            try:
                kan_trainer = ChebyKANTrainer(input_dim=len(available_cols), hidden_dim=16, poly_degree=4, lr=0.005, epochs=200, batch_size=256)
                kan_trainer.train(X_train, y_train, X_val, y_val)
                val_probs_kan = kan_trainer.predict_proba(X_val)
                kan_preds = val_probs_kan >= CATBOOST_BUY_THRESHOLD
                if kan_preds.sum() > 0:
                    kan_precision = y_val[kan_preds].mean()
                    print(f"  ChebyKAN Precision@0.65: {kan_precision:.2%}")
            except Exception as e:
                print(f"  ChebyKAN训练失败: {e}")
                kan_trainer = None

            ensemble = RADEEnsemble(gamma=0.5)
            ensemble.set_models(catboost_model, kan_trainer)
            print(f"  {ensemble.summary()}")

            print("  扫描买入信号...")
            buy_candidates = []
            for ts_code, info in all_stock_data.items():
                if ts_code in portfolio.positions:
                    continue
                if ts_code in portfolio.recent_sells:
                    sell_date = portfolio.recent_sells[ts_code]
                    if (latest_date - pd.Timestamp(sell_date)).days < COOLDOWN_DAYS:
                        continue
                df = info['data']
                featured_df = all_featured_data[ts_code]
                if latest_date not in df.index or latest_date not in featured_df.index:
                    continue
                feat_row = featured_df.loc[latest_date]
                available_cols = [c for c in feature_cols if c in feat_row.index]
                if any(pd.isna(feat_row.get(c)) for c in available_cols):
                    continue
                clean_data = feat_row[available_cols].to_frame().T
                for col in ['price_zone', 'j_zone', 'k_pattern']:
                    if col in clean_data.columns:
                        clean_data[col] = clean_data[col].astype(int)

                atr_ratio = 0.02
                if 'atr_ratio' in feat_row.index and pd.notna(feat_row['atr_ratio']):
                    atr_ratio = float(feat_row['atr_ratio'])

                prob = ensemble.predict(clean_data, oamv_x_pct=latest_x, atr_ratio=atr_ratio)[0]

                if prob >= CATBOOST_BUY_THRESHOLD:
                    white_above = df.loc[latest_date, 'white_above_yellow'] if 'white_above_yellow' in df.columns else True
                    if pd.notna(white_above) and not white_above:
                        continue

                    j_val = df.loc[latest_date, 'j_clean'] if 'j_clean' in df.columns else 50
                    if pd.notna(j_val) and j_val >= 13:
                        continue

                    pwvc_val = df.loc[latest_date, 'pwvc'] if 'pwvc' in df.columns else 0
                    if pd.notna(pwvc_val) and pwvc_val > 0.8:
                        continue

                    amp = df.loc[latest_date, 'amplitude_20'] if 'amplitude_20' in df.columns else 0
                    if pd.notna(amp) and amp > 0 and amp < FRICTION_COST_PCT * MIN_AMPLITUDE_MULT:
                        continue
                    atr_val = df.loc[latest_date, 'ATR14'] if 'ATR14' in df.columns else 0
                    if pd.isna(atr_val):
                        atr_val = 0
                    stop_loss = max(float(df.loc[latest_date, 'Close']) - 1.5 * atr_val,
                                   float(df.loc[latest_date, 'Close']) * 0.92) if atr_val > 0 else 0
                    daily_vol = float(df.loc[latest_date, 'amplitude_20']) / 100 if 'amplitude_20' in df.columns and pd.notna(df.loc[latest_date, 'amplitude_20']) else 0.02
                    daily_vol_val = float(df.loc[latest_date, 'Volume']) if 'Volume' in df.columns and pd.notna(df.loc[latest_date, 'Volume']) else 0
                    order_shares = int(portfolio.initial_cash * POSITION_SIZE_PCT / float(df.loc[latest_date, 'Close']) / 100) * 100
                    if daily_vol_val > 0 and order_shares > 0:
                        participation = order_shares / daily_vol_val
                        impact_slippage = IMPACT_COEFFICIENT * daily_vol * np.sqrt(participation) + SPREAD_HALF
                        impact_slippage = min(impact_slippage, MAX_SLIPPAGE_PCT / 100)
                    else:
                        impact_slippage = SLIPPAGE_RATE
                    if impact_slippage > 0.005:
                        continue
                    buy_candidates.append({
                        'code': ts_code,
                        'name': info['name'],
                        'industry': info.get('industry', ''),
                        'prob': float(prob),
                        'price': float(df.loc[latest_date, 'Close']),
                        'atr': float(atr_val),
                        'stop_loss': float(stop_loss),
                        'impact_slippage': float(impact_slippage),
                    })

            buy_candidates.sort(key=lambda x: -x['prob'])

            top_n = min(5, len(buy_candidates))
            if top_n >= 2:
                candidate_codes = [c['code'] for c in buy_candidates[:top_n]]
                ml_probs = {c['code']: c['prob'] for c in buy_candidates[:top_n]}
                try:
                    industry_map = {code: all_stock_data.get(code, {}).get('industry', '') for code in candidate_codes}
                    weights, valid_codes = portfolio_optimizer.optimize(
                        candidate_codes, all_stock_data, oamv_state_df, latest_date, ml_probs,
                        industry_map=industry_map
                    )
                    if len(valid_codes) > 0 and len(weights) > 0:
                        for code, w in zip(valid_codes, weights):
                            for c in buy_candidates:
                                if c['code'] == code:
                                    c['mvo_weight'] = w
                                    break
                        buy_candidates.sort(key=lambda x: -x.get('mvo_weight', 0))
                        print(f"  MVO优化完成: {len(valid_codes)} 只股票")
                except Exception:
                    print(f"  MVO优化失败，按概率排序")

            available_slots = MAX_PORTFOLIO_STOCKS - portfolio.get_position_count() + len(sell_signals)

            for candidate in buy_candidates[:available_slots]:
                buy_signals.append({
                    'code': candidate['code'],
                    'name': candidate['name'],
                    'action': 'BUY',
                    'price': candidate['price'],
                    'prob': candidate['prob'],
                    'atr': candidate['atr'],
                    'stop_loss': candidate['stop_loss'],
                    'industry': candidate['industry'],
                    'mvo_weight': candidate.get('mvo_weight'),
                    'impact_slippage': candidate.get('impact_slippage', SLIPPAGE_RATE),
                })

            print(f"  候选股: {len(buy_candidates)}, 买入信号: {len(buy_signals)}")

    signals = sell_signals + buy_signals

    print("\n" + "=" * 90)
    print("[Step 4] 交易信号汇总")
    print("=" * 90)

    if not signals:
        print("\n  今日无交易信号。")
    else:
        print()
        for sig in signals:
            action = sig['action']
            if action == 'SELL':
                print(f"  [卖出] {sig['code']} {sig['name']}")
                print(f"         价格: {sig['price']:.2f} | 原因: {sig['reason']}")
                print(f"         收益: {sig['profit_pct']:+.2f}% | 持仓: {sig['hold_days']}天")
            elif action == 'BUY':
                mvo_str = f" | MVO权重: {sig['mvo_weight']:.2f}" if sig.get('mvo_weight') is not None else ""
                impact_str = f" | 冲击滑点: {sig['impact_slippage']:.2%}" if sig.get('impact_slippage') else ""
                print(f"  [买入] {sig['code']} {sig['name']} ({sig['industry']})")
                print(f"         价格: {sig['price']:.2f} | RADE概率: {sig['prob']:.1%}{mvo_str}{impact_str}")
                print(f"         止损: {sig['stop_loss']:.2f} | ATR: {sig['atr']:.2f}")
            print()

    print("=" * 90)
    print("[Step 4.5] LLM中长期分析")
    print("-" * 90)

    llm_analyzer = LLMStockAnalyzer()
    llm_analyses = {}

    analyze_stocks = []
    for ts_code, pos in portfolio.positions.items():
        if ts_code in all_stock_data:
            analyze_stocks.append({
                'ts_code': ts_code,
                'name': pos['name'],
                'holding_info': None,
                'buy_signal': None,
            })
    for sig in buy_signals:
        already = any(s['ts_code'] == sig['code'] for s in analyze_stocks)
        if not already:
            analyze_stocks.append({
                'ts_code': sig['code'],
                'name': sig['name'],
                'holding_info': None,
                'buy_signal': sig,
            })

    for stock in analyze_stocks:
        ts_code = stock['ts_code']
        name = stock['name']
        data = all_stock_data.get(ts_code)
        if data is None:
            continue
        try:
            analysis = llm_analyzer.analyze_stock(
                ts_code, name, data,
                holding_info=stock.get('holding_info'),
                buy_signal=stock.get('buy_signal'),
            )
            if analysis:
                llm_analyses[ts_code] = analysis
                print(f"  {ts_code} {name} 分析完成")
            else:
                print(f"  {ts_code} {name} 分析为空")
        except Exception as e:
            print(f"  {ts_code} {name} 分析失败: {e}")

    print(f"  LLM分析完成: {len(llm_analyses)}/{len(analyze_stocks)} 只")

    print("=" * 90)
    print("[Step 5] 执行模式选择")
    print("-" * 90)

    auto_mode = '--auto' in sys.argv or '--execute' in sys.argv
    dry_run = '--dry-run' in sys.argv
    broker = os.getenv('EASYTRADER_BROKER', 'yh')
    exe_path = os.getenv('EASYTRADER_EXE_PATH', '')

    if auto_mode:
        auto_trader = AutoTrader(broker=broker, exe_path=exe_path if exe_path else None, dry_run=dry_run)
        print(f"\n  >>> 全自动执行模式 ({'模拟' if dry_run else '实盘'}) <<<")
        print(f"  券商: {AutoTrader.BROKER_MAP.get(broker, broker)}")

        for sig in sell_signals:
            code = sig['code']
            price = sig['price']
            date_str = latest_date.strftime('%Y-%m-%d')
            reason = sig['reason']
            trade = portfolio.remove_position(code, price, date_str, reason)
            if trade:
                pos_info = portfolio.positions.get(code, {})
                sell_amount = pos_info.get('amount', int(portfolio.initial_cash * POSITION_SIZE_PCT / price / 100) * 100)
                auto_trader.sell_all(code, sig['name'], price, reason)
                print(f"  [已卖出] {code} {sig['name']} @ {price:.2f}, 原因={reason}, "
                      f"收益={trade['profit_pct']:+.2f}%")

        for sig in buy_signals:
            if not portfolio.can_buy():
                break
            code = sig['code']
            name = sig['name']
            price = sig['price']
            prob = sig['prob']
            atr = sig['atr']
            date_str = latest_date.strftime('%Y-%m-%d')
            mvo_weight = sig.get('mvo_weight')
            buy_amount = int(portfolio.initial_cash * POSITION_SIZE_PCT / price / 100) * 100
            success = portfolio.add_position(code, name, price, prob, atr, date_str, mvo_weight, impact_slippage=sig.get('impact_slippage', SLIPPAGE_RATE))
            if success:
                auto_trader.buy(code, name, price, buy_amount, f"RADE={prob:.1%}")
                mvo_str = f"  MVO={mvo_weight:.2f}" if mvo_weight else ""
                print(f"  [已买入] {code} {name} @ {price:.2f}, RADE概率={prob:.1%}{mvo_str}")
            else:
                print(f"  [买入失败] {code} {name} - 资金不足或仓位已满")

        portfolio.save()
        print(f"\n  组合状态已保存")

        current_prices = {}
        for ts_code in portfolio.positions:
            if ts_code in all_stock_data:
                current_prices[ts_code] = float(all_stock_data[ts_code]['data']['Close'].iloc[-1])
            else:
                current_prices[ts_code] = portfolio.positions[ts_code]['entry_price']

        total_value = portfolio.get_total_value(current_prices)
        total_return = (total_value - portfolio.initial_cash) / portfolio.initial_cash * 100
        print(f"\n  总资产: {total_value:,.2f}")
        print(f"  总收益: {total_return:+.2f}%")
        print(f"  现金: {portfolio.cash:,.2f}")
        print(f"  持仓数: {portfolio.get_position_count()}/{MAX_PORTFOLIO_STOCKS}")

        if SERVERCHAN_KEYS:
            push_title = f"v9.0自动交易 {latest_date.strftime('%m-%d')}"
            push_msg = f"## v9.0 自动交易执行报告\n\n"
            push_msg += f"**模式**: {'模拟' if dry_run else '实盘'}\n\n"
            if sell_signals:
                push_msg += "### 卖出\n"
                for sig in sell_signals:
                    push_msg += f"- {sig['name']}({sig['code']}) @ {sig['price']:.2f} 原因:{sig['reason']}\n"
            if buy_signals:
                push_msg += "### 买入\n"
                for sig in buy_signals:
                    push_msg += f"- {sig['name']}({sig['code']}) @ {sig['price']:.2f} RADE={sig['prob']:.1%}\n"
            push_msg += f"\n总资产: {total_value:,.0f} | 收益: {total_return:+.2f}%"
            send_serverchan(push_title, push_msg)

    else:
        print("  半自动模式: 请根据上述信号手动在券商APP中操作")
        print("  全自动模式: 输入 'execute' 自动记录所有信号")
        print("  命令行全自动: python live_trader.py --auto --dry-run")
        print()

        mode = input("  请选择模式 (半自动=直接回车 / 全自动=execute): ").strip().lower()

        if mode == 'execute':
            print("\n  >>> 全自动执行模式 <<<")

            for sig in sell_signals:
                code = sig['code']
                price = sig['price']
                date_str = latest_date.strftime('%Y-%m-%d')
                reason = sig['reason']
                trade = portfolio.remove_position(code, price, date_str, reason)
                if trade:
                    print(f"  [已卖出] {code} {sig['name']} @ {price:.2f}, 原因={reason}, "
                          f"收益={trade['profit_pct']:+.2f}%")

            for sig in buy_signals:
                if not portfolio.can_buy():
                    break
                code = sig['code']
                name = sig['name']
                price = sig['price']
                prob = sig['prob']
                atr = sig['atr']
                date_str = latest_date.strftime('%Y-%m-%d')
                mvo_weight = sig.get('mvo_weight')
                success = portfolio.add_position(code, name, price, prob, atr, date_str, mvo_weight, impact_slippage=sig.get('impact_slippage', SLIPPAGE_RATE))
                if success:
                    mvo_str = f"  MVO={mvo_weight:.2f}" if mvo_weight else ""
                    print(f"  [已买入] {code} {name} @ {price:.2f}, RADE概率={prob:.1%}{mvo_str}")
                else:
                    print(f"  [买入失败] {code} {name} - 资金不足或仓位已满")

            portfolio.save()
            print(f"\n  组合状态已保存")

            current_prices = {}
            for ts_code in portfolio.positions:
                if ts_code in all_stock_data:
                    current_prices[ts_code] = float(all_stock_data[ts_code]['data']['Close'].iloc[-1])
                else:
                    current_prices[ts_code] = portfolio.positions[ts_code]['entry_price']

            total_value = portfolio.get_total_value(current_prices)
            total_return = (total_value - portfolio.initial_cash) / portfolio.initial_cash * 100
            print(f"\n  总资产: {total_value:,.2f}")
            print(f"  总收益: {total_return:+.2f}%")
            print(f"  现金: {portfolio.cash:,.2f}")
            print(f"  持仓数: {portfolio.get_position_count()}/{MAX_PORTFOLIO_STOCKS}")

        else:
            print("\n  >>> 半自动模式 <<<")
            print("  请在券商APP中手动执行上述信号。")
            print("  执行后请运行: python live_trader.py update")
            print("  手动更新持仓状态。")

    signal_data = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S'),
        'version': 'v9.0',
        'oamv_daily': 'BULL' if daily_ok else 'BEAR',
        'oamv_weekly': 'BULL' if weekly_ok else 'BEAR',
        'oamv_x': latest_x,
        'panic': is_panic,
        'signals': signals,
        'portfolio': {
            'cash': portfolio.cash,
            'positions': portfolio.positions,
            'position_count': portfolio.get_position_count(),
        },
        'llm_analyses': llm_analyses,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    signal_file = SIGNAL_DIR / f"signal_{timestamp}.json"
    with open(signal_file, 'w', encoding='utf-8') as f:
        json.dump(signal_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n  信号文件已保存: {signal_file}")

    if portfolio.trade_history:
        with open(TRADE_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(portfolio.trade_history, f, ensure_ascii=False, indent=2, default=str)
        print(f"  交易日志已保存: {TRADE_LOG_FILE}")

    if drift_detector.drift_detected:
        drift_alert = f"[ALERT] 检测到市场概念漂移（ADDM触发）！系统已启动紧急重新训练... 漂移次数: {drift_detector.drift_count}"
        print(f"\n  {drift_alert}")
        if SERVERCHAN_KEYS:
            send_serverchan(f"[ALERT] ADDM漂移检测 {latest_date.strftime('%Y-%m-%d')}", drift_alert)

    if signals and SERVERCHAN_KEYS:
        push_title = f"v9.0交易信号 {latest_date.strftime('%Y-%m-%d')}"
        push_desp = "## 交易信号\n\n"
        for sig in signals:
            if sig['action'] == 'SELL':
                push_desp += f"- **卖出** {sig['code']} {sig['name']} @ {sig['price']:.2f} 原因={sig['reason']} 收益={sig['profit_pct']:+.2f}%\n"
            elif sig['action'] == 'BUY':
                mvo_str = f" MVO={sig['mvo_weight']:.2f}" if sig.get('mvo_weight') else ""
                impact_str = f" 冲击滑点={sig['impact_slippage']:.2%}" if sig.get('impact_slippage') else ""
                push_desp += f"- **买入** {sig['code']} {sig['name']} @ {sig['price']:.2f} RADE={sig['prob']:.1%}{mvo_str}{impact_str}\n"
        if llm_analyses:
            push_desp += "\n### AI中长期分析（DeepSeek）\n\n"
            for ts_code, analysis in llm_analyses.items():
                pos_name = portfolio.positions.get(ts_code, {}).get('name', ts_code)
                push_desp += f"**{ts_code} {pos_name}**\n\n{analysis}\n\n"
        send_serverchan(push_title, push_desp)

    print("\n" + "=" * 90)
    print("v9.0 实盘交易系统运行完毕")
    print("=" * 90)


def show_portfolio_status():
    portfolio = LivePortfolio.load()
    print("=" * 60)
    print("v9.0 实盘持仓状态")
    print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"  初始资金: {portfolio.initial_cash:,.0f}")
    print(f"  当前现金: {portfolio.cash:,.2f}")
    print(f"  持仓数: {portfolio.get_position_count()}/{MAX_PORTFOLIO_STOCKS}")

    if portfolio.positions:
        print(f"\n  当前持仓:")
        current_prices = {}
        for ts_code, pos in portfolio.positions.items():
            try:
                df = get_stock_daily(ts_code, '20240101')
                if df is not None and len(df) > 0:
                    current_prices[ts_code] = float(df['Close'].iloc[-1])
                else:
                    current_prices[ts_code] = pos['entry_price']
            except Exception:
                current_prices[ts_code] = pos['entry_price']

        total_value = portfolio.cash
        for ts_code, pos in portfolio.positions.items():
            price = current_prices.get(ts_code, pos['entry_price'])
            profit_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
            value = pos['shares'] * price
            total_value += value
            mvo_str = f"  MVO={pos['mvo_weight']:.2f}" if pos.get('mvo_weight') else ""
            print(f"    {ts_code} {pos['name']}: {pos['shares']}股 @ {pos['entry_price']:.2f} -> {price:.2f} "
                  f"({profit_pct:+.2f}%) = {value:,.0f}{mvo_str}")

        total_return = (total_value - portfolio.initial_cash) / portfolio.initial_cash * 100
        print(f"\n  总资产: {total_value:,.2f}")
        print(f"  总收益: {total_return:+.2f}%")

    if portfolio.trade_history:
        print(f"\n  历史交易: {len(portfolio.trade_history)} 笔")
        wins = [t for t in portfolio.trade_history if t['profit_pct'] > 0]
        print(f"  胜率: {len(wins)/len(portfolio.trade_history)*100:.1f}%")
        if wins:
            print(f"  平均盈利: {np.mean([t['profit_pct'] for t in wins]):.2f}%")
        losses = [t for t in portfolio.trade_history if t['profit_pct'] <= 0]
        if losses:
            print(f"  平均亏损: {np.mean([t['profit_pct'] for t in losses]):.2f}%")

    print("=" * 60)


def manual_update():
    portfolio = LivePortfolio.load()
    print("=" * 60)
    print("手动更新持仓")
    print("=" * 60)
    print(f"  当前持仓: {portfolio.get_position_count()}/{MAX_PORTFOLIO_STOCKS}")
    for code, pos in portfolio.positions.items():
        print(f"    {code} {pos['name']}: {pos['shares']}股 @ {pos['entry_price']:.2f}")

    print("\n  操作选项:")
    print("  1. 手动买入")
    print("  2. 手动卖出")
    print("  3. 修改资金")
    print("  4. 清空所有持仓")
    print("  5. 重置账户")

    choice = input("\n  请选择 (1-5): ").strip()

    if choice == '1':
        code = input("  股票代码 (如 000001.SZ): ").strip()
        name = input("  股票名称: ").strip()
        price = float(input("  买入价格: ").strip())
        prob = float(input("  RADE概率 (如 0.72): ").strip())
        atr = float(input("  ATR14 (如 0.5): ").strip())
        date_str = datetime.now().strftime('%Y-%m-%d')
        success = portfolio.add_position(code, name, price, prob, atr, date_str)
        if success:
            print(f"  买入成功: {code} {name}")
        else:
            print(f"  买入失败: 资金不足或仓位已满")

    elif choice == '2':
        code = input("  股票代码: ").strip()
        price = float(input("  卖出价格: ").strip())
        reason = input("  卖出原因: ").strip()
        date_str = datetime.now().strftime('%Y-%m-%d')
        trade = portfolio.remove_position(code, price, date_str, reason)
        if trade:
            print(f"  卖出成功: {code} {trade['name']}, 收益={trade['profit_pct']:+.2f}%")
        else:
            print(f"  卖出失败: 未持有 {code}")

    elif choice == '3':
        new_cash = float(input("  新的资金金额: ").strip())
        portfolio.cash = new_cash
        print(f"  资金已更新: {portfolio.cash:,.2f}")

    elif choice == '4':
        confirm = input("  确认清空所有持仓? (yes/no): ").strip().lower()
        if confirm == 'yes':
            for code in list(portfolio.positions.keys()):
                pos = portfolio.positions[code]
                portfolio.remove_position(code, pos['entry_price'],
                                         datetime.now().strftime('%Y-%m-%d'), 'manual_clear')
            print("  所有持仓已清空")

    elif choice == '5':
        confirm = input("  确认重置账户? (yes/no): ").strip().lower()
        if confirm == 'yes':
            portfolio = LivePortfolio()
            print("  账户已重置")

    portfolio.save()
    print(f"\n  状态已保存")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 检查是否有 status/update/reset 命令
        if 'status' in sys.argv:
            show_portfolio_status()
        elif 'update' in sys.argv:
            manual_update()
        elif 'reset' in sys.argv:
            portfolio = LivePortfolio()
            portfolio.save()
            print("账户已重置")
        else:
            # 否则运行 live trader，参数会在 run_live_trader 中处理
            run_live_trader()
    else:
        run_live_trader()
