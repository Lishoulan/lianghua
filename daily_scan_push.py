import os
import sys
import gc
import json
import time
import warnings
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List

os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

sys.path.insert(0, str(Path(__file__).parent))
sys.path = [p for p in sys.path if not p.endswith('pip_libs')]

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
from ml_strategy.drift_detector import ADDMDriftDetector
from ml_strategy.rade_ensemble import RADEEnsemble
from ml_strategy.sterile_cleaner import SterileDataCleaner
from ml_strategy.disagreement_features import DisagreementFeatureBuilder
from ml_strategy.cost_aware_optimizer import CostAwarePortfolioOptimizer
from ml_strategy.path_signature import PathSignatureBuilder
from ml_strategy.llm_analyzer import LLMStockAnalyzer


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
SERVERCHAN_KEYS = [k.strip() for k in os.getenv('SERVERCHAN_KEY', '').split(',') if k.strip()]

pro = None
if TUSHARE_TOKEN:
    try:
        pro = ts.pro_api(TUSHARE_TOKEN)
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api(timeout=120)
    except Exception:
        pass

OAMV_UPPER = 4.0
OAMV_LOWER = -2.3
CATBOOST_BUY_THRESHOLD = 0.65
MAX_PORTFOLIO_STOCKS = 3
COMMISSION_RATE = 0.0003
STAMP_DUTY_RATE = 0.001
SLIPPAGE_RATE = 0.0005
POSITION_SIZE_PCT = 0.25
MIN_HOLD_DAYS = 3
COOLDOWN_DAYS = 5
MIN_PRICE = 3.0
TRAIN_WINDOW_MONTHS = 12
FRICTION_COST_PCT = (COMMISSION_RATE * 2 + STAMP_DUTY_RATE + SLIPPAGE_RATE * 2) * 100
MIN_AMPLITUDE_MULT = 1.5
TRAILING_STOP_PCT = 8.0
ATR_STOP_MULT = 1.5
STOCK_LIMIT = 500
IMPACT_COEFFICIENT = 0.4
SPREAD_HALF = 0.001
MAX_SLIPPAGE_PCT = 2.0
INITIAL_CASH = 500000

PWVC_VETO_THRESHOLD = 0.8
J_OVERSOLD_THRESHOLD = 13

PORTFOLIO_DIR = Path(__file__).parent / "portfolio_state"
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_FILE = PORTFOLIO_DIR / "portfolio.json"
TRADE_LOG_FILE = PORTFOLIO_DIR / "trade_log.json"


def is_st_stock(name: str) -> bool:
    for kw in ['ST', '*ST', 'S*ST', 'SST']:
        if name.startswith(kw):
            return True
    return False


class ScanPortfolio:
    def __init__(self, initial_cash=INITIAL_CASH):
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
    def load(cls) -> 'ScanPortfolio':
        if not PORTFOLIO_FILE.exists():
            return cls()
        try:
            with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            portfolio = cls(data.get('initial_cash', INITIAL_CASH))
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

    def get_available_slots(self) -> int:
        return MAX_PORTFOLIO_STOCKS - self.get_position_count()

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        total = self.cash
        for ts_code, pos in self.positions.items():
            price = current_prices.get(ts_code, pos.get('entry_price', 0))
            total += pos.get('shares', 0) * price
        return total

    def add_position(self, ts_code, name, price, prob, atr, date_str,
                     mvo_weight=None, impact_slippage=SLIPPAGE_RATE):
        target_value = self.initial_cash * POSITION_SIZE_PCT
        target_value = min(target_value, self.cash * 0.9)
        shares = int(target_value / price / 100) * 100
        if shares <= 0:
            return False
        amount = shares * price
        commission = max(amount * COMMISSION_RATE, 5.0)
        slippage = amount * impact_slippage
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


def get_index_daily(ts_code='000001.SH', start_date='20210101', end_date='20260520'):
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


def get_industry_daily(ts_code, start_date='20210101', end_date='20260520'):
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


def get_stock_daily(ts_code, start_date='20210101', end_date='20260520'):
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
    df['white_above_yellow'] = df['white_line'] > df['yellow_line']
    low_list = df['Low'].rolling(window=9, min_periods=1).min()
    high_list = df['High'].rolling(window=9, min_periods=1).max()
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)
    df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df['D_val'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D_val']
    df['amplitude_20'] = ((df['High'] - df['Low']) / df['Close'] * 100).rolling(window=20).mean()
    return df


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


def send_serverchan(title, desp):
    if not SERVERCHAN_KEYS:
        print("SERVERCHAN_KEY未配置，跳过推送")
        return False
    all_ok = True
    for idx, key in enumerate(SERVERCHAN_KEYS):
        url = f"https://sctapi.ftqq.com/{key}.send"
        data = {"title": title, "desp": desp}
        label = f"Key#{idx+1}" if len(SERVERCHAN_KEYS) > 1 else ""
        try:
            resp = requests.post(url, data=data, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                if result.get('code') == 0:
                    print(f"Server酱推送成功 {label}".strip())
                else:
                    print(f"Server酱推送失败 {label}: {result}")
                    all_ok = False
            else:
                print(f"Server酱HTTP错误 {label}: {resp.status_code}")
                all_ok = False
        except Exception as e:
            print(f"Server酱推送异常 {label}: {e}")
            all_ok = False
    return all_ok


def fetch_stock_news(ts_code, max_items=5):
    try:
        import akshare as ak
        symbol = ts_code.split('.')[0]
        news_df = ak.stock_news_em(symbol=symbol)
        if news_df is not None and len(news_df) > 0:
            titles = news_df['新闻标题'].tolist()[:max_items] if '新闻标题' in news_df.columns else []
            if not titles and '标题' in news_df.columns:
                titles = news_df['标题'].tolist()[:max_items]
            return titles
    except Exception:
        pass
    return []


def check_sell_conditions(ts_code, pos, df, last_date, oamv_daily, market_state):
    if last_date not in df.index:
        return None, 0, 0
    row = df.loc[last_date]
    current_price = float(row['Close'])
    entry_price = pos.get('entry_price', 0)
    peak_price = pos.get('peak_price', entry_price)

    if current_price > peak_price:
        pos['peak_price'] = current_price
        peak_price = current_price

    profit_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
    dd_pct = (peak_price - current_price) / peak_price * 100 if peak_price > 0 else 0
    hold_days = (pd.Timestamp(last_date) - pd.Timestamp(pos.get('entry_date', last_date))).days

    sell_reason = None

    if market_state == 'panic':
        sell_reason = '熔断器触发'
    elif not oamv_daily:
        sell_reason = '0AMV日线BEAR'

    if sell_reason is None and hold_days >= MIN_HOLD_DAYS:
        yellow_line = row.get('yellow_line')
        if yellow_line is not None and not pd.isna(yellow_line):
            if current_price < yellow_line:
                sell_reason = f'收盘<{yellow_line:.2f}(黄线)'

        if sell_reason is None and dd_pct >= TRAILING_STOP_PCT:
            sell_reason = f'峰值回撤{dd_pct:.1f}%≥{TRAILING_STOP_PCT}%'

        if sell_reason is None:
            entry_atr = pos.get('entry_atr', 0)
            if entry_atr > 0 and peak_price > entry_price:
                dd_atr = (peak_price - current_price) / entry_atr
                if dd_atr >= ATR_STOP_MULT:
                    sell_reason = f'ATR止损{dd_atr:.1f}x≥{ATR_STOP_MULT}x'

    return sell_reason, profit_pct, dd_pct


def run_daily_scan(mode='afternoon'):
    today = datetime.now().strftime('%Y%m%d')
    today_display = datetime.now().strftime('%Y-%m-%d')

    mode_label = {'morning': '早盘', 'afternoon': '尾盘', 'evening': '盘后'}.get(mode, mode)
    print(f"[{today_display}] v9.1 {mode_label}模式 每日扫描 + 带盘推送 启动...")

    portfolio = ScanPortfolio.load()
    print(f"  初始资金: {portfolio.initial_cash:,.0f}")
    print(f"  当前现金: {portfolio.cash:,.2f}")
    print(f"  当前持仓: {portfolio.get_position_count()}/{MAX_PORTFOLIO_STOCKS}")
    if portfolio.positions:
        for code, pos in portfolio.positions.items():
            print(f"    {code} {pos['name']}: {pos['shares']}股 @ {pos['entry_price']:.2f}")

    print("加载行业数据...")
    industry_j_cache = {}
    for ind_name, sw_code in INDUSTRY_MAP.items():
        try:
            ind_df = get_industry_daily(sw_code, end_date=today)
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
                ind_df = get_industry_daily(etf_code, end_date=today)
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

    print("加载股票数据...")
    stock_basic = pro.stock_basic(exchange='', list_status='L',
                                   fields='ts_code,symbol,name,industry,list_date')
    a_stocks = stock_basic[
        (stock_basic['ts_code'].str.endswith('.SH')) |
        (stock_basic['ts_code'].str.endswith('.SZ'))
    ]
    a_stocks = a_stocks[~a_stocks['name'].apply(is_st_stock)]
    a_stocks = a_stocks.head(STOCK_LIMIT)

    index_df = get_index_daily('000001.SH', end_date=today)
    if index_df is None:
        return
    index_df = compute_indicators(index_df)
    market_vol = 0.02
    if 'ATR14' in index_df.columns and len(index_df) > 0:
        market_atr = float(index_df['ATR14'].iloc[-1])
        market_close = float(index_df['Close'].iloc[-1])
        market_vol = market_atr / market_close if market_close > 0 else 0.02
        drift_detector.set_market_volatility(market_vol)

    all_stock_data = {}
    all_featured_data = {}
    discretizer = FeatureDiscretizer()
    total = len(a_stocks)

    for idx, (_, row) in enumerate(a_stocks.iterrows(), 1):
        ts_code = row['ts_code']
        name = row['name']
        industry = row.get('industry', '')
        if idx % 50 == 0:
            print(f"  进度: {idx}/{total}")
        try:
            df = get_stock_daily(ts_code, end_date=today)
            if df is None or len(df) < 200:
                continue
            if df['Close'].iloc[-1] < MIN_PRICE:
                continue
            df = compute_indicators(df)
            if df is None:
                continue
            all_stock_data[ts_code] = {'data': df, 'name': name, 'industry': industry}
            feature_df_raw = sterile_cleaner.get_feature_dataframe(df, ts_code)
            featured_df = discretizer.transform(feature_df_raw)
            ind_j = industry_j_cache.get(industry)
            featured_df = discretizer.add_market_context(featured_df, index_df, None, ind_j)
            featured_df = disagreement_builder.build_features(featured_df)
            featured_df = ssa_denoiser.denoise_features(featured_df)
            featured_df = discretizer.add_path_signatures(featured_df, sig_builder)
            all_featured_data[ts_code] = featured_df
        except Exception:
            continue
        if idx % 100 == 0:
            gc.collect()

    print(f"  加载: {len(all_stock_data)} 只")

    print("0AMV过滤器...")
    oamv_filter = OAMVHysteresisFilter(
        upper_threshold=OAMV_UPPER, lower_threshold=OAMV_LOWER,
        cost_ma_period=34, weekly_ema_period=5, weekly_use_ema=True,
    )
    oamv_filter.fit(index_df, all_stock_data=all_stock_data)

    print("熔断器...")
    panic_breaker = MarketPanicCircuitBreaker(breadth_threshold=0.85, limit_down_threshold=150, ma_period=20)
    panic_breaker.compute_market_breadth(all_stock_data)

    print("训练RADE模型...")
    from dateutil.relativedelta import relativedelta
    train_end = today
    train_start_dt = pd.Timestamp(train_end) - relativedelta(months=TRAIN_WINDOW_MONTHS)
    train_start = train_start_dt.strftime('%Y%m%d')

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
        print("样本不足")
        return

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

    kan_trainer = None
    try:
        kan_trainer = ChebyKANTrainer(input_dim=len(available_cols), hidden_dim=16, poly_degree=4, lr=0.005, epochs=200, batch_size=256)
        kan_trainer.train(X_train, y_train, X_val, y_val)
    except Exception:
        kan_trainer = None

    ensemble = RADEEnsemble(gamma=0.5)
    ensemble.set_models(catboost_model, kan_trainer)

    print("扫描今日信号...")
    oamv_state_df = oamv_filter.get_state_df()

    last_date = index_df.index[-1]
    date_str = last_date.strftime('%Y-%m-%d')
    oamv_daily = oamv_filter.is_trading_allowed(last_date, require_weekly=False)
    oamv_weekly = oamv_filter.is_trading_allowed(last_date, require_weekly=True)
    market_state = panic_breaker.get_market_state(last_date)
    is_panic = market_state == 'panic'

    oamv_x = 0.0
    if last_date in oamv_state_df.index:
        oamv_x = float(oamv_state_df.loc[last_date, 'oamv_x'])

    index_close = float(index_df.loc[last_date, 'Close'])
    index_change = 0.0
    if len(index_df) > 1:
        prev_close = float(index_df['Close'].iloc[-2])
        if prev_close > 0:
            index_change = (index_close - prev_close) / prev_close * 100

    # ========== Step 1: 检查持仓卖出条件 ==========
    print("\n[Step 1] 持仓风控检查...")
    sell_signals = []
    holding_status = []

    for ts_code in list(portfolio.positions.keys()):
        pos = portfolio.positions[ts_code]
        if ts_code not in all_stock_data:
            holding_status.append({
                'ts_code': ts_code, 'name': pos['name'],
                'current_price': pos['entry_price'], 'entry_price': pos['entry_price'],
                'profit_pct': 0, 'dd_pct': 0, 'hold_days': 0,
                'status': '⚠️ 数据缺失', 'action': '持有',
            })
            continue
        df = all_stock_data[ts_code]['data']
        sell_reason, profit_pct, dd_pct = check_sell_conditions(
            ts_code, pos, df, last_date, oamv_daily, market_state)
        current_price = float(df.loc[last_date, 'Close']) if last_date in df.index else pos['entry_price']
        hold_days = (pd.Timestamp(date_str) - pd.Timestamp(pos.get('entry_date', date_str))).days

        if sell_reason:
            sell_signals.append({
                'ts_code': ts_code, 'name': pos['name'],
                'price': current_price, 'reason': sell_reason,
                'profit_pct': profit_pct, 'hold_days': hold_days,
                'entry_price': pos['entry_price'],
            })
            holding_status.append({
                'ts_code': ts_code, 'name': pos['name'],
                'current_price': current_price, 'entry_price': pos['entry_price'],
                'profit_pct': profit_pct, 'dd_pct': dd_pct, 'hold_days': hold_days,
                'status': f'🔴 {sell_reason}', 'action': '卖出',
            })
        else:
            status_icon = '🟢'
            action = '持有'
            if dd_pct >= TRAILING_STOP_PCT * 0.6:
                status_icon = '🟡'
                action = '关注'
            holding_status.append({
                'ts_code': ts_code, 'name': pos['name'],
                'current_price': current_price, 'entry_price': pos['entry_price'],
                'profit_pct': profit_pct, 'dd_pct': dd_pct, 'hold_days': hold_days,
                'status': f'{status_icon} 正常', 'action': action,
            })

    # ========== Step 2: 执行卖出 ==========
    print(f"[Step 2] 执行卖出 ({len(sell_signals)} 只)...")
    for sig in sell_signals:
        trade = portfolio.remove_position(sig['ts_code'], sig['price'], date_str, sig['reason'])
        if trade:
            print(f"  [卖出] {sig['ts_code']} {sig['name']} @ {sig['price']:.2f} "
                  f"收益={trade['profit_pct']:+.2f}% 原因={sig['reason']}")

    # ========== Step 3: 扫描买入信号 ==========
    buy_signals = []
    if mode == 'morning':
        print("[Step 3] 早盘模式：跳过买入扫描")
    elif mode == 'evening':
        print("[Step 3] 盘后模式：跳过买入扫描")
    else:
        print(f"[Step 3] 扫描买入信号...")
        if oamv_weekly and market_state != 'panic' and market_state != 'warning':
            for ts_code, info in all_stock_data.items():
                if ts_code in portfolio.positions:
                    continue
                if ts_code in portfolio.recent_sells:
                    sell_date = portfolio.recent_sells[ts_code]
                    if (pd.Timestamp(date_str) - pd.Timestamp(sell_date)).days < COOLDOWN_DAYS:
                        continue
                df = info['data']
                featured_df = all_featured_data[ts_code]
                if last_date not in df.index or last_date not in featured_df.index:
                    continue
                feat_row = featured_df.loc[last_date]
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

                prob = ensemble.predict(clean_data, oamv_x_pct=oamv_x, atr_ratio=atr_ratio)[0]

                if prob >= CATBOOST_BUY_THRESHOLD:
                    white_above = df.loc[last_date, 'white_above_yellow'] if 'white_above_yellow' in df.columns else True
                    if pd.notna(white_above) and not white_above:
                        continue

                    j_val = df.loc[last_date, 'J'] if 'J' in df.columns else 100
                    if pd.notna(j_val) and j_val >= J_OVERSOLD_THRESHOLD:
                        continue

                    pwvc_val = feat_row.get('pwvc', 0.0)
                    if pd.notna(pwvc_val) and pwvc_val > PWVC_VETO_THRESHOLD:
                        continue

                    amp = df.loc[last_date, 'amplitude_20'] if 'amplitude_20' in df.columns else 0
                    if pd.notna(amp) and amp > 0 and amp < FRICTION_COST_PCT * MIN_AMPLITUDE_MULT:
                        continue

                    current_price = float(df.loc[last_date, 'Close'])
                    current_vol = float(df.loc[last_date, 'Volume']) if 'Volume' in df.columns and pd.notna(df.loc[last_date, 'Volume']) else 0
                    daily_vol_value = current_price * current_vol * 100
                    order_value = portfolio.initial_cash * POSITION_SIZE_PCT
                    participation = order_value / daily_vol_value if daily_vol_value > 0 else 1.0

                    impact_slippage = (IMPACT_COEFFICIENT * np.sqrt(participation) + SPREAD_HALF) * 100
                    impact_slippage = min(impact_slippage, MAX_SLIPPAGE_PCT)

                    if impact_slippage > 1.0:
                        continue

                    buy_signals.append({
                        'ts_code': ts_code,
                        'name': info['name'],
                        'prob': prob,
                        'price': current_price,
                        'atr': float(df.loc[last_date, 'ATR14']) if 'ATR14' in df.columns and not pd.isna(df.loc[last_date, 'ATR14']) else 0,
                        'industry': info.get('industry', ''),
                        'impact_slippage': float(impact_slippage),
                        'j_val': float(j_val) if not pd.isna(j_val) else 0,
                        'pwvc': float(pwvc_val) if not pd.isna(pwvc_val) else 0,
                        'accumulation_score': float(feat_row.get('accumulation_score', 0)),
                    })

        buy_signals.sort(key=lambda x: -x['prob'])

        mvo_info = ""
        top_n = min(5, len(buy_signals))
        if top_n >= 2:
            candidate_codes = [c['ts_code'] for c in buy_signals[:top_n]]
            ml_probs = {c['ts_code']: c['prob'] for c in buy_signals[:top_n]}

            try:
                industry_map = {code: all_stock_data.get(code, {}).get('industry', '') for code in candidate_codes}
                weights, valid_codes = portfolio_optimizer.optimize(
                    candidate_codes, all_stock_data, oamv_state_df, last_date, ml_probs,
                    industry_map=industry_map
                )
                if len(valid_codes) > 0 and len(weights) > 0:
                    for code, w in zip(valid_codes, weights):
                        for s in buy_signals:
                            if s['ts_code'] == code:
                                s['mvo_weight'] = w
                                break
                    buy_signals.sort(key=lambda x: -x.get('mvo_weight', 0))
                    mvo_info = "MVO优化完成"
            except Exception:
                mvo_info = "MVO优化失败，按概率排序"

    # ========== Step 4: 执行买入 ==========
    executed_buys = []
    if mode != 'afternoon':
        print(f"[Step 4] {mode_label}模式：跳过买入执行")
    else:
        available_slots = portfolio.get_available_slots()
        print(f"[Step 4] 执行买入 (可用仓位: {available_slots})...")
        for sig in buy_signals:
            if available_slots <= 0:
                break
            success = portfolio.add_position(
                sig['ts_code'], sig['name'], sig['price'], sig['prob'],
                sig['atr'], date_str,
                mvo_weight=sig.get('mvo_weight'),
                impact_slippage=sig.get('impact_slippage', SLIPPAGE_RATE) / 100,
            )
            if success:
                executed_buys.append(sig)
                available_slots -= 1
                mvo_str = f" MVO={sig.get('mvo_weight', 0):.2f}" if sig.get('mvo_weight') else ""
                print(f"  [买入] {sig['ts_code']} {sig['name']} @ {sig['price']:.2f} "
                      f"RADE={sig['prob']:.1%}{mvo_str}")

    # ========== Step 5: 保存持仓 ==========
    portfolio.save()
    if portfolio.trade_history:
        with open(TRADE_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(portfolio.trade_history, f, ensure_ascii=False, indent=2, default=str)

    # ========== Step 6: LLM中长期分析 ==========
    print("[Step 6] LLM中长期分析...")
    llm_analyzer = LLMStockAnalyzer()
    llm_analyses = {}
    event_analyses = {}
    high_risk_codes = set()

    analyze_stocks = []
    for ts_code, pos in portfolio.positions.items():
        if ts_code in all_stock_data:
            h_info = None
            for h in holding_status:
                if h['ts_code'] == ts_code:
                    h_info = h
                    break
            analyze_stocks.append({
                'ts_code': ts_code,
                'name': pos['name'],
                'holding_info': h_info,
                'buy_signal': None,
            })
    for sig in executed_buys:
        already = any(s['ts_code'] == sig['ts_code'] for s in analyze_stocks)
        if not already:
            analyze_stocks.append({
                'ts_code': sig['ts_code'],
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
                print(f"  ✅ {ts_code} {name} 技术面分析完成")
            else:
                print(f"  ⚠️ {ts_code} {name} 技术面分析为空")
        except Exception as e:
            print(f"  ❌ {ts_code} {name} 技术面分析失败: {e}")

        try:
            news_list = fetch_stock_news(ts_code)
            if news_list:
                event_result = llm_analyzer.analyze_event(ts_code, name, news_list)
                if event_result:
                    event_analyses[ts_code] = event_result
                    if '风险：高' in event_result or '风险:高' in event_result:
                        high_risk_codes.add(ts_code)
                    print(f"  📰 {ts_code} {name} 事件分析完成")
                else:
                    print(f"  ⚠️ {ts_code} {name} 事件分析为空")
            else:
                print(f"  ⚠️ {ts_code} {name} 无新闻数据")
        except Exception as e:
            print(f"  ❌ {ts_code} {name} 事件分析失败: {e}")

    print(f"  LLM分析完成: 技术{len(llm_analyses)}/{len(analyze_stocks)} 事件{len(event_analyses)}/{len(analyze_stocks)}")

    if high_risk_codes:
        vetoed_buys = [s for s in executed_buys if s['ts_code'] in high_risk_codes]
        for sig in vetoed_buys:
            print(f"  🚫 {sig['ts_code']} {sig['name']} 事件高风险，取消买入")
            portfolio.remove_position(sig['ts_code'], sig['price'], date_str, '事件高风险否决')
            executed_buys.remove(sig)
        portfolio.save()

    # ========== Step 7: 计算组合概况 ==========
    current_prices = {}
    for ts_code in portfolio.positions:
        if ts_code in all_stock_data:
            df = all_stock_data[ts_code]['data']
            if last_date in df.index:
                current_prices[ts_code] = float(df.loc[last_date, 'Close'])
            else:
                current_prices[ts_code] = portfolio.positions[ts_code]['entry_price']
        else:
            current_prices[ts_code] = portfolio.positions[ts_code]['entry_price']

    total_value = portfolio.get_total_value(current_prices)
    total_return = (total_value - portfolio.initial_cash) / portfolio.initial_cash * 100
    position_value = sum(portfolio.positions[c].get('shares', 0) * current_prices.get(c, 0)
                         for c in portfolio.positions)

    # ========== Step 7: 生成推送消息 ==========
    oamv_status = "🟢 BULL" if oamv_daily else "🔴 BEAR"
    oamv_weekly_status = "🟢 BULL" if oamv_weekly else "🔴 BEAR"
    if market_state == 'panic':
        panic_status = "🔴 熔断"
    elif market_state == 'warning':
        panic_status = "⚠️ 预警"
    else:
        panic_status = "✅ 正常"

    title = f"v9.1{mode_label} {today_display} 持仓{portfolio.get_position_count()}仓 收益{total_return:+.1f}%"

    desp = f"## v9.1 {mode_label}带盘信号\n\n"
    desp += f"**日期**: {today_display}\n\n"

    desp += f"### 一、大势判断\n\n"
    desp += f"| 指标 | 状态 |\n|------|------|\n"
    desp += f"| 0AMV日线 | {oamv_status} (X={oamv_x:+.2f}%) |\n"
    desp += f"| 0AMV周线 | {oamv_weekly_status} |\n"
    desp += f"| 熔断器 | {panic_status} |\n"
    desp += f"| 上证指数 | {index_close:.2f} ({index_change:+.2f}%) |\n\n"

    desp += f"### 二、组合概况\n\n"
    desp += f"| 指标 | 数值 |\n|------|------|\n"
    desp += f"| 总资产 | {total_value:,.0f} |\n"
    desp += f"| 总收益 | {total_return:+.2f}% |\n"
    desp += f"| 现金 | {portfolio.cash:,.0f} |\n"
    desp += f"| 持仓市值 | {position_value:,.0f} |\n"
    desp += f"| 持仓数 | {portfolio.get_position_count()}/{MAX_PORTFOLIO_STOCKS} |\n"
    desp += f"| 可用仓位 | {portfolio.get_available_slots()} |\n\n"

    if portfolio.trade_history:
        wins = [t for t in portfolio.trade_history if t['profit_pct'] > 0]
        win_rate = len(wins) / len(portfolio.trade_history) * 100
        desp += f"| 历史交易 | {len(portfolio.trade_history)}笔 |\n"
        desp += f"| 历史胜率 | {win_rate:.0f}% |\n\n"

    if holding_status:
        desp += f"### 三、当前持仓\n\n"
        desp += f"| 代码 | 名称 | 买入价 | 现价 | 收益 | 回撤 | 持仓天数 | 状态 | 操作 |\n"
        desp += f"|------|------|--------|------|------|------|----------|------|------|\n"
        for h in holding_status:
            desp += (f"| {h['ts_code']} | {h['name']} | {h['entry_price']:.2f} | "
                     f"{h['current_price']:.2f} | {h['profit_pct']:+.2f}% | "
                     f"{h['dd_pct']:.1f}% | {h['hold_days']}天 | {h['status']} | {h['action']} |\n")
        desp += f"\n"

    if sell_signals:
        desp += f"### 四、🔴 今日卖出\n\n"
        desp += f"| 代码 | 名称 | 卖出价 | 买入价 | 收益 | 持仓天数 | 卖出原因 |\n"
        desp += f"|------|------|--------|--------|------|----------|----------|\n"
        for s in sell_signals:
            desp += (f"| {s['ts_code']} | {s['name']} | {s['price']:.2f} | "
                     f"{s['entry_price']:.2f} | {s['profit_pct']:+.2f}% | "
                     f"{s['hold_days']}天 | {s['reason']} |\n")
        desp += f"\n"

    if executed_buys:
        desp += f"### 五、🟢 今日买入\n\n"
        desp += f"| 代码 | 名称 | 买入价 | RADE | ATR | MVO权重 | 滑点 |\n"
        desp += f"|------|------|--------|------|-----|--------|------|\n"
        for s in executed_buys:
            mvo_w = f"{s['mvo_weight']:.2f}" if s.get('mvo_weight') is not None else "-"
            desp += (f"| {s['ts_code']} | {s['name']} | {s['price']:.2f} | "
                     f"{s['prob']:.1%} | {s['atr']:.2f} | {mvo_w} | {s['impact_slippage']:.3f}% |\n")
        desp += f"\n"

    if not oamv_weekly:
        desp += f"### 操作建议：空仓观望\n\n"
        desp += f"0AMV周线BEAR，不建议买入。等待周线翻多信号。\n\n"
    elif market_state == 'panic':
        desp += f"### 操作建议：🔴 熔断器触发\n\n"
        desp += f"市场出现恐慌信号，建议清仓避险。\n\n"
    elif market_state == 'warning':
        desp += f"### 操作建议：⚠️ 市场预警\n\n"
        desp += f"市场出现预警信号（跌停增速/宽度恶化），禁止新买入，持仓可正常止盈止损。\n\n"
    elif buy_signals and not executed_buys:
        desp += f"### 六、候选买入信号（仓位已满）\n\n"
        desp += f"当前{portfolio.get_position_count()}仓已满，以下候选股待仓位释放后关注：\n\n"
        desp += f"| # | 代码 | 名称 | 行业 | 现价 | RADE | J值 | PWVC | ATR |\n"
        desp += f"|---|------|------|------|------|------|-----|------|-----|\n"
        for i, s in enumerate(buy_signals[:5], 1):
            desp += (f"| {i} | {s['ts_code']} | {s['name']} | {s['industry']} | "
                     f"{s['price']:.2f} | {s['prob']:.1%} | {s['j_val']:.1f} | "
                     f"{s['pwvc']:.2f} | {s['atr']:.2f} |\n")
        desp += f"\n"
    elif buy_signals and len(executed_buys) < len(buy_signals):
        remaining = [s for s in buy_signals if s not in executed_buys]
        if remaining:
            desp += f"### 六、其他候选信号\n\n"
            desp += f"| # | 代码 | 名称 | 行业 | 现价 | RADE | J值 | ATR |\n"
            desp += f"|---|------|------|------|------|------|-----|-----|\n"
            for i, s in enumerate(remaining[:5], 1):
                desp += (f"| {i} | {s['ts_code']} | {s['name']} | {s['industry']} | "
                         f"{s['price']:.2f} | {s['prob']:.1%} | {s['j_val']:.1f} | "
                         f"{s['atr']:.2f} |\n")
            desp += f"\n{mvo_info}\n\n"

    desp += f"**四大共识过滤**: PWVC>{PWVC_VETO_THRESHOLD}否决 / 白>黄强制 / J<{J_OVERSOLD_THRESHOLD}冰点\n\n"

    if llm_analyses:
        desp += f"### 🤖 AI技术面分析（DeepSeek）\n\n"
        for ts_code, analysis in llm_analyses.items():
            pos_name = portfolio.positions.get(ts_code, {}).get('name', '')
            if not pos_name:
                for sig in executed_buys:
                    if sig['ts_code'] == ts_code:
                        pos_name = sig['name']
                        break
            if not pos_name:
                pos_name = ts_code
            desp += f"**{ts_code} {pos_name}**\n\n{analysis}\n\n"

    if event_analyses:
        desp += f"### 📰 AI事件面分析（DeepSeek）\n\n"
        for ts_code, analysis in event_analyses.items():
            pos_name = portfolio.positions.get(ts_code, {}).get('name', '')
            if not pos_name:
                for sig in executed_buys:
                    if sig['ts_code'] == ts_code:
                        pos_name = sig['name']
                        break
            if not pos_name:
                pos_name = ts_code
            risk_tag = " ⚠️高风险" if ts_code in high_risk_codes else ""
            desp += f"**{ts_code} {pos_name}{risk_tag}**\n\n{analysis}\n\n"

    drift_alert_msg = ""
    if drift_detector.drift_detected:
        drift_alert_msg = f"\n\n### ADDM漂移告警\n\n检测到市场概念漂移！漂移次数: {drift_detector.drift_count}\n"
        if SERVERCHAN_KEYS:
            send_serverchan(f"[ALERT] ADDM漂移检测 {today_display}", drift_alert_msg)

    desp += drift_alert_msg

    desp += f"---\n\n"
    desp += f"*v9.1 {mode_label}模式 | 自动卖出+自动建仓+Server酱推送 | 仅供参考，不构成投资建议*\n"

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"0AMV日线: {oamv_status} (X={oamv_x:+.2f}%)")
    print(f"0AMV周线: {oamv_weekly_status}")
    print(f"熔断器: {panic_status}")
    print(f"上证: {index_close:.2f} ({index_change:+.2f}%)")
    print(f"总资产: {total_value:,.0f} | 收益: {total_return:+.2f}%")
    print(f"现金: {portfolio.cash:,.0f} | 持仓: {portfolio.get_position_count()}/{MAX_PORTFOLIO_STOCKS}")

    if sell_signals:
        print(f"\n今日卖出 ({len(sell_signals)} 只):")
        for s in sell_signals:
            print(f"  🔴 {s['ts_code']} {s['name']} @ {s['price']:.2f} "
                  f"收益={s['profit_pct']:+.2f}% 原因={s['reason']}")

    if holding_status:
        print(f"\n当前持仓:")
        for h in holding_status:
            print(f"  {h['status']} {h['ts_code']} {h['name']} "
                  f"收益={h['profit_pct']:+.2f}% 回撤={h['dd_pct']:.1f}% {h['action']}")

    if executed_buys:
        print(f"\n今日买入 ({len(executed_buys)} 只):")
        for s in executed_buys:
            mvo_w = f" MVO={s['mvo_weight']:.2f}" if s.get('mvo_weight') else ""
            print(f"  🟢 {s['ts_code']} {s['name']} @ {s['price']:.2f} "
                  f"RADE={s['prob']:.1%}{mvo_w}")

    send_serverchan(title, desp)

    output_dir = Path(__file__).parent / "daily_scan_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"scan_v90_{today}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'version': 'v9.0',
            'date': today_display,
            'oamv_daily': oamv_daily,
            'oamv_weekly': oamv_weekly,
            'oamv_x': oamv_x,
            'is_panic': is_panic,
            'index_close': index_close,
            'index_change': index_change,
            'sell_signals': sell_signals,
            'executed_buys': executed_buys,
            'buy_signals': buy_signals[:10],
            'holding_status': holding_status,
            'portfolio': {
                'total_value': total_value,
                'total_return': total_return,
                'cash': portfolio.cash,
                'position_count': portfolio.get_position_count(),
                'positions': portfolio.positions,
            },
            'push_sent': bool(SERVERCHAN_KEYS),
            'drift_detected': drift_detector.drift_detected,
            'drift_count': drift_detector.drift_count,
            'market_volatility': market_vol,
            'consensus_filters': f'PWVC>{PWVC_VETO_THRESHOLD} / white_above_yellow / J<{J_OVERSOLD_THRESHOLD}',
            'llm_analyses': llm_analyses,
            'event_analyses': event_analyses,
            'high_risk_codes': list(high_risk_codes),
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {result_file}")


if __name__ == "__main__":
    mode = 'afternoon'
    if '--mode' in sys.argv:
        idx = sys.argv.index('--mode')
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]
    if mode not in ('morning', 'afternoon', 'evening'):
        print(f"未知模式: {mode}，支持: morning, afternoon, evening")
        sys.exit(1)
    run_daily_scan(mode=mode)
