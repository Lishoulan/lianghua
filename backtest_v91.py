import os
import sys
import gc
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List

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
from ml_strategy.drift_detector import ADDMDriftDetector
from ml_strategy.rade_ensemble import RADEEnsemble
from ml_strategy.sterile_cleaner import SterileDataCleaner
from ml_strategy.disagreement_features import DisagreementFeatureBuilder
from ml_strategy.cost_aware_optimizer import CostAwarePortfolioOptimizer
from ml_strategy.path_signature import PathSignatureBuilder
from ml_strategy.portfolio_backtest import PortfolioBacktester

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
pro = None
if TUSHARE_TOKEN:
    try:
        pro = ts.pro_api(TUSHARE_TOKEN)
        print("Tushare init OK")
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
POSITION_SIZE_PCT = 0.25
MIN_HOLD_DAYS = 3
COOLDOWN_DAYS = 5
MIN_PRICE = 3.0
TRAIN_WINDOW_MONTHS = 12
FRICTION_COST_PCT = (COMMISSION_RATE * 2 + STAMP_DUTY_RATE + SLIPPAGE_RATE * 2) * 100
MIN_AMPLITUDE_MULT = 1.5
TRAILING_STOP_PCT = 8.0
ATR_STOP_MULT = 1.5
SIM_DAYS = 120
STOCK_LIMIT = 200
END_DATE = '20260520'
API_SLEEP = 0.15

IMPACT_COEFFICIENT = 0.4
SPREAD_HALF = 0.001
MAX_SLIPPAGE_PCT = 2.0
INITIAL_CASH = 1000000

PWVC_VETO_THRESHOLD = 0.8
J_OVERSOLD_THRESHOLD = 13

LIMIT_UP_PCT = 9.5
LIMIT_DOWN_PCT = -9.5
ST_LIMIT_UP_PCT = 4.5
ST_LIMIT_DOWN_PCT = -4.5


def is_st_stock(name: str) -> bool:
    for kw in ['ST', '*ST', 'S*ST', 'SST']:
        if name.startswith(kw):
            return True
    return False


def get_index_daily(ts_code='000001.SH', start_date='20210101', end_date=END_DATE):
    try:
        time.sleep(API_SLEEP)
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


def get_industry_daily(ts_code, start_date='20210101', end_date=END_DATE):
    try:
        time.sleep(API_SLEEP)
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


def get_stock_daily(ts_code, start_date='20210101', end_date=END_DATE):
    for attempt in range(3):
        try:
            time.sleep(API_SLEEP)
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
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
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
    df['MACD_cross_up'] = False
    df['MACD_cross_down'] = False
    for i in range(1, len(df)):
        if pd.notna(df['MACD'].iloc[i]) and pd.notna(df['MACD'].iloc[i-1]):
            if df['MACD'].iloc[i] >= 0 and df['MACD'].iloc[i-1] < 0:
                df.iloc[i, df.columns.get_loc('MACD_cross_up')] = True
            if df['MACD'].iloc[i] < 0 and df['MACD'].iloc[i-1] >= 0:
                df.iloc[i, df.columns.get_loc('MACD_cross_down')] = True
    df['white_above_yellow'] = df['white_line'] > df['yellow_line']
    low_list = df['Low'].rolling(window=9, min_periods=1).min()
    high_list = df['High'].rolling(window=9, min_periods=1).max()
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)
    df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df['D_val'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D_val']
    df['J_below_0_recent5'] = df['J'].rolling(window=5, min_periods=1).min() < 0
    df['J_rising'] = False
    for i in range(1, len(df)):
        if pd.notna(df['J'].iloc[i]) and pd.notna(df['J'].iloc[i-1]):
            df.iloc[i, df.columns.get_loc('J_rising')] = df['J'].iloc[i] > df['J'].iloc[i-1]
    df['yellow_rising'] = False
    for i in range(1, len(df)):
        if pd.notna(df['yellow_line'].iloc[i]) and pd.notna(df['yellow_line'].iloc[i-1]):
            df.iloc[i, df.columns.get_loc('yellow_rising')] = df['yellow_line'].iloc[i] > df['yellow_line'].iloc[i-1]
    dist_pct = (df['Close'] - df['yellow_line']).abs() / df['yellow_line'] * 100
    near_yellow = (dist_pct < 2.0).astype(int)
    df['sideways'] = near_yellow.rolling(window=8, min_periods=1).sum() >= 6
    df['not_sideways'] = ~df['sideways']
    df['vol_ma5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
    df['vol_above_ma5'] = df['Volume'] > df['vol_ma5']
    df['low_above_yellow'] = df['Low'] > df['yellow_line']
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['amplitude_20'] = ((df['High'] - df['Low']) / df['Close'] * 100).rolling(window=20).mean()
    df['pre_close'] = df['Close'].shift(1)
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


def is_limit_up(row, name=''):
    pre_close = row.get('pre_close', np.nan)
    close = row.get('Close', np.nan)
    if pd.isna(pre_close) or pd.isna(close) or pre_close <= 0:
        return False
    pct = (close - pre_close) / pre_close * 100
    threshold = ST_LIMIT_UP_PCT if is_st_stock(name) else LIMIT_UP_PCT
    return pct >= threshold


def is_limit_down(row, name=''):
    pre_close = row.get('pre_close', np.nan)
    close = row.get('Close', np.nan)
    if pd.isna(pre_close) or pd.isna(close) or pre_close <= 0:
        return False
    pct = (close - pre_close) / pre_close * 100
    threshold = ST_LIMIT_DOWN_PCT if is_st_stock(name) else LIMIT_DOWN_PCT
    return pct <= threshold


def check_sell_conditions_v91(pos, row, current_date, oamv_daily, market_state):
    current_price = float(row['Close'])
    entry_price = pos.entry_price
    peak_price = pos.peak_price
    hold_days = (current_date - pos.entry_date).days

    if current_price > peak_price:
        pos.peak_price = current_price
        peak_price = current_price

    profit_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
    dd_pct = (peak_price - current_price) / peak_price * 100 if peak_price > 0 else 0

    if hold_days < 1:
        return None, profit_pct, dd_pct

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

        if sell_reason is None and pos.entry_atr > 0 and peak_price > entry_price:
            dd_atr = (peak_price - current_price) / pos.entry_atr
            if dd_atr >= ATR_STOP_MULT:
                sell_reason = f'ATR止损{dd_atr:.1f}x≥{ATR_STOP_MULT}x'

    return sell_reason, profit_pct, dd_pct


def run_backtest_v91():
    print("=" * 100)
    print("v9.1 策略回测 - 风控止损优先 + 3级市场状态 + 行业约束 + A股真实约束")
    print(f"最近 {SIM_DAYS} 个交易日逐日明细")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    print("\n[1/8] 加载行业数据...")
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
    print(f"  行业数据: {len(industry_j_cache)} 个")

    print("\n[2/8] 初始化v9.1组件...")
    ssa_denoiser = SSADenoiser(window_length=10, n_signal_groups=2)
    sterile_cleaner = SterileDataCleaner()
    disagreement_builder = DisagreementFeatureBuilder(ssa_window=10, ssa_signal_groups=2)
    sig_builder = PathSignatureBuilder(truncation_level=2, path_dims=3, path_length=5, lead_lag=False)
    portfolio_optimizer = CostAwarePortfolioOptimizer(
        n_scenarios=500, block_size=5, lookback_days=200,
        risk_aversion=0.5, cost_aversion=0.5, max_weight=0.25,
        min_weight=0.0, total_max_weight=0.75,
        impact_coefficient=0.4, spread_half=0.001,
        total_capital=INITIAL_CASH
    )
    drift_detector = ADDMDriftDetector(ar_order=3, ph_threshold=2.0, ph_delta=0.01, use_vol_filter=True, decay_lambda=0.005, retrain_cooldown_days=10)
    backtester = PortfolioBacktester(
        initial_cash=INITIAL_CASH,
        max_stocks=MAX_PORTFOLIO_STOCKS,
        commission_rate=COMMISSION_RATE,
        stamp_duty_rate=STAMP_DUTY_RATE,
        slippage_rate=SLIPPAGE_RATE,
        position_size_pct=POSITION_SIZE_PCT,
        catboost_threshold=CATBOOST_BUY_THRESHOLD,
        impact_model='sqrt',
        impact_coefficient=IMPACT_COEFFICIENT,
        spread_half=SPREAD_HALF,
    )
    print(f"  PortfolioBacktester: initial_cash={INITIAL_CASH}, max_stocks={MAX_PORTFOLIO_STOCKS}")
    print(f"  CostAware MVO: 500 scenarios, cost_aversion=0.5, industry_max_weight=40%")

    print("\n[3/8] 加载股票数据...")
    stock_basic = pro.stock_basic(exchange='', list_status='L',
                                   fields='ts_code,symbol,name,industry,list_date')
    a_stocks = stock_basic[
        (stock_basic['ts_code'].str.endswith('.SH')) |
        (stock_basic['ts_code'].str.endswith('.SZ'))
    ]
    a_stocks = a_stocks[~a_stocks['name'].apply(is_st_stock)]
    a_stocks = a_stocks.head(STOCK_LIMIT)
    print(f"  股票数: {len(a_stocks)}")

    index_df = get_index_daily('000001.SH')
    if index_df is None:
        return
    index_df = compute_indicators(index_df)

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
            print(f"  进度: {idx}/{total} ({idx/total*100:.1f}%)")
        try:
            df = get_stock_daily(ts_code)
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

    print(f"  加载完成: {len(all_stock_data)} 只股票")

    print("\n[4/8] Universe AMV + 周线过滤器...")
    oamv_filter = OAMVHysteresisFilter(
        upper_threshold=OAMV_UPPER, lower_threshold=OAMV_LOWER,
        cost_ma_period=34, weekly_ema_period=5, weekly_use_ema=True,
    )
    oamv_filter.fit(index_df, all_stock_data=all_stock_data)
    print(oamv_filter.summary())

    print("\n[5/8] 市场宽度熔断器 (v9.1 3级状态)...")
    panic_breaker = MarketPanicCircuitBreaker(
        breadth_threshold=0.85, limit_down_threshold=150, ma_period=20,
        limit_down_accel_factor=3.0, breadth_deterioration_pct=0.20,
        breadth_deterioration_window=5,
    )
    panic_breaker.compute_market_breadth(all_stock_data)
    print(panic_breaker.summary())

    print("\n[6/8] 训练 RADE集成模型 (CatBoost + ChebyKAN)...")
    from dateutil.relativedelta import relativedelta
    train_end = END_DATE
    train_start_dt = pd.Timestamp(train_end) - relativedelta(months=TRAIN_WINDOW_MONTHS)
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

    print(f"  训练样本: {len(all_samples)}")
    if len(all_samples) < 50:
        print("  样本不足，退出")
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

    catboost_model = CatBoostPredictor(buy_threshold=CATBOOST_BUY_THRESHOLD, l2_leaf_reg=8, max_depth=4, use_amse_loss=False, amse_omega=10.0)
    catboost_model.train(X_train, y_train, X_val, y_val)
    val_probs_cb = catboost_model.predict_proba(X_val)
    val_preds = val_probs_cb >= CATBOOST_BUY_THRESHOLD
    if val_preds.sum() > 0:
        precision = y_val[val_preds].mean()
        print(f"  CatBoost Precision@0.65: {precision:.2%}")

    chebykan_trainer = None
    try:
        chebykan_trainer = ChebyKANTrainer(
            input_dim=len(available_cols),
            hidden_dim=16,
            poly_degree=4,
            lr=0.005,
            epochs=200,
            batch_size=256,
            use_amse_loss=True,
            amse_omega=10.0,
        )
        chebykan_trainer.train(X_train, y_train, X_val, y_val)
        val_probs_ck = chebykan_trainer.predict_proba(X_val)
        ck_preds = val_probs_ck >= CATBOOST_BUY_THRESHOLD
        if ck_preds.sum() > 0:
            ck_precision = y_val[ck_preds].mean()
            print(f"  ChebyKAN Precision@0.65: {ck_precision:.2%}")
    except Exception as e:
        print(f"  ChebyKAN训练失败: {e}, 仅用CatBoost")
        chebykan_trainer = None

    ensemble = RADEEnsemble(gamma=0.5)
    ensemble.set_models(catboost_model, chebykan_trainer)
    print(f"  {ensemble.summary()}")

    print(f"\n[7/8] 模拟交易回放 (v9.1策略, 最近 {SIM_DAYS} 个交易日)...")
    print("=" * 100)

    all_dates = set()
    for ts_code, info in all_stock_data.items():
        df = info['data']
        mask = df.index <= pd.Timestamp(END_DATE)
        all_dates.update(df.index[mask])
    all_dates = sorted(all_dates)
    sim_dates = all_dates[-SIM_DAYS:]

    if len(sim_dates) == 0:
        print("无交易日数据")
        return

    print(f"  回放区间: {sim_dates[0].strftime('%Y-%m-%d')} ~ {sim_dates[-1].strftime('%Y-%m-%d')} ({len(sim_dates)} 个交易日)")

    recent_sell_dates: Dict[str, pd.Timestamp] = {}
    daily_log: List[Dict] = []
    oamv_state_df = oamv_filter.get_state_df()

    index_start = None
    index_end = None
    if index_df is not None and len(index_df) > 0:
        mask = (index_df.index >= sim_dates[0]) & (index_df.index <= sim_dates[-1])
        index_subset = index_df[mask]
        if len(index_subset) >= 2:
            index_start = float(index_subset['Close'].iloc[0])
            index_end = float(index_subset['Close'].iloc[-1])

    for day_idx, current_date in enumerate(sim_dates):
        date_str = current_date.strftime('%Y-%m-%d')
        oamv_daily = oamv_filter.is_trading_allowed(current_date, require_weekly=False)
        oamv_weekly = oamv_filter.is_trading_allowed(current_date, require_weekly=True)
        market_state = panic_breaker.get_market_state(current_date)

        oamv_x = 0.0
        if current_date in oamv_state_df.index:
            oamv_x = float(oamv_state_df.loc[current_date, 'oamv_x'])

        day_sells = []
        day_buys = []

        for ts_code in list(backtester.positions.keys()):
            pos = backtester.positions[ts_code]
            info = all_stock_data[ts_code]
            df = info['data']
            if current_date not in df.index:
                continue
            row = df.loc[current_date]

            if is_limit_down(row, info['name']):
                continue

            sell_reason, profit_pct, dd_pct = check_sell_conditions_v91(
                pos, row, current_date, oamv_daily, market_state
            )

            if sell_reason:
                current_price = float(row['Close'])
                current_vol = float(row.get('Volume', 0))
                daily_vol_value = current_price * current_vol * 100
                daily_volatility = float(row.get('ATR14', 0)) / current_price if current_price > 0 else 0.02

                trade = backtester.sell(
                    ts_code, current_price, current_date,
                    reason=sell_reason,
                    daily_volume=daily_vol_value,
                    daily_volatility=daily_volatility,
                )

                if trade:
                    day_sells.append({
                        'code': ts_code,
                        'name': pos.name,
                        'price': current_price,
                        'reason': sell_reason,
                        'profit_pct': trade.profit_pct,
                        'hold_days': trade.hold_days,
                        'entry_price': pos.entry_price,
                    })
                    recent_sell_dates[ts_code] = current_date

        can_buy = (oamv_weekly
                   and market_state != 'panic'
                   and market_state != 'warning'
                   and backtester.can_buy())

        if can_buy:
            buy_candidates = []
            for ts_code, info in all_stock_data.items():
                if ts_code in backtester.positions:
                    continue
                if ts_code in recent_sell_dates:
                    if (current_date - recent_sell_dates[ts_code]).days < COOLDOWN_DAYS:
                        continue
                df = info['data']
                featured_df = all_featured_data[ts_code]
                if current_date not in df.index or current_date not in featured_df.index:
                    continue

                row = df.loc[current_date]
                if is_limit_up(row, info['name']):
                    continue

                feat_row = featured_df.loc[current_date]
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
                    white_above = df.loc[current_date, 'white_above_yellow'] if 'white_above_yellow' in df.columns else True
                    if pd.notna(white_above) and not white_above:
                        continue

                    j_val = df.loc[current_date, 'J'] if 'J' in df.columns else 100
                    if pd.notna(j_val) and j_val >= J_OVERSOLD_THRESHOLD:
                        continue

                    pwvc_val = feat_row.get('pwvc', 0.0)
                    if pd.notna(pwvc_val) and pwvc_val > PWVC_VETO_THRESHOLD:
                        continue

                    amp = df.loc[current_date, 'amplitude_20'] if 'amplitude_20' in df.columns else 0
                    if pd.notna(amp) and amp > 0 and amp < FRICTION_COST_PCT * MIN_AMPLITUDE_MULT:
                        continue

                    current_price = float(df.loc[current_date, 'Close'])
                    current_vol = float(df.loc[current_date, 'Volume']) if 'Volume' in df.columns and pd.notna(df.loc[current_date, 'Volume']) else 0
                    daily_vol_value = current_price * current_vol * 100
                    order_value = INITIAL_CASH * POSITION_SIZE_PCT
                    participation = order_value / daily_vol_value if daily_vol_value > 0 else 1.0
                    impact_slippage = (IMPACT_COEFFICIENT * np.sqrt(participation) + SPREAD_HALF) * 100
                    impact_slippage = min(impact_slippage, MAX_SLIPPAGE_PCT)

                    if impact_slippage > 1.0:
                        continue

                    buy_candidates.append({
                        'ts_code': ts_code,
                        'name': info['name'],
                        'prob': prob,
                        'price': current_price,
                        'atr': float(df.loc[current_date, 'ATR14']) if 'ATR14' in df.columns and not pd.isna(df.loc[current_date, 'ATR14']) else 0,
                        'industry': info.get('industry', ''),
                        'impact_slippage': impact_slippage,
                        'j_val': float(j_val) if not pd.isna(j_val) else 0,
                        'pwvc': float(pwvc_val) if not pd.isna(pwvc_val) else 0,
                        'accumulation_score': float(feat_row.get('accumulation_score', 0)),
                    })

            buy_candidates.sort(key=lambda x: -x['prob'])

            top_n = min(5, len(buy_candidates))
            if top_n >= 2:
                candidate_codes = [c['ts_code'] for c in buy_candidates[:top_n]]
                ml_probs = {c['ts_code']: c['prob'] for c in buy_candidates[:top_n]}

                try:
                    industry_map = {code: all_stock_data.get(code, {}).get('industry', '') for code in candidate_codes}
                    weights, valid_codes = portfolio_optimizer.optimize(
                        candidate_codes, all_stock_data, oamv_state_df,
                        current_date, ml_probs,
                        industry_map=industry_map
                    )
                    if len(valid_codes) > 0 and len(weights) > 0:
                        for code, w in zip(valid_codes, weights):
                            for c in buy_candidates:
                                if c['ts_code'] == code:
                                    c['mvo_weight'] = w
                                    break
                        buy_candidates.sort(key=lambda x: -x.get('mvo_weight', 0))
                except Exception:
                    pass

            for candidate in buy_candidates:
                if not backtester.can_buy():
                    break
                ts_code = candidate['ts_code']
                current_price = candidate['price']
                current_vol = float(all_stock_data[ts_code]['data'].loc[current_date, 'Volume']) if current_date in all_stock_data[ts_code]['data'].index and pd.notna(all_stock_data[ts_code]['data'].loc[current_date, 'Volume']) else 0
                daily_vol_value = current_price * current_vol * 100
                daily_volatility = candidate['atr'] / current_price if current_price > 0 and candidate['atr'] > 0 else 0.02

                pos = backtester.buy(
                    ts_code=ts_code,
                    name=candidate['name'],
                    price=current_price,
                    date=current_date,
                    prob=candidate['prob'],
                    atr=candidate['atr'],
                    daily_volume=daily_vol_value,
                    daily_volatility=daily_volatility,
                )

                if pos:
                    day_buys.append({
                        'code': ts_code,
                        'name': candidate['name'],
                        'price': current_price,
                        'prob': candidate['prob'],
                        'mvo_weight': candidate.get('mvo_weight'),
                    })

        stock_prices = {}
        for ts_code, pos in backtester.positions.items():
            info = all_stock_data.get(ts_code)
            if info and current_date in info['data'].index:
                stock_prices[ts_code] = float(info['data'].loc[current_date, 'Close'])
        backtester.update_equity(current_date, stock_prices)

        state_display = {'normal': '✅正常', 'warning': '⚠️预警', 'panic': '🔴熔断'}.get(market_state, market_state)

        daily_log.append({
            'date': date_str,
            'oamv_daily': 'BULL' if oamv_daily else 'BEAR',
            'oamv_weekly': 'BULL' if oamv_weekly else 'BEAR',
            'oamv_x': round(oamv_x, 2),
            'market_state': market_state,
            'sells': day_sells,
            'buys': day_buys,
            'positions': {code: {
                'name': p.name,
                'entry_date': p.entry_date.strftime('%Y-%m-%d'),
                'entry_price': p.entry_price,
                'current_price': stock_prices.get(code, p.entry_price),
                'profit_pct': round((stock_prices.get(code, p.entry_price) - p.entry_price) / p.entry_price * 100, 2) if p.entry_price > 0 else 0,
                'hold_days': (current_date - p.entry_date).days,
                'peak_price': p.peak_price,
            } for code, p in backtester.positions.items()},
            'equity': backtester.equity_curve[-1] if backtester.equity_curve else {},
        })

        if day_sells or day_buys:
            sell_str = " ".join([f"卖{s['code']}({s['profit_pct']:+.1f}%)" for s in day_sells])
            buy_str = " ".join([f"买{b['code']}({b['prob']:.0%})" for b in day_buys])
            equity_val = backtester.equity_curve[-1]['total_equity'] if backtester.equity_curve else INITIAL_CASH
            print(f"  {date_str} | {state_display} | {sell_str} {buy_str} | 净值={equity_val:,.0f}")

    print(f"\n{'=' * 100}")
    print("v9.1 回测结果")
    print("=" * 100)

    stock_prices_final = {}
    for ts_code, pos in backtester.positions.items():
        info = all_stock_data.get(ts_code)
        if info and len(info['data']) > 0:
            last_date = info['data'].index[-1]
            stock_prices_final[ts_code] = float(info['data'].loc[last_date, 'Close'])

    backtester.finalize(sim_dates[-1], stock_prices_final)

    summary = backtester.get_summary()
    trade_details = backtester.get_trade_details()

    print(f"\n  回测区间: {sim_dates[0].strftime('%Y-%m-%d')} ~ {sim_dates[-1].strftime('%Y-%m-%d')} ({len(sim_dates)} 个交易日)")
    print(f"  初始资金: {INITIAL_CASH:,.0f}")
    print(f"  最终净值: {summary.get('final_equity', 0):,.0f}")
    print(f"  总收益率: {summary.get('total_return', 0):+.2f}%")
    print(f"  年化收益率: {summary.get('annual_return', 0):+.2f}%")
    print(f"  年化波动率: {summary.get('volatility', 0):.2f}%")
    print(f"  Sharpe比率: {summary.get('sharpe', 0):.2f}")
    print(f"  最大回撤: {summary.get('max_drawdown', 0):.2f}%")
    print(f"  交易笔数: {summary.get('trade_count', 0)}")
    print(f"  胜率: {summary.get('win_rate', 0):.1f}%")
    print(f"  平均盈利: {summary.get('avg_win', 0):+.2f}%")
    print(f"  平均亏损: {summary.get('avg_loss', 0):+.2f}%")
    print(f"  盈亏比: {summary.get('profit_factor', 0):.2f}")
    print(f"  平均持仓天数: {summary.get('avg_hold_days', 0):.1f}")

    if index_start and index_end and index_start > 0:
        index_return = (index_end - index_start) / index_start * 100
        print(f"\n  沪深300同期收益: {index_return:+.2f}%")
        print(f"  超额收益: {summary.get('total_return', 0) - index_return:+.2f}%")

    if trade_details:
        print(f"\n  逐笔明细:")
        print(f"  {'代码':>10s}  {'名称':<8s}  {'买入日':>12s}  {'卖出日':>12s}  "
              f"{'买入价':>8s}  {'卖出价':>8s}  {'收益':>8s}  {'持仓':>4s}  {'卖出原因'}")
        print("  " + "-" * 95)
        for t in trade_details:
            print(f"  {t['code']:>10s}  {t['name']:<8s}  {t['entry_date']:>12s}  {t['exit_date']:>12s}  "
                  f"{t['entry_price']:>8.2f}  {t['exit_price']:>8.2f}  {t['profit_pct']:>+7.2f}%  "
                  f"{t['hold_days']:>4d}天  {t['exit_reason']}")

    equity_df = pd.DataFrame(backtester.equity_curve)
    monthly_returns = {}
    if not equity_df.empty and 'date' in equity_df.columns:
        equity_df['month'] = pd.to_datetime(equity_df['date']).dt.to_period('M')
        for month, group in equity_df.groupby('month'):
            if len(group) >= 2:
                start_val = group['total_equity'].iloc[0]
                end_val = group['total_equity'].iloc[-1]
                monthly_returns[str(month)] = round((end_val - start_val) / start_val * 100, 2)

    if monthly_returns:
        print(f"\n  月度收益表:")
        for month, ret in monthly_returns.items():
            print(f"    {month}: {ret:+.2f}%")

    output_dir = Path(__file__).parent / "backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"backtest_v91_{timestamp}.json"

    result_data = {
        'version': 'v9.1',
        'strategy_features': [
            '风控止损优先（熔断/0AMV不受3天限制）',
            '3级市场状态（normal/warning/panic）',
            'MVO行业集中度约束（同行业≤40%）',
            'A股真实约束（T+1/涨跌停/冷却期）',
            'PortfolioBacktester完整成本计算',
        ],
        'sim_period': f"{sim_dates[0].strftime('%Y-%m-%d')}~{sim_dates[-1].strftime('%Y-%m-%d')}",
        'sim_days': len(sim_dates),
        'initial_cash': INITIAL_CASH,
        'summary': summary,
        'index_return': float((index_end - index_start) / index_start * 100) if index_start and index_end and index_start > 0 else None,
        'excess_return': float(summary.get('total_return', 0) - ((index_end - index_start) / index_start * 100)) if index_start and index_end and index_start > 0 else None,
        'monthly_returns': monthly_returns,
        'trade_details': trade_details,
        'daily_log': daily_log,
        'equity_curve': [{**e, 'date': str(e['date'])} for e in backtester.equity_curve],
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {result_file}")
    print("=" * 100)


if __name__ == "__main__":
    run_backtest_v91()
