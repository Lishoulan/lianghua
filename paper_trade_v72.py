import os
import sys
import gc
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent / "pip_libs"))
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
from ml_strategy.portfolio_optimizer import BootstrappedMVO
from ml_strategy.sterile_cleaner import SterileDataCleaner
from ml_strategy.disagreement_features import DisagreementFeatureBuilder

load_dotenv(Path(__file__).parent / ".env")

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
STOCK_LIMIT = 500
END_DATE = '20260520'

IMPACT_COEFFICIENT = 0.4
SPREAD_HALF = 0.001
MAX_SLIPPAGE_PCT = 2.0
INITIAL_CASH = 500000


def is_st_stock(name: str) -> bool:
    for kw in ['ST', '*ST', 'S*ST', 'SST']:
        if name.startswith(kw):
            return True
    return False


def get_index_daily(ts_code='000001.SH', start_date='20210101', end_date=END_DATE):
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


def get_industry_daily(ts_code, start_date='20210101', end_date=END_DATE):
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


def get_stock_daily(ts_code, start_date='20210101', end_date=END_DATE):
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


class SimPosition:
    def __init__(self, ts_code, name, entry_date, entry_price, entry_prob, entry_atr, impact_slippage=0.0):
        self.ts_code = ts_code
        self.name = name
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.entry_prob = entry_prob
        self.entry_atr = entry_atr
        self.impact_slippage = impact_slippage
        self.peak_price = entry_price
        self.shares = 0

    def profit_pct(self, current_price):
        return (current_price - self.entry_price) / self.entry_price * 100

    def drawdown_from_peak(self, current_price):
        if self.peak_price <= 0:
            return 0
        return (self.peak_price - current_price) / self.peak_price * 100


def compute_sqrt_impact_slippage(order_value, daily_volume_value):
    if daily_volume_value <= 0:
        return MAX_SLIPPAGE_PCT
    participation = order_value / daily_volume_value
    slippage = (IMPACT_COEFFICIENT * np.sqrt(participation) + SPREAD_HALF) * 100
    return min(slippage, MAX_SLIPPAGE_PCT)


def run_paper_trade_sim():
    print("=" * 100)
    print(f"v7.2 模拟交易回放 - ChebyKAN + GARCH-ADDM + Sqrt Impact")
    print(f"最近 {SIM_DAYS} 个交易日逐日明细")
    print(f"SSA降噪 + ADDM漂移检测 + RADE集成(CatBoost+ChebyKAN) + Bootstrapped MVO + Sqrt Impact")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    print("\n[1/7] 加载行业数据...")
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

    print("\n[2/7] 初始化v7.2组件...")
    ssa_denoiser = SSADenoiser(window_length=10, n_signal_groups=2)
    sterile_cleaner = SterileDataCleaner()
    disagreement_builder = DisagreementFeatureBuilder(ssa_window=10, ssa_signal_groups=2)
    portfolio_optimizer = BootstrappedMVO(
        n_scenarios=500, block_size=5, lookback_days=200,
        risk_aversion=0.5, max_weight=0.25, min_weight=0.0, total_max_weight=0.75
    )
    drift_detector = ADDMDriftDetector(ar_order=3, ph_threshold=2.0, ph_delta=0.01, use_vol_filter=True)
    print(f"  SSA: window=10, signal_groups=2")
    print(f"  Sterile Cleaner: ON")
    print(f"  Disagreement Features: ssa_window=10, ssa_signal_groups=2")
    print(f"  MVO: 500 scenarios, block_size=5")
    print(f"  ADDM: ar_order=3, ph_threshold=2.0, ph_delta=0.01, vol_filter=True")
    print(f"  Sqrt Impact: coeff={IMPACT_COEFFICIENT}, spread_half={SPREAD_HALF}, max_slip={MAX_SLIPPAGE_PCT}%")

    print("\n[3/7] 加载股票数据 (with Sterile Cleaner + Disagreement Features + SSA denoising)...")
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
    print(f"  ADDM GARCH vol filter: ON, market_vol={market_vol:.4f}" if 'market_vol' in dir() else "  ADDM GARCH vol filter: ON")

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
            all_featured_data[ts_code] = featured_df
        except Exception:
            continue
        if idx % 100 == 0:
            gc.collect()

    print(f"  加载完成: {len(all_stock_data)} 只股票")

    print("\n[4/7] Universe AMV + 周线过滤器...")
    oamv_filter = OAMVHysteresisFilter(
        upper_threshold=OAMV_UPPER, lower_threshold=OAMV_LOWER,
        cost_ma_period=34, weekly_ema_period=5, weekly_use_ema=True,
    )
    oamv_filter.fit(index_df, all_stock_data=all_stock_data)
    print(oamv_filter.summary())

    print("\n[5/7] 市场宽度熔断器...")
    panic_breaker = MarketPanicCircuitBreaker(breadth_threshold=0.85, limit_down_threshold=150, ma_period=20)
    panic_breaker.compute_market_breadth(all_stock_data)
    print(panic_breaker.summary())

    print("\n[6/7] 训练 RADE集成模型 (CatBoost + ChebyKAN)...")
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

    catboost_model = CatBoostPredictor(buy_threshold=CATBOOST_BUY_THRESHOLD, l2_leaf_reg=8, max_depth=4)
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

    print(f"\n[7/7] 模拟交易回放 (最近 {SIM_DAYS} 个交易日)...")
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

    positions: Dict[str, SimPosition] = {}
    recent_sell_dates: Dict[str, pd.Timestamp] = {}
    all_trades: List[Dict] = []
    daily_log: List[Dict] = []
    oamv_state_df = oamv_filter.get_state_df()

    for day_idx, current_date in enumerate(sim_dates):
        date_str = current_date.strftime('%Y-%m-%d')
        oamv_daily = oamv_filter.is_trading_allowed(current_date, require_weekly=False)
        oamv_weekly = oamv_filter.is_trading_allowed(current_date, require_weekly=True)
        is_panic = panic_breaker.is_panic(current_date)

        oamv_x = 0.0
        if current_date in oamv_state_df.index:
            oamv_x = float(oamv_state_df.loc[current_date, 'oamv_x'])

        day_actions = []
        day_sells = []
        day_buys = []

        for ts_code in list(positions.keys()):
            pos = positions[ts_code]
            info = all_stock_data[ts_code]
            df = info['data']
            if current_date not in df.index:
                continue
            row = df.loc[current_date]
            current_price = float(row['Close'])
            if current_price > pos.peak_price:
                pos.peak_price = current_price

            hold_days = (current_date - pos.entry_date).days
            if hold_days < MIN_HOLD_DAYS:
                continue

            sell_reason = None

            if is_panic:
                sell_reason = '熔断器触发'
            elif not oamv_daily:
                sell_reason = '0AMV日线BEAR'
            else:
                yellow_line = row.get('yellow_line')
                if yellow_line is not None and not pd.isna(yellow_line):
                    if current_price < yellow_line:
                        sell_reason = f'收盘<{yellow_line:.2f}(黄线)'

                if sell_reason is None:
                    dd = pos.drawdown_from_peak(current_price)
                    if dd >= TRAILING_STOP_PCT:
                        sell_reason = f'峰值回撤{dd:.1f}%≥{TRAILING_STOP_PCT}%'

                if sell_reason is None and pos.entry_atr > 0:
                    dd_atr = (pos.peak_price - current_price) / pos.entry_atr
                    if dd_atr >= ATR_STOP_MULT:
                        sell_reason = f'ATR止损{dd_atr:.1f}x≥{ATR_STOP_MULT}x'

            if sell_reason:
                profit = pos.profit_pct(current_price)
                day_sells.append({
                    'code': ts_code,
                    'name': pos.name,
                    'price': current_price,
                    'reason': sell_reason,
                    'profit_pct': profit,
                    'hold_days': hold_days,
                    'entry_price': pos.entry_price,
                    'entry_date': pos.entry_date.strftime('%Y-%m-%d'),
                    'impact_slippage': pos.impact_slippage,
                })
                all_trades.append({
                    'code': ts_code, 'name': pos.name,
                    'entry_date': pos.entry_date.strftime('%Y-%m-%d'),
                    'exit_date': date_str,
                    'entry_price': pos.entry_price,
                    'exit_price': current_price,
                    'profit_pct': profit,
                    'hold_days': hold_days,
                    'exit_reason': sell_reason,
                    'impact_slippage': pos.impact_slippage,
                })
                del positions[ts_code]
                recent_sell_dates[ts_code] = current_date

        if oamv_weekly and not is_panic and len(positions) < MAX_PORTFOLIO_STOCKS:
            buy_candidates = []
            for ts_code, info in all_stock_data.items():
                if ts_code in positions:
                    continue
                if ts_code in recent_sell_dates:
                    if (current_date - recent_sell_dates[ts_code]).days < COOLDOWN_DAYS:
                        continue
                df = info['data']
                featured_df = all_featured_data[ts_code]
                if current_date not in df.index or current_date not in featured_df.index:
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
                    amp = df.loc[current_date, 'amplitude_20'] if 'amplitude_20' in df.columns else 0
                    if pd.notna(amp) and amp > 0 and amp < FRICTION_COST_PCT * MIN_AMPLITUDE_MULT:
                        continue

                    current_price = float(df.loc[current_date, 'Close'])
                    current_vol = float(df.loc[current_date, 'Volume']) if 'Volume' in df.columns and pd.notna(df.loc[current_date, 'Volume']) else 0
                    daily_vol_value = current_price * current_vol * 100
                    order_value = INITIAL_CASH * POSITION_SIZE_PCT
                    impact_slippage = compute_sqrt_impact_slippage(order_value, daily_vol_value)

                    if impact_slippage > 1.0:
                        continue

                    buy_candidates.append({
                        'ts_code': ts_code,
                        'name': info['name'],
                        'prob': prob,
                        'price': current_price,
                        'atr': float(df.loc[current_date, 'ATR14']) if 'ATR14' in df.columns and not pd.isna(df.loc[current_date, 'ATR14']) else 0,
                        'amplitude_20': float(amp) if not pd.isna(amp) else 0,
                        'impact_slippage': impact_slippage,
                    })
            buy_candidates.sort(key=lambda x: -x['prob'])

            top_n = min(5, len(buy_candidates))
            if top_n >= 2:
                candidate_codes = [c['ts_code'] for c in buy_candidates[:top_n]]
                ml_probs = {c['ts_code']: c['prob'] for c in buy_candidates[:top_n]}
                try:
                    weights, valid_codes = portfolio_optimizer.optimize(
                        candidate_codes, all_stock_data, oamv_state_df,
                        current_date, ml_probs
                    )
                    if len(valid_codes) > 0 and len(weights) > 0:
                        weighted_candidates = []
                        for code, w in zip(valid_codes, weights):
                            for c in buy_candidates:
                                if c['ts_code'] == code:
                                    c['mvo_weight'] = w
                                    weighted_candidates.append(c)
                                    break
                        weighted_candidates.sort(key=lambda x: -x.get('mvo_weight', 0))
                        buy_candidates = weighted_candidates
                except Exception:
                    pass

            for candidate in buy_candidates:
                if len(positions) >= MAX_PORTFOLIO_STOCKS:
                    break
                ts_code = candidate['ts_code']
                pos = SimPosition(
                    ts_code=ts_code,
                    name=candidate['name'],
                    entry_date=current_date,
                    entry_price=candidate['price'],
                    entry_prob=candidate['prob'],
                    entry_atr=candidate['atr'],
                    impact_slippage=candidate['impact_slippage'],
                )
                positions[ts_code] = pos
                day_buys.append({
                    'code': ts_code,
                    'name': candidate['name'],
                    'price': candidate['price'],
                    'prob': candidate['prob'],
                    'atr': candidate['atr'],
                    'mvo_weight': candidate.get('mvo_weight', None),
                    'impact_slippage': candidate['impact_slippage'],
                })

        daily_log.append({
            'date': date_str,
            'oamv_daily': 'BULL' if oamv_daily else 'BEAR',
            'oamv_weekly': 'BULL' if oamv_weekly else 'BEAR',
            'oamv_x': round(oamv_x, 2),
            'panic': is_panic,
            'sells': day_sells,
            'buys': day_buys,
            'positions': {code: {
                'name': p.name,
                'entry_date': p.entry_date.strftime('%Y-%m-%d'),
                'entry_price': p.entry_price,
                'current_price': float(all_stock_data[code]['data'].loc[current_date, 'Close']) if current_date in all_stock_data[code]['data'].index else p.entry_price,
                'profit_pct': round(p.profit_pct(float(all_stock_data[code]['data'].loc[current_date, 'Close'])) if current_date in all_stock_data[code]['data'].index else 0, 2),
                'hold_days': (current_date - p.entry_date).days,
                'peak_price': p.peak_price,
                'impact_slippage': p.impact_slippage,
            } for code, p in positions.items()},
        })

    print()
    print("=" * 100)
    print("逐日交易明细")
    print("=" * 100)

    for day in daily_log:
        print(f"\n{'─' * 100}")
        print(f"📅 {day['date']}  |  0AMV日线={day['oamv_daily']} 周线={day['oamv_weekly']} (X={day['oamv_x']:+.2f}%)  |  熔断={'⚠️触发' if day['panic'] else '正常'}")

        if day['sells']:
            print(f"  🔴 卖出:")
            for s in day['sells']:
                impact_str = f"  冲击滑点={s['impact_slippage']:.3f}%" if s.get('impact_slippage', 0) > 0 else ""
                print(f"     {s['code']} {s['name']}  卖出价={s['price']:.2f}  买入价={s['entry_price']:.2f}  "
                      f"收益={s['profit_pct']:+.2f}%  持仓={s['hold_days']}天  原因={s['reason']}{impact_str}")

        if day['buys']:
            print(f"  🟢 买入:")
            for b in day['buys']:
                mvo_str = f"  MVO权重={b['mvo_weight']:.2f}" if b.get('mvo_weight') is not None else ""
                print(f"     {b['code']} {b['name']}  买入价={b['price']:.2f}  RADE概率={b['prob']:.1%}  ATR={b['atr']:.2f}{mvo_str}  冲击滑点={b['impact_slippage']:.3f}%")

        if day['positions']:
            print(f"  📊 持仓 ({len(day['positions'])}/{MAX_PORTFOLIO_STOCKS}):")
            for code, p in day['positions'].items():
                dd = 0
                if p['peak_price'] > 0:
                    dd = (p['peak_price'] - p['current_price']) / p['peak_price'] * 100
                impact_str = f"  滑点={p['impact_slippage']:.3f}%" if p.get('impact_slippage', 0) > 0 else ""
                print(f"     {code} {p['name']}  现价={p['current_price']:.2f}  "
                      f"收益={p['profit_pct']:+.2f}%  持仓={p['hold_days']}天  峰值回撤={dd:.1f}%{impact_str}")

        if not day['sells'] and not day['buys'] and not day['positions']:
            if day['oamv_weekly'] == 'BEAR':
                print(f"  ⏸️  周线BEAR，空仓观望")
            elif day['oamv_daily'] == 'BEAR':
                print(f"  ⏸️  日线BEAR，空仓观望")
            else:
                print(f"  ⏸️  无信号，空仓观望")

    print(f"\n{'=' * 100}")
    print("交易汇总")
    print("=" * 100)

    if all_trades:
        print(f"\n  总交易笔数: {len(all_trades)}")
        wins = [t for t in all_trades if t['profit_pct'] > 0]
        losses = [t for t in all_trades if t['profit_pct'] <= 0]
        print(f"  盈利: {len(wins)} 笔, 亏损: {len(losses)} 笔")
        if wins:
            print(f"  平均盈利: {np.mean([t['profit_pct'] for t in wins]):+.2f}%")
        if losses:
            print(f"  平均亏损: {np.mean([t['profit_pct'] for t in losses]):+.2f}%")
        print(f"  总收益: {sum(t['profit_pct'] for t in all_trades):+.2f}%")

        avg_impact = np.mean([t['impact_slippage'] for t in all_trades])
        print(f"  平均冲击滑点: {avg_impact:.3f}%")

        print(f"\n  逐笔明细:")
        print(f"  {'代码':>10s}  {'名称':<8s}  {'买入日':>12s}  {'卖出日':>12s}  "
              f"{'买入价':>8s}  {'卖出价':>8s}  {'收益':>8s}  {'持仓':>4s}  {'滑点':>6s}  {'卖出原因'}")
        print("  " + "-" * 105)
        for t in all_trades:
            print(f"  {t['code']:>10s}  {t['name']:<8s}  {t['entry_date']:>12s}  {t['exit_date']:>12s}  "
                  f"{t['entry_price']:>8.2f}  {t['exit_price']:>8.2f}  {t['profit_pct']:>+7.2f}%  "
                  f"{t['hold_days']:>4d}天  {t['impact_slippage']:>5.3f}%  {t['exit_reason']}")
    else:
        print("\n  无交易记录")

    output_dir = Path(__file__).parent / "paper_trade_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"paper_trade_v72_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'version': 'v7.2',
            'sim_period': f"{sim_dates[0].strftime('%Y-%m-%d')}~{sim_dates[-1].strftime('%Y-%m-%d')}",
            'daily_log': daily_log,
            'all_trades': all_trades,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {result_file}")
    print("=" * 100)


if __name__ == "__main__":
    run_paper_trade_sim()
