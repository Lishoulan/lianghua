import os
import sys
import gc
import json
import time
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
from ml_strategy.portfolio_backtest import PortfolioBacktester
from ml_strategy.panic_breaker import MarketPanicCircuitBreaker
from ml_strategy.ssa_denoiser import SSADenoiser
from ml_strategy.drift_detector import ADDMDriftDetector
from ml_strategy.kan_predictor import KANTrainer
from ml_strategy.chebykan_predictor import ChebyKANTrainer
from ml_strategy.rade_ensemble import RADEEnsemble
from ml_strategy.portfolio_optimizer import BootstrappedMVO

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

WF_QUARTERS = []
from dateutil.relativedelta import relativedelta
for year in [2024, 2025]:
    for q_month in [1, 4, 7, 10]:
        test_start = f"{year}{q_month:02d}01"
        if q_month == 10:
            test_end = f"{year}1231"
        else:
            end_month = q_month + 2
            last_day = 30 if end_month in [6, 9, 11] else 31
            test_end = f"{year}{end_month:02d}{last_day}"
        train_end = f"{year}{q_month:02d}01"
        train_start_dt = pd.Timestamp(train_end) - relativedelta(months=TRAIN_WINDOW_MONTHS)
        train_start = train_start_dt.strftime('%Y%m%d')
        WF_QUARTERS.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
        })

WF_QUARTERS.append({
    'train_start': (pd.Timestamp('20260101') - relativedelta(months=TRAIN_WINDOW_MONTHS)).strftime('%Y%m%d'),
    'train_end': '20260101',
    'test_start': '20260101',
    'test_end': '20260520',
})


def is_st_stock(name: str) -> bool:
    for kw in ['ST', '*ST', 'S*ST', 'SST']:
        if name.startswith(kw):
            return True
    return False


def get_all_a_stocks(limit=None, filter_st=True):
    print("Fetching all A-share stocks...")
    try:
        stock_basic = pro.stock_basic(exchange='', list_status='L',
                                       fields='ts_code,symbol,name,industry,list_date')
        a_stocks = stock_basic[
            (stock_basic['ts_code'].str.endswith('.SH')) |
            (stock_basic['ts_code'].str.endswith('.SZ'))
        ]
        if filter_st:
            before = len(a_stocks)
            a_stocks = a_stocks[~a_stocks['name'].apply(is_st_stock)]
            print(f"Filtered ST: {before} -> {len(a_stocks)}")
        if limit:
            a_stocks = a_stocks.head(limit)
        print(f"Total: {len(a_stocks)} stocks")
        return a_stocks
    except Exception as e:
        print(f"Failed: {e}")
        return None


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


def load_industry_data(start_date='20210101', end_date='20260520'):
    print("Loading industry data...")
    industry_j_cache = {}
    sw_count = 0
    etf_count = 0
    for ind_name, sw_code in INDUSTRY_MAP.items():
        try:
            ind_df = get_industry_daily(sw_code, start_date, end_date)
            if ind_df is not None and len(ind_df) > 20:
                ind_j = compute_industry_j(ind_df)
                if ind_j is not None:
                    industry_j_cache[ind_name] = ind_j
                    sw_count += 1
                    continue
        except Exception:
            pass
        etf_code = INDUSTRY_ETF_FALLBACK.get(ind_name)
        if etf_code:
            try:
                ind_df = get_industry_daily(etf_code, start_date, end_date)
                if ind_df is not None and len(ind_df) > 20:
                    ind_j = compute_industry_j(ind_df)
                    if ind_j is not None:
                        industry_j_cache[ind_name] = ind_j
                        etf_count += 1
            except Exception:
                pass
    print(f"  sw_daily: {sw_count}, ETF fallback: {etf_count}, Total: {len(industry_j_cache)}")
    return industry_j_cache


def load_all_stock_data(stock_list, index_df, industry_j_cache, ssa_denoiser,
                        start_date='20210101', end_date='20260520'):
    print("Loading all stock data (with SSA denoising)...")
    all_stock_data = {}
    all_featured_data = {}
    discretizer = FeatureDiscretizer()
    total = len(stock_list)
    for idx, (_, row) in enumerate(stock_list.iterrows(), 1):
        ts_code = row['ts_code']
        name = row['name']
        industry = row.get('industry', '')
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{total} ({idx/total*100:.1f}%)")
        try:
            df = get_stock_daily(ts_code, start_date, end_date)
            if df is None or len(df) < 200:
                continue
            if df['Close'].iloc[-1] < MIN_PRICE:
                continue
            df = compute_indicators(df)
            if df is None:
                continue
            all_stock_data[ts_code] = {'data': df, 'name': name, 'industry': industry}
            featured_df = discretizer.transform(df)
            industry_j = industry_j_cache.get(industry)
            featured_df = discretizer.add_market_context(featured_df, index_df, None, industry_j)
            featured_df = ssa_denoiser.denoise_features(featured_df)
            all_featured_data[ts_code] = featured_df
        except Exception:
            continue
        if idx % 100 == 0:
            gc.collect()
    print(f"Loaded {len(all_stock_data)} stocks")
    return all_stock_data, all_featured_data


def train_models(all_stock_data, all_featured_data, oamv_state_dict,
                 train_start, train_end, drift_detector=None,
                 use_temporal_weights=False, train_dates=None):
    print(f"  Training: {train_start} ~ {train_end}")
    all_samples = []
    sample_dates = []
    labeler = TripleBarrierLabeler()
    labeler.max_hold_days = 10
    feature_cols = CatBoostPredictor.FEATURE_COLS
    ts_codes = list(all_stock_data.keys())
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)

    for ts_code in ts_codes:
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
                if li < len(df):
                    sample_dates.append(df.index[li])
        except Exception:
            continue

    if len(all_samples) < 50:
        print(f"  Only {len(all_samples)} samples, skipping")
        return None, None, None

    samples_df = pd.DataFrame(all_samples)
    available_cols = [c for c in feature_cols if c in samples_df.columns]
    X_all = samples_df[available_cols].dropna()
    y_all = samples_df.loc[X_all.index, 'label'].values
    split_idx = int(len(X_all) * 0.8)
    X_train = X_all.iloc[:split_idx]
    y_train = y_all[:split_idx]
    X_val = X_all.iloc[split_idx:]
    y_val = y_all[split_idx:]

    sample_weights = None
    if use_temporal_weights and drift_detector is not None and len(sample_dates) > 0:
        train_sample_dates = sample_dates[:split_idx]
        if len(train_sample_dates) == len(X_train):
            sample_weights = drift_detector.compute_temporal_weights(train_sample_dates)

    catboost_model = CatBoostPredictor(buy_threshold=CATBOOST_BUY_THRESHOLD, l2_leaf_reg=8, max_depth=4)
    catboost_model.train(X_train, y_train, X_val, y_val)
    val_probs_cb = catboost_model.predict_proba(X_val)
    val_preds = val_probs_cb >= CATBOOST_BUY_THRESHOLD
    if val_preds.sum() > 0:
        precision = y_val[val_preds].mean()
        print(f"  CatBoost Samples: {len(all_samples)}, Precision@0.65: {precision:.2%}")

    kan_trainer = None
    try:
        kan_trainer = ChebyKANTrainer(
            input_dim=len(available_cols),
            hidden_dim=16,
            poly_degree=4,
            lr=0.005,
            epochs=200,
            batch_size=256,
        )
        kan_trainer.train(X_train, y_train, X_val, y_val,
                          sample_weights=sample_weights)
        val_probs_kan = kan_trainer.predict_proba(X_val)
        kan_preds = val_probs_kan >= CATBOOST_BUY_THRESHOLD
        if kan_preds.sum() > 0:
            kan_precision = y_val[kan_preds].mean()
            print(f"  ChebyKAN Precision@0.65: {kan_precision:.2%}")
    except Exception as e:
        print(f"  ChebyKAN training failed: {e}, using CatBoost only")
        kan_trainer = None

    ensemble = RADEEnsemble(gamma=0.5)
    ensemble.set_models(catboost_model, kan_trainer)

    return catboost_model, kan_trainer, ensemble


def run_portfolio_backtest(all_stock_data, all_featured_data, ensemble, oamv_filter,
                           panic_breaker, drift_detector, portfolio_optimizer,
                           test_start, test_end, index_df=None):
    print(f"  Backtest: {test_start} ~ {test_end}")
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    backtester = PortfolioBacktester(
        initial_cash=10000000,
        max_stocks=MAX_PORTFOLIO_STOCKS,
        commission_rate=COMMISSION_RATE,
        stamp_duty_rate=STAMP_DUTY_RATE,
        slippage_rate=SLIPPAGE_RATE,
        position_size_pct=POSITION_SIZE_PCT,
        catboost_threshold=CATBOOST_BUY_THRESHOLD,
        impact_model='sqrt',
        impact_coefficient=0.4,
        spread_half=0.001,
    )
    all_dates = set()
    for ts_code, info in all_stock_data.items():
        df = info['data']
        mask = (df.index >= test_start_ts) & (df.index <= test_end_ts)
        all_dates.update(df.index[mask])
    eval_dates = sorted(all_dates)
    if len(eval_dates) == 0:
        return backtester

    feature_cols = CatBoostPredictor.FEATURE_COLS
    recent_sell_dates: Dict[str, pd.Timestamp] = {}
    oamv_state_df = oamv_filter.get_state_df()
    drift_retrain_count = 0
    oos_predictions = []
    oos_labels = []

    for date_idx, current_date in enumerate(eval_dates, 1):
        oamv_state = 1 if oamv_filter.is_trading_allowed(current_date, require_weekly=False) else 0
        weekly_ok = oamv_filter.is_trading_allowed(current_date, require_weekly=True)
        is_panic = panic_breaker.is_panic(current_date)

        oamv_x = 0.0
        if current_date in oamv_state_df.index:
            oamv_x = float(oamv_state_df.loc[current_date, 'oamv_x'])

        stock_prices = {}
        for ts_code, info in all_stock_data.items():
            df = info['data']
            if current_date in df.index:
                stock_prices[ts_code] = df.loc[current_date, 'Close']

        existing_positions = list(backtester.positions.keys())
        for ts_code in existing_positions:
            pos = backtester.positions[ts_code]
            info = all_stock_data[ts_code]
            df = info['data']
            if current_date not in df.index:
                continue
            row = df.loc[current_date]
            current_price = row['Close']
            if current_price > pos.peak_price:
                pos.peak_price = current_price
            hold_days = (current_date - pos.entry_date).days
            if hold_days < MIN_HOLD_DAYS:
                continue
            if is_panic:
                backtester.sell(ts_code, current_price, current_date, reason='panic_circuit_breaker')
                recent_sell_dates[ts_code] = current_date
                continue
            individual_stop = False
            yellow_line = row.get('yellow_line')
            if yellow_line is not None and not pd.isna(yellow_line):
                if current_price < yellow_line:
                    individual_stop = True
            trailing_stop_pct = False
            if pos.entry_price > 0:
                drawdown_pct = (pos.peak_price - current_price) / pos.peak_price * 100
                if drawdown_pct >= TRAILING_STOP_PCT:
                    trailing_stop_pct = True
            trailing_atr_stop = False
            if pos.entry_atr > 0:
                drawdown_atr = (pos.peak_price - current_price) / pos.entry_atr
                if drawdown_atr >= ATR_STOP_MULT:
                    trailing_atr_stop = True
            system_liquidation = (oamv_state == 0)
            if individual_stop or trailing_stop_pct or trailing_atr_stop or system_liquidation:
                reason = []
                if individual_stop:
                    reason.append('close<yellow_line')
                if trailing_stop_pct:
                    reason.append(f'{TRAILING_STOP_PCT}%trailing')
                if trailing_atr_stop:
                    reason.append(f'{ATR_STOP_MULT}xATR')
                if system_liquidation:
                    reason.append('system_liquidation')
                backtester.sell(ts_code, current_price, current_date, reason=', '.join(reason))
                recent_sell_dates[ts_code] = current_date

        if weekly_ok and backtester.can_buy():
            buy_candidates = []
            for ts_code, info in all_stock_data.items():
                if ts_code in backtester.positions:
                    continue
                if ts_code in recent_sell_dates:
                    if (current_date - recent_sell_dates[ts_code]).days < COOLDOWN_DAYS:
                        continue
                df = info['data']
                featured_df = all_featured_data[ts_code]
                name = info['name']
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
                    buy_candidates.append({
                        'ts_code': ts_code,
                        'name': name,
                        'prob': prob,
                        'price': df.loc[current_date, 'Close'],
                        'atr': df.loc[current_date, 'ATR14'] if 'ATR14' in df.columns and pd.notna(df.loc[current_date, 'ATR14']) else 0,
                        'amplitude_20': float(amp) if not pd.isna(amp) else 0,
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
                if not backtester.can_buy():
                    break
                if is_panic:
                    continue
                ts_code = candidate['ts_code']
                price = candidate['price']
                if pd.isna(price) or price <= 0:
                    continue
                backtester.buy(ts_code, candidate['name'], price, current_date,
                              candidate['prob'], candidate.get('atr', 0.0),
                              daily_volume=float(df.loc[current_date, 'Volume']) if 'Volume' in df.columns and current_date in df.index and pd.notna(df.loc[current_date, 'Volume']) else 0,
                              daily_volatility=float(df.loc[current_date, 'amplitude_20']) / 100 if 'amplitude_20' in df.columns and current_date in df.index and pd.notna(df.loc[current_date, 'amplitude_20']) else 0.02)

        backtester.update_equity(current_date, stock_prices)

        if drift_detector is not None and date_idx % 5 == 0 and len(backtester.trades) > 0:
            recent_trades = backtester.trades[-20:]
            if len(recent_trades) >= 5:
                trade_probs = np.array([t.entry_prob for t in recent_trades])
                trade_labels = np.array([1.0 if t.profit_pct > 0 else 0.0 for t in recent_trades])
                if index_df is not None and 'ATR14' in index_df.columns and current_date in index_df.index:
                    market_atr = float(index_df.loc[current_date, 'ATR14'])
                    market_close = float(index_df.loc[current_date, 'Close'])
                    market_vol = market_atr / market_close if market_close > 0 else 0.02
                    drift_detector.set_market_volatility(market_vol)
                drift_detected = drift_detector.update(trade_labels, trade_probs)
                if drift_detected:
                    drift_retrain_count += 1
                    print(f"    >>> ADDM drift detected at {current_date.strftime('%Y-%m-%d')}! "
                          f"(retrain #{drift_retrain_count})")

    backtester.finalize(eval_dates[-1], stock_prices)
    if drift_detector is not None:
        print(f"  ADDM summary: {drift_detector.drift_count} drift detections")
    return backtester


def run_backtest(stock_limit=300):
    print("=" * 120)
    print("v7.1: ChebyKAN + GARCH-filtered ADDM + Sqrt Impact Model")
    print("=" * 120)
    print(f"\nConfig:")
    print(f"  SSA window: 10, signal groups: 2")
    print(f"  ADDM: AR(3), PH threshold=2.0, delta=0.01, GARCH vol filter=ON")
    print(f"  RADE: gamma=0.5, CatBoost weight range [0.2, 0.9]")
    print(f"  MVO: 500 scenarios, block_size=5, risk_aversion=0.5")
    print(f"  Train window: {TRAIN_WINDOW_MONTHS} months rolling")
    print(f"  WF quarters: {len(WF_QUARTERS)}")
    print(f"  CatBoost: l2_leaf_reg=8, max_depth=4")
    print(f"  ChebyKAN: hidden_dim=16, poly_degree=4")
    print(f"  Impact: sqrt model, Y=0.4, spread=0.1%")

    print("\n[Stage 1] 加载行业数据")
    print("-" * 80)
    industry_j_cache = load_industry_data('20210101', '20260520')

    print("\n[Stage 2] 初始化v7.0组件")
    print("-" * 80)
    ssa_denoiser = SSADenoiser(window_length=10, n_signal_groups=2)
    drift_detector = ADDMDriftDetector(ar_order=3, ph_threshold=2.0, ph_delta=0.01, use_vol_filter=True)
    portfolio_optimizer = BootstrappedMVO(
        n_scenarios=500, block_size=5, lookback_days=200,
        risk_aversion=0.5, max_weight=0.25, min_weight=0.0, total_max_weight=0.75
    )
    print(f"  SSA Denoiser: window=10, signal_groups=2")
    print(f"  ADDM Drift Detector: AR(3), PH threshold=2.0")
    print(f"  Bootstrapped MVO: 500 scenarios, block_size=5")
    print(drift_detector.summary())
    print(portfolio_optimizer.summary())

    print("\n[Stage 3] 加载股票数据 (with SSA denoising)")
    print("-" * 80)
    stock_list = get_all_a_stocks(limit=stock_limit, filter_st=True)
    if stock_list is None:
        return

    index_df = get_index_daily('000001.SH', '20210101', '20260520')
    if index_df is None:
        return
    index_df = compute_indicators(index_df)

    all_stock_data, all_featured_data = load_all_stock_data(
        stock_list, index_df, industry_j_cache, ssa_denoiser, '20210101', '20260520'
    )
    if not all_stock_data:
        return

    print("\n[Stage 4] Universe AMV + 自适应周线过滤器")
    print("-" * 80)
    oamv_filter = OAMVHysteresisFilter(
        upper_threshold=OAMV_UPPER,
        lower_threshold=OAMV_LOWER,
        cost_ma_period=34,
        weekly_ema_period=5,
        weekly_use_ema=True,
    )
    oamv_filter.fit(index_df, all_stock_data=all_stock_data)
    print(oamv_filter.summary())

    oamv_state_dict = oamv_filter.get_state_dict()

    transitions = oamv_filter.get_transition_dates()
    print(f"\n0AMV Transitions (total {len(transitions)}):")
    for t in transitions[-10:]:
        direction = "→ BULL" if t['to'] == 1 else "→ BEAR"
        print(f"  {t['date'].strftime('%Y-%m-%d')}: State {t['from']} {direction} (X={t['x_value']:.2f}%)")

    print("\n[Stage 5] 市场宽度熔断器")
    print("-" * 80)
    panic_breaker = MarketPanicCircuitBreaker(breadth_threshold=0.85, limit_down_threshold=150, ma_period=20)
    panic_breaker.compute_market_breadth(all_stock_data)
    print(panic_breaker.summary())

    print("\n[Stage 6] 季度Walk-Forward滚动验证 (v7.1)")
    print("-" * 80)

    all_wf_results = []

    for wf_idx, wf_window in enumerate(WF_QUARTERS, 1):
        test_start_ts = pd.Timestamp(wf_window['test_start'])
        if test_start_ts > pd.Timestamp('20260520'):
            continue

        print(f"\n--- Q{wf_idx}: Train {wf_window['train_start']}~{wf_window['train_end']}, "
              f"Test {wf_window['test_start']}~{wf_window['test_end']} ---")

        drift_detector.reset()

        catboost_model, kan_trainer, ensemble = train_models(
            all_stock_data, all_featured_data, oamv_state_dict,
            wf_window['train_start'], wf_window['train_end'],
            drift_detector=drift_detector,
            use_temporal_weights=False,
        )

        if catboost_model is None:
            print("  Model failed, skipping")
            continue

        if ensemble is not None:
            print(f"  {ensemble.summary()}")

        backtester = run_portfolio_backtest(
        all_stock_data, all_featured_data, ensemble, oamv_filter,
        panic_breaker, drift_detector, portfolio_optimizer,
        wf_window['test_start'], wf_window['test_end'], index_df
    )

        summary = backtester.get_summary()
        s = summary
        print(f"  Return: {s.get('total_return', 0):+.2f}%, Sharpe: {s.get('sharpe', 0):.3f}, "
              f"MaxDD: {s.get('max_drawdown', 0):.2f}%, WinRate: {s.get('win_rate', 0):.1f}%, "
              f"Trades: {s.get('trade_count', 0)}")

        all_wf_results.append({
            'window': wf_window,
            'summary': summary,
            'drift_detections': drift_detector.drift_count,
            'ensemble_weights': ensemble.get_weight_info() if ensemble else {},
        })

        gc.collect()

    print("\n" + "=" * 120)
    print("v7.1 WALK-FORWARD RESULTS")
    print("=" * 120)

    for wf_idx, result in enumerate(all_wf_results, 1):
        s = result['summary']
        w = result['window']
        dd = result.get('drift_detections', 0)
        ew = result.get('ensemble_weights', {})
        print(f"  Q{wf_idx} ({w['test_start']}~{w['test_end']}): "
              f"Return={s.get('total_return', 0):+.2f}%, Sharpe={s.get('sharpe', 0):.3f}, "
              f"MaxDD={s.get('max_drawdown', 0):.2f}%, WR={s.get('win_rate', 0):.1f}%, "
              f"Trades={s.get('trade_count', 0)}, Drift={dd}, "
              f"CB/KAN={ew.get('catboost_weight', '-')}/{ew.get('kan_weight', '-')}")

    if all_wf_results:
        returns = [r['summary'].get('total_return', 0) for r in all_wf_results]
        sharps = [r['summary'].get('sharpe', 0) for r in all_wf_results]
        maxdds = [r['summary'].get('max_drawdown', 0) for r in all_wf_results]
        winrates = [r['summary'].get('win_rate', 0) for r in all_wf_results]
        total_trades = sum(r['summary'].get('trade_count', 0) for r in all_wf_results)
        positive_quarters = sum(1 for r in returns if r > 0)
        total_drifts = sum(r.get('drift_detections', 0) for r in all_wf_results)

        print(f"\n  Aggregate ({len(all_wf_results)} quarters):")
        print(f"    Avg return: {np.mean(returns):+.2f}%")
        print(f"    Median return: {np.median(returns):+.2f}%")
        print(f"    Avg Sharpe: {np.mean(sharps):.3f}")
        print(f"    Avg MaxDD: {np.mean(maxdds):.2f}%")
        print(f"    Avg WinRate: {np.mean(winrates):.1f}%")
        print(f"    Positive quarters: {positive_quarters}/{len(all_wf_results)}")
        print(f"    Total trades: {total_trades}")
        print(f"    Total drift detections: {total_drifts}")

    output_dir = Path(__file__).parent / "double_line_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = {
        'strategy': 'v7.1_ChebyKAN_GARCH_ADDM_SqrtImpact',
        'config': {
            'ssa_window': 10,
            'ssa_signal_groups': 2,
            'addm_ar_order': 3,
            'addm_ph_threshold': 2.0,
            'addm_garch_filter': True,
            'rade_gamma': 0.5,
            'mvo_scenarios': 500,
            'mvo_block_size': 5,
            'mvo_risk_aversion': 0.5,
            'catboost_l2_reg': 8,
            'catboost_depth': 4,
            'kan_type': 'ChebyKAN',
            'kan_hidden_dim': 16,
            'kan_poly_degree': 4,
            'impact_model': 'sqrt',
            'impact_coefficient': 0.4,
        },
        'wf_results': all_wf_results,
        'oamv_summary': oamv_filter.summary(),
    }

    report_file = output_dir / f"v7.1_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nResults saved: {report_file}")
    print("\n" + "=" * 120)
    print("v7.1 complete!")
    print("=" * 120)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run_backtest(stock_limit=limit)
