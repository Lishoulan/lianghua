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

# Disable proxies
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

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

load_dotenv(Path(__file__).parent / ".env")

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
SERVERCHAN_KEY = os.getenv('SERVERCHAN_KEY', '')

pro = None
if TUSHARE_TOKEN:
    try:
        pro = ts.pro_api(TUSHARE_TOKEN)
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


def is_st_stock(name: str) -> bool:
    for kw in ['ST', '*ST', 'S*ST', 'SST']:
        if name.startswith(kw):
            return True
    return False


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
    if not SERVERCHAN_KEY:
        print("SERVERCHAN_KEY not configured, skipping push")
        return False
    url = f"https://sctapi.ftqq.com/{SERVERCHAN_KEY}.send"
    data = {"title": title, "desp": desp}
    try:
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            if result.get('code') == 0:
                print("Server酱推送成功")
                return True
            else:
                print(f"Server酱推送失败: {result}")
        else:
            print(f"Server酱HTTP错误: {resp.status_code}")
    except Exception as e:
        print(f"Server酱推送异常: {e}")
    return False


def run_daily_scan():
    today = datetime.now().strftime('%Y%m%d')
    today_display = datetime.now().strftime('%Y-%m-%d')

    print(f"[{today_display}] v7.1 每日扫描启动...")

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
    portfolio_optimizer = BootstrappedMVO(
        n_scenarios=500, block_size=5, lookback_days=200,
        risk_aversion=0.5, max_weight=0.25, min_weight=0.0, total_max_weight=0.75
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
    oamv_daily = oamv_filter.is_trading_allowed(last_date, require_weekly=False)
    oamv_weekly = oamv_filter.is_trading_allowed(last_date, require_weekly=True)
    is_panic = panic_breaker.is_panic(last_date)

    oamv_x = 0.0
    if last_date in oamv_state_df.index:
        oamv_x = float(oamv_state_df.loc[last_date, 'oamv_x'])

    index_close = float(index_df.loc[last_date, 'Close'])
    index_change = 0.0
    if len(index_df) > 1:
        prev_close = float(index_df['Close'].iloc[-2])
        if prev_close > 0:
            index_change = (index_close - prev_close) / prev_close * 100

    buy_signals = []
    if oamv_weekly and not is_panic:
        for ts_code, info in all_stock_data.items():
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
                amp = df.loc[last_date, 'amplitude_20'] if 'amplitude_20' in df.columns else 0
                if pd.notna(amp) and amp > 0 and amp < FRICTION_COST_PCT * MIN_AMPLITUDE_MULT:
                    continue
                daily_vol = float(df.loc[last_date, 'amplitude_20']) / 100 if 'amplitude_20' in df.columns and pd.notna(df.loc[last_date, 'amplitude_20']) else 0.02
                daily_vol_val = float(df.loc[last_date, 'Volume']) if 'Volume' in df.columns and pd.notna(df.loc[last_date, 'Volume']) else 0
                order_shares = int(INITIAL_CASH * POSITION_SIZE_PCT / float(df.loc[last_date, 'Close']) / 100) * 100
                if daily_vol_val > 0 and order_shares > 0:
                    participation = order_shares / daily_vol_val
                    impact_slippage = IMPACT_COEFFICIENT * daily_vol * np.sqrt(participation) + SPREAD_HALF
                    impact_slippage = min(impact_slippage, MAX_SLIPPAGE_PCT / 100)
                else:
                    impact_slippage = 0.0005
                if impact_slippage > 0.005:
                    continue
                buy_signals.append({
                    'ts_code': ts_code,
                    'name': info['name'],
                    'prob': prob,
                    'price': float(df.loc[last_date, 'Close']),
                    'atr': float(df.loc[last_date, 'ATR14']) if 'ATR14' in df.columns and not pd.isna(df.loc[last_date, 'ATR14']) else 0,
                    'industry': info.get('industry', ''),
                    'impact_slippage': float(impact_slippage),
                })

    buy_signals.sort(key=lambda x: -x['prob'])

    mvo_info = ""
    top_n = min(5, len(buy_signals))
    if top_n >= 2:
        candidate_codes = [c['ts_code'] for c in buy_signals[:top_n]]
        ml_probs = {c['ts_code']: c['prob'] for c in buy_signals[:top_n]}
        try:
            weights, valid_codes = portfolio_optimizer.optimize(
                candidate_codes, all_stock_data, oamv_state_df, last_date, ml_probs
            )
            if len(valid_codes) > 0 and len(weights) > 0:
                for code, w in zip(valid_codes, weights):
                    for s in buy_signals:
                        if s['ts_code'] == code:
                            s['mvo_weight'] = w
                            break
                buy_signals.sort(key=lambda x: -x.get('mvo_weight', 0))
                mvo_info = "✅ MVO优化完成"
        except Exception:
            mvo_info = "⚠️ MVO优化失败，按概率排序"

    oamv_status = "🟢 BULL" if oamv_daily else "🔴 BEAR"
    oamv_weekly_status = "🟢 BULL" if oamv_weekly else "🔴 BEAR"
    panic_status = "⚠️ 触发" if is_panic else "✅ 正常"

    title = f"v7.1每日信号 {today_display}"

    desp = f"## 📊 v7.1 每日交易信号\n\n"
    desp += f"**日期**: {today_display}\n\n"
    desp += f"### 大势判断\n\n"
    desp += f"| 指标 | 状态 |\n|------|------|\n"
    desp += f"| 0AMV日线 | {oamv_status} (X={oamv_x:+.2f}%) |\n"
    desp += f"| 0AMV周线 | {oamv_weekly_status} |\n"
    desp += f"| 熔断器 | {panic_status} |\n"
    desp += f"| 上证指数 | {index_close:.2f} ({index_change:+.2f}%) |\n\n"

    if not oamv_weekly:
        desp += f"### ⏸️ 操作建议：空仓观望\n\n"
        desp += f"0AMV周线BEAR，不建议买入。等待周线翻多信号。\n\n"
    elif is_panic:
        desp += f"### ⚠️ 操作建议：熔断器触发\n\n"
        desp += f"市场出现恐慌信号，建议清仓避险。\n\n"
    else:
        desp += f"### 🟢 操作建议：可买入\n\n"
        if buy_signals:
            desp += f"**扫描到 {len(buy_signals)} 只候选股**（RADE概率≥65%）\n\n"
            desp += f"| 排名 | 代码 | 名称 | 行业 | 现价 | RADE概率 | ATR | MVO权重 | 冲击滑点 |\n"
            desp += f"|------|------|------|------|------|----------|-----|--------|----------|\n"
            for i, s in enumerate(buy_signals[:10], 1):
                mvo_w = f"{s['mvo_weight']:.2f}" if s.get('mvo_weight') is not None else "-"
                impact_s = f"{s['impact_slippage']:.2%}" if s.get('impact_slippage') is not None else "-"
                desp += f"| {i} | {s['ts_code']} | {s['name']} | {s['industry']} | {s['price']:.2f} | {s['prob']:.1%} | {s['atr']:.2f} | {mvo_w} | {impact_s} |\n"
            desp += f"\n{mvo_info}\n\n"
            desp += f"**建议买入前3只**（MVO权重最高）\n\n"
        else:
            desp += f"今日无符合条件的买入信号。\n\n"

    drift_alert_msg = ""
    if drift_detector.drift_detected:
        drift_alert_msg = f"\n\n### ⚠️ ADDM漂移告警\n\n[ALERT] 检测到市场概念漂移（ADDM触发）！系统已启动紧急重新训练... 漂移次数: {drift_detector.drift_count}\n"
        if SERVERCHAN_KEY:
            send_serverchan(f"[ALERT] ADDM漂移检测 {today_display}", drift_alert_msg)

    desp += f"{drift_alert_msg}"

    desp += f"---\n\n"
    desp += f"*策略: v7.1 ChebyKAN+GARCH-ADDM+SqrtImpact | 数据: Tushare | 本信号仅供参考，不构成投资建议*\n"

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"0AMV日线: {oamv_status} (X={oamv_x:+.2f}%)")
    print(f"0AMV周线: {oamv_weekly_status}")
    print(f"熔断器: {panic_status}")
    print(f"上证: {index_close:.2f} ({index_change:+.2f}%)")

    if buy_signals:
        print(f"\n买入信号 ({len(buy_signals)} 只):")
        for i, s in enumerate(buy_signals[:10], 1):
            mvo_w = f" MVO={s['mvo_weight']:.2f}" if s.get('mvo_weight') is not None else ""
            print(f"  {i}. {s['ts_code']} {s['name']} {s['price']:.2f} P={s['prob']:.1%}{mvo_w}")
    else:
        print("\n无买入信号")

    send_serverchan(title, desp)

    output_dir = Path(__file__).parent / "daily_scan_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"scan_{today}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'date': today_display,
            'oamv_daily': oamv_daily,
            'oamv_weekly': oamv_weekly,
            'oamv_x': oamv_x,
            'is_panic': is_panic,
            'index_close': index_close,
            'index_change': index_change,
            'buy_signals': buy_signals[:10],
            'push_sent': bool(SERVERCHAN_KEY),
            'drift_detected': drift_detector.drift_detected,
            'drift_count': drift_detector.drift_count,
            'market_volatility': market_vol,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {result_file}")


if __name__ == "__main__":
    run_daily_scan()
