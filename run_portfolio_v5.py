import os
import sys
import gc
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

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

EVAL_START = pd.Timestamp('2024-05-17')
TRAIN_START = pd.Timestamp('2021-01-01')
DATA_START = '20210101'
DATA_END = '20260520'
OAMV_UPPER = 4.0
OAMV_LOWER = -2.3
CATBOOST_BUY_THRESHOLD = 0.65
MAX_PORTFOLIO_STOCKS = 3
MAX_WORKERS = 8

COMMISSION_RATE = 0.0003
STAMP_DUTY_RATE = 0.001
SLIPPAGE_RATE = 0.0005
POSITION_SIZE_PCT = 0.25

MIN_HOLD_DAYS = 3
COOLDOWN_DAYS = 5
MIN_PRICE = 3.0
MIN_MARKET_CAP_HINT = 5.0


def is_st_stock(name: str) -> bool:
    st_keywords = ['ST', '*ST', 'S*ST', 'SST', 'S']
    for kw in st_keywords:
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
            print(f"Filtered ST stocks: {before} -> {len(a_stocks)} (removed {before - len(a_stocks)})")
        if limit:
            a_stocks = a_stocks.head(limit)
        print(f"Total: {len(a_stocks)} stocks")
        return a_stocks
    except Exception as e:
        print(f"Failed: {e}")
        return None


def get_index_daily(ts_code='000001.SH', start_date='20210101', end_date='20260520'):
    print(f"Fetching index {ts_code} data...")
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
        print(f"Index data: {len(df)} rows, {df.index[0]} to {df.index[-1]}")
        return df
    except Exception as e:
        print(f"Failed to fetch index: {e}")
        return None


def get_industry_daily(ts_code, start_date='20210101', end_date='20260520'):
    try:
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


INDUSTRY_ETF_MAP = {
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


def load_and_preprocess_all_data(stock_list):
    print("Loading and preprocessing all data...")
    all_stock_data = {}
    all_featured_data = {}
    index_df = get_index_daily('000001.SH', DATA_START, DATA_END)

    if index_df is None:
        return None, None, None, None

    index_df = compute_indicators(index_df)

    print("Loading industry ETFs...")
    industry_j_cache = {}
    for ind_name, etf_code in INDUSTRY_ETF_MAP.items():
        try:
            ind_df = get_industry_daily(etf_code, DATA_START, DATA_END)
            if ind_df is not None and len(ind_df) > 20:
                ind_j = compute_industry_j(ind_df)
                if ind_j is not None:
                    industry_j_cache[ind_name] = ind_j
        except Exception:
            pass
    print(f"Loaded {len(industry_j_cache)} industry ETFs")

    discretizer = FeatureDiscretizer()

    total = len(stock_list)
    for idx, (_, row) in enumerate(stock_list.iterrows(), 1):
        ts_code = row['ts_code']
        name = row['name']
        industry = row.get('industry', '')

        if idx % 50 == 0:
            print(f"  Progress: {idx}/{total} ({idx/total*100:.1f}%)")

        try:
            df = get_stock_daily(ts_code, DATA_START, DATA_END)
            if df is None or len(df) < 200:
                continue

            if df['Close'].iloc[-1] < MIN_PRICE:
                continue

            df = compute_indicators(df)
            if df is None:
                continue

            all_stock_data[ts_code] = {
                'data': df,
                'name': name,
                'industry': industry,
            }

            featured_df = discretizer.transform(df)
            industry_j = industry_j_cache.get(industry)
            featured_df = discretizer.add_market_context(featured_df, index_df, None, industry_j)
            all_featured_data[ts_code] = featured_df

        except Exception:
            continue

    print(f"Loaded {len(all_stock_data)} stocks with features")
    return all_stock_data, all_featured_data, index_df, industry_j_cache


def collect_training_samples(all_stock_data, all_featured_data, oamv_state_dict):
    print("Collecting training samples...")
    all_samples = []
    discretizer = FeatureDiscretizer()
    labeler = TripleBarrierLabeler()
    labeler.max_hold_days = 10

    ts_codes = list(all_stock_data.keys())

    split_idx = int(len(ts_codes) * 0.6)
    train_ts_codes = ts_codes[:split_idx]

    for idx, ts_code in enumerate(train_ts_codes, 1):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(train_ts_codes)}, samples: {len(all_samples)}")

        stock_info = all_stock_data[ts_code]
        df = stock_info['data']
        industry = stock_info['industry']
        featured_df = all_featured_data[ts_code]

        try:
            oamv_stock = pd.Series(0, index=df.index, dtype=int)
            for d, state in oamv_state_dict.items():
                if d in oamv_stock.index:
                    oamv_stock.loc[d] = state

            train_mask = (df.index >= TRAIN_START) & (df.index < EVAL_START)
            state_mask = oamv_stock == 1
            combined_mask = state_mask & train_mask

            candidate_indices = [i for i in range(len(df)) if combined_mask.iloc[i]]
            if len(candidate_indices) == 0:
                continue

            atr14 = df['ATR14'] if 'ATR14' in df.columns else pd.Series(np.nan, index=df.index)
            labels = labeler.label_all(df, candidate_indices, atr14)

            feature_cols = CatBoostPredictor.FEATURE_COLS

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
                sample['ts_code'] = ts_code
                all_samples.append(sample)

        except Exception:
            continue

    print(f"Total training samples: {len(all_samples)} from {len(train_ts_codes)} stocks")
    return all_samples


def run_portfolio_backtest(all_stock_data, all_featured_data, model, oamv_state_dict):
    print("Running portfolio backtest (v5.1: ST filtered + relaxed exit + cooldown)...")
    backtester = PortfolioBacktester(
        initial_cash=10000000,
        max_stocks=MAX_PORTFOLIO_STOCKS,
        commission_rate=COMMISSION_RATE,
        stamp_duty_rate=STAMP_DUTY_RATE,
        slippage_rate=SLIPPAGE_RATE,
        position_size_pct=POSITION_SIZE_PCT,
        catboost_threshold=CATBOOST_BUY_THRESHOLD,
    )

    all_dates = set()
    for ts_code, info in all_stock_data.items():
        all_dates.update(info['data'].index)
    eval_dates = [d for d in sorted(all_dates) if d >= EVAL_START]

    feature_cols = CatBoostPredictor.FEATURE_COLS

    recent_sell_dates: Dict[str, pd.Timestamp] = {}

    for date_idx, current_date in enumerate(eval_dates, 1):
        if date_idx % 20 == 0:
            stock_prices_now = {}
            for ts_code, info in all_stock_data.items():
                df = info['data']
                if current_date in df.index:
                    stock_prices_now[ts_code] = df.loc[current_date, 'Close']
            pv = backtester.get_portfolio_value(stock_prices_now)
            print(f"  Progress: {date_idx}/{len(eval_dates)}, "
                  f"Portfolio: {pv:,.0f}, Positions: {len(backtester.positions)}")

        oamv_state = oamv_state_dict.get(current_date, 0)
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

            individual_stop = False
            white_line = row.get('white_line')
            yellow_line = row.get('yellow_line')
            if white_line is not None and not pd.isna(white_line):
                if yellow_line is not None and not pd.isna(yellow_line):
                    if current_price < yellow_line:
                        individual_stop = True
                else:
                    if current_price < white_line * 0.97:
                        individual_stop = True

            trailing_stop_pct = False
            if pos.entry_price > 0:
                drawdown_pct = (pos.peak_price - current_price) / pos.peak_price * 100
                if drawdown_pct >= 8.0:
                    trailing_stop_pct = True

            trailing_atr_stop = False
            if pos.entry_atr > 0:
                drawdown_atr = (pos.peak_price - current_price) / pos.entry_atr
                if drawdown_atr >= 1.5:
                    trailing_atr_stop = True

            system_liquidation = (oamv_state == 0)

            sell_signal = individual_stop or trailing_stop_pct or trailing_atr_stop or system_liquidation

            if sell_signal:
                reason = []
                if individual_stop:
                    reason.append('close < yellow_line')
                if trailing_stop_pct:
                    reason.append('8% trailing')
                if trailing_atr_stop:
                    reason.append('1.5x ATR trailing')
                if system_liquidation:
                    reason.append('system_liquidation')

                backtester.sell(ts_code, current_price, current_date, reason=', '.join(reason))
                recent_sell_dates[ts_code] = current_date

        if oamv_state == 1 and backtester.can_buy():
            buy_candidates = []

            for ts_code, info in all_stock_data.items():
                if ts_code in backtester.positions:
                    continue

                if ts_code in recent_sell_dates:
                    days_since_sell = (current_date - recent_sell_dates[ts_code]).days
                    if days_since_sell < COOLDOWN_DAYS:
                        continue

                df = info['data']
                featured_df = all_featured_data[ts_code]
                name = info['name']

                if current_date not in df.index or current_date not in featured_df.index:
                    continue

                row_idx = df.index.get_loc(current_date)
                feat_row = featured_df.loc[current_date]

                available_cols = [c for c in feature_cols if c in feat_row.index]
                if any(pd.isna(feat_row.get(c)) for c in available_cols):
                    continue

                clean_data = feat_row[available_cols].to_frame().T
                for col in ['price_zone', 'j_zone', 'k_pattern']:
                    if col in clean_data.columns:
                        clean_data[col] = clean_data[col].astype(int)

                prob = model.predict_proba(clean_data)[0]

                if prob >= CATBOOST_BUY_THRESHOLD:
                    buy_candidates.append({
                        'ts_code': ts_code,
                        'name': name,
                        'prob': prob,
                        'price': df.loc[current_date, 'Close'],
                        'atr': df.loc[current_date, 'ATR14'],
                    })

            buy_candidates.sort(key=lambda x: -x['prob'])
            top_n_candidates = buy_candidates[:backtester.get_available_slot_count() * 2]

            for candidate in top_n_candidates:
                if not backtester.can_buy():
                    break

                ts_code = candidate['ts_code']
                name = candidate['name']
                price = candidate['price']
                prob = candidate['prob']
                atr = candidate.get('atr', 0.0)

                if pd.isna(price) or price <= 0:
                    continue

                backtester.buy(ts_code, name, price, current_date, prob, atr)

        backtester.update_equity(current_date, stock_prices)

    backtester.finalize(eval_dates[-1], stock_prices)
    return backtester


def run_backtest(stock_limit=500):
    print("=" * 120)
    print("0AMV + CatBoost + 双线出场 v5.1 (ST过滤 + 放宽出场 + 冷却期)")
    print("=" * 120)
    print(f"\nConfig:")
    print(f"  Initial cash: 10,000,000")
    print(f"  Max portfolio stocks: {MAX_PORTFOLIO_STOCKS}")
    print(f"  Position size: {POSITION_SIZE_PCT*100:.0f}% per stock")
    print(f"  Commission: {COMMISSION_RATE*10000:.1f}bp, Stamp: {STAMP_DUTY_RATE*10000:.1f}bp, Slippage: {SLIPPAGE_RATE*10000:.1f}bp")
    print(f"  CatBoost buy threshold: P >= {CATBOOST_BUY_THRESHOLD}")
    print(f"  Min hold days: {MIN_HOLD_DAYS}")
    print(f"  Cooldown after sell: {COOLDOWN_DAYS} days")
    print(f"  Min stock price: {MIN_PRICE}")
    print(f"  Exit rules: (close < yellow_line) OR (8% trailing) OR (1.5x ATR trailing) OR (oamv_state == 0)")

    print("\n[Stage 1] 0AMV迟滞状态机")
    print("-" * 80)

    index_df = get_index_daily('000001.SH', DATA_START, DATA_END)
    if index_df is None:
        print("Failed to fetch index data!")
        return

    index_df = compute_indicators(index_df)

    oamv_filter = OAMVHysteresisFilter(
        upper_threshold=OAMV_UPPER,
        lower_threshold=OAMV_LOWER,
        cost_ma_period=34,
    )
    oamv_filter.fit(index_df)
    print(oamv_filter.summary())

    oamv_state_df = oamv_filter.get_state_df()
    oamv_state_dict = oamv_filter.get_state_dict()

    transitions = oamv_filter.get_transition_dates()
    print(f"\n0AMV State Transitions (total {len(transitions)}):")
    for t in transitions[-10:]:
        direction = "→ BULL" if t['to'] == 1 else "→ BEAR"
        print(f"  {t['date'].strftime('%Y-%m-%d')}: State {t['from']} {direction} (X={t['x_value']:.2f}%)")

    eval_oamv = oamv_state_df[oamv_state_df.index >= EVAL_START]
    eval_bullish = (eval_oamv['oamv_state'] == 1).sum()
    eval_total = len(eval_oamv)
    print(f"\nEvaluation period: {eval_bullish}/{eval_total} bullish days ({eval_bullish/eval_total*100:.1f}%)")

    stock_list = get_all_a_stocks(limit=stock_limit, filter_st=True)
    if stock_list is None:
        return

    all_stock_data, all_featured_data, index_df, industry_j_cache = load_and_preprocess_all_data(stock_list)
    if all_stock_data is None:
        print("Failed to load data!")
        return

    print("\n[Stage 2] Training CatBoost model")
    print("-" * 80)

    all_samples = collect_training_samples(all_stock_data, all_featured_data, oamv_state_dict)

    if len(all_samples) < 100:
        print("Not enough training samples!")
        return

    samples_df = pd.DataFrame(all_samples)
    feature_cols = CatBoostPredictor.FEATURE_COLS

    available_cols = [c for c in feature_cols if c in samples_df.columns]
    X_all = samples_df[available_cols].dropna()
    y_all = samples_df.loc[X_all.index, 'label'].values

    split_idx = int(len(X_all) * 0.8)
    X_train = X_all.iloc[:split_idx]
    y_train = y_all[:split_idx]
    X_val = X_all.iloc[split_idx:]
    y_val = y_all[split_idx:]

    print(f"Train: {len(X_train)} (pos: {y_train.mean():.2%}), Val: {len(X_val)} (pos: {y_val.mean():.2%})")

    model = CatBoostPredictor(buy_threshold=CATBOOST_BUY_THRESHOLD)
    model.train(X_train, y_train, X_val, y_val)

    model_dir = Path(__file__).parent / "ml_strategy" / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "catboost_latest.cbm"))

    importance = model.feature_importance()
    if importance:
        print("\nFeature Importance:")
        for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
            bar = '#' * (int(imp) // 2)
            print(f"  {name:>22s}: {imp:7.2f} {bar}")

    val_probs = model.predict_proba(X_val)
    val_preds = val_probs >= CATBOOST_BUY_THRESHOLD
    val_accuracy = (val_preds == y_val).mean()
    val_pos_precision = y_val[val_preds].mean() if val_preds.sum() > 0 else 0
    print(f"\nValidation: accuracy={val_accuracy:.2%}, precision@{CATBOOST_BUY_THRESHOLD}={val_pos_precision:.2%}")
    print(f"Val >= {CATBOOST_BUY_THRESHOLD}: {val_preds.sum()}/{len(val_preds)}")
    print(f"Prob range: [{val_probs.min():.3f}, {val_probs.max():.3f}], std={val_probs.std():.3f}")
    if val_preds.sum() > 0:
        print(f"  When P>={CATBOOST_BUY_THRESHOLD}: actual win rate = {y_val[val_preds].mean():.2%}")

    print("\n[Stage 3] Portfolio backtest (v5.1)")
    print("-" * 80)
    print(f"  Universe size: {len(all_stock_data)} stocks (ST filtered)")

    backtester = run_portfolio_backtest(all_stock_data, all_featured_data, model, oamv_state_dict)

    summary = backtester.get_summary()

    print("\n" + "=" * 120)
    print("PORTFOLIO BACKTEST RESULTS (v5.1)")
    print("=" * 120)
    print()
    print(f"Initial cash:  10,000,000.00")
    print(f"Final equity: {summary.get('final_equity', 0):,.2f}")
    print()
    print(f"Total return:       {summary.get('total_return', 0):>+10.2f}%")
    print(f"Annualized return:  {summary.get('annual_return', 0):>+10.2f}%")
    print(f"Volatility (annual):{summary.get('volatility', 0):>10.2f}%")
    print(f"Sharpe ratio:       {summary.get('sharpe', 0):>10.3f}")
    print(f"Max drawdown:       {summary.get('max_drawdown', 0):>10.2f}%")
    print()
    print(f"Trade count:        {summary.get('trade_count', 0):>10}")
    print(f"Win rate:           {summary.get('win_rate', 0):>10.1f}%")
    print(f"Avg trade profit:   {summary.get('avg_trade_profit', 0):>10.2f}%")
    print(f"Avg win profit:     {summary.get('avg_win', 0):>10.2f}%")
    print(f"Avg loss profit:    {summary.get('avg_loss', 0):>10.2f}%")
    print(f"Profit factor:      {summary.get('profit_factor', 0):>10.2f}")
    print(f"Avg hold days:      {summary.get('avg_hold_days', 0):>10.1f}")

    print("\nRecent trades:")
    trade_details = backtester.get_trade_details()
    for t in trade_details[-15:]:
        emoji = '+' if t['profit_pct'] > 0 else '-'
        print(f"  [{emoji}] {t['code']:>10s} {t['name']:<10s} {t['entry_date']} -> {t['exit_date']} "
              f"({t['hold_days']}d) {t['profit_pct']:>+8.2f}% {t['exit_reason']}")

    output_dir = Path(__file__).parent / "double_line_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = {
        'strategy': '0AMV_CatBoost_DualExit_Portfolio_v5.1',
        'config': {
            'initial_cash': 10000000,
            'max_portfolio_stocks': MAX_PORTFOLIO_STOCKS,
            'position_size_pct': POSITION_SIZE_PCT,
            'commission_rate': COMMISSION_RATE,
            'stamp_duty_rate': STAMP_DUTY_RATE,
            'slippage_rate': SLIPPAGE_RATE,
            'oamv_upper': OAMV_UPPER,
            'oamv_lower': OAMV_LOWER,
            'catboost_buy_threshold': CATBOOST_BUY_THRESHOLD,
            'min_hold_days': MIN_HOLD_DAYS,
            'cooldown_days': COOLDOWN_DAYS,
            'min_price': MIN_PRICE,
            'filter_st': True,
            'eval_start': str(EVAL_START),
            'train_start': str(TRAIN_START),
            'stock_limit': stock_limit,
            'exit_rules': [
                'close < yellow_line (not white_line!)',
                '8% trailing stop from peak',
                '1.5x ATR trailing stop from peak',
                'oamv_state == 0 (system liquidation)',
            ],
        },
        'summary': summary,
        'trades': trade_details,
        'feature_importance': importance,
        'oamv_summary': oamv_filter.summary(),
    }

    report_file = output_dir / f"portfolio_v5.1_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nResults saved: {report_file}")
    print("\n" + "=" * 120)
    print("System run complete!")
    print("=" * 120)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run_backtest(stock_limit=limit)
