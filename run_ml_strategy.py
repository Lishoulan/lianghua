"""
0AMV迟滞状态机 + CatBoost + 双线出场 三层流水线量化系统 v4
=========================================================
Stage 1: 0AMV迟滞滤波器 (非对称双阈值: +4%做多 / -2.3%做空)
Stage 2: CatBoost概率触发器 (逆小势, P>=0.65)
Stage 3: 双线出场 (个股保护线 + 系统清算线)
"""

import os
import sys
import gc
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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
DATA_END = '20260519'
OAMV_UPPER = 4.0
OAMV_LOWER = -2.3
CATBOOST_BUY_THRESHOLD = 0.65
MAX_WORKERS = 8


def get_all_a_stocks(limit=None):
    print("Fetching all A-share stocks...")
    try:
        stock_basic = pro.stock_basic(exchange='', list_status='L',
                                       fields='ts_code,symbol,name,industry,list_date')
        a_stocks = stock_basic[
            (stock_basic['ts_code'].str.endswith('.SH')) |
            (stock_basic['ts_code'].str.endswith('.SZ'))
        ]
        if limit:
            a_stocks = a_stocks.head(limit)
        print(f"Total: {len(a_stocks)} stocks")
        return a_stocks
    except Exception as e:
        print(f"Failed: {e}")
        return None


def get_index_daily(ts_code='000001.SH', start_date='20210101', end_date='20260519'):
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


def get_industry_daily(ts_code, start_date='20210101', end_date='20260519'):
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


def get_stock_daily(ts_code, start_date='20210101', end_date='20260519'):
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


def backtest_ultimate_j0(df, initial_cash=1000000, position_size=0.95):
    cash = initial_cash
    position = 0
    entry_price = 0
    peak_price = 0
    trade_count = 0
    win_count = 0
    total_profit = 0

    for i in range(len(df)):
        row = df.iloc[i]
        if df.index[i] < EVAL_START:
            continue
        if any(pd.isna(row.get(col)) for col in ['Close', 'MACD', 'white_line', 'yellow_line']):
            continue

        current_price = row['Close']

        if position == 0:
            base_j0 = row['white_above_yellow'] and row.get('J_below_0_recent5', False)
            base_j0 = row.get('MACD_cross_up', False) and base_j0
            tc = row.get('yellow_rising', False) and row.get('not_sideways', True) and row.get('vol_above_ma5', False)
            buy_signal = base_j0 and tc and row.get('J_rising', False) and row.get('low_above_yellow', True)

            if buy_signal:
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares
                    entry_price = current_price
                    peak_price = current_price
                    cash -= shares * current_price

        elif position > 0:
            if current_price > peak_price:
                peak_price = current_price
            profit_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            base_sell = row.get('MACD_cross_down', False) and not row.get('white_above_yellow', False)
            drawdown = (peak_price - current_price) / peak_price * 100 if peak_price > 0 else 0
            sell_signal = base_sell or (row.get('RSI', 0) > 70 and profit_pct > 5) or (drawdown > 8 and profit_pct > 3)

            if sell_signal:
                trade_profit = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                trade_count += 1
                if trade_profit > 0:
                    win_count += 1
                total_profit += trade_profit
                cash += position * current_price
                position = 0

    if position > 0:
        trade_profit = (df.iloc[-1]['Close'] - entry_price) / entry_price * 100 if entry_price > 0 and position > 0 else 0
        trade_count += 1
        if trade_profit > 0:
            win_count += 1
        total_profit += trade_profit
        cash += position * df.iloc[-1]['Close']
        position = 0

    return_pct = (cash - initial_cash) / initial_cash * 100
    avg_trade_profit = total_profit / trade_count if trade_count > 0 else 0
    win_rate = win_count / trade_count * 100 if trade_count > 0 else 0

    return return_pct, trade_count, avg_trade_profit, win_rate


def backtest_oamv_catboost(stock_df, featured_df, oamv_state_dict, model,
                            initial_cash=1000000, position_size=0.95):
    cash = initial_cash
    position = 0
    entry_price = 0
    entry_idx = 0
    entry_atr = 0
    peak_price = 0
    trade_count = 0
    win_count = 0
    total_profit = 0

    feature_cols = [c for c in model.FEATURE_COLS if c in featured_df.columns]

    for i in range(len(stock_df)):
        row = stock_df.iloc[i]
        date = stock_df.index[i]
        if date < EVAL_START:
            continue
        if pd.isna(row.get('Close')):
            continue

        current_price = row['Close']

        oamv_state = oamv_state_dict.get(date, 0)

        if position == 0:
            if oamv_state != 1:
                continue

            if i < len(featured_df) and date in featured_df.index:
                feat_row = featured_df.loc[[date]]
                available_cols = [c for c in feature_cols if c in feat_row.columns]
                clean = feat_row[available_cols].dropna()
                if len(clean) > 0:
                    for col in ['price_zone', 'j_zone', 'k_pattern']:
                        if col in clean.columns:
                            clean[col] = clean[col].astype(int)
                    prob = model.predict_proba(clean)[0]

                    if prob >= CATBOOST_BUY_THRESHOLD:
                        shares = int(cash * position_size / current_price)
                        if shares > 0:
                            position = shares
                            entry_price = current_price
                            entry_idx = i
                            entry_atr = row.get('ATR14', 0)
                            peak_price = current_price
                            cash -= shares * current_price

        elif position > 0:
            if current_price > peak_price:
                peak_price = current_price

            atr_val = stock_df.iloc[entry_idx].get('ATR14', 0) if entry_atr == 0 else entry_atr

            individual_stop = False
            white_val = row.get('white_line', None)
            if white_val is not None and not pd.isna(white_val):
                if current_price < white_val:
                    individual_stop = True

            trailing_atr_stop = False
            if atr_val > 0:
                drawdown_atr = (peak_price - current_price) / atr_val
                if drawdown_atr >= 1.2:
                    trailing_atr_stop = True

            system_liquidation = (oamv_state == 0)

            sell_signal = individual_stop or trailing_atr_stop or system_liquidation

            if sell_signal:
                trade_profit = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                trade_count += 1
                if trade_profit > 0:
                    win_count += 1
                total_profit += trade_profit
                cash += position * current_price
                position = 0

    if position > 0:
        trade_profit = (stock_df.iloc[-1]['Close'] - entry_price) / entry_price * 100 if entry_price > 0 and position > 0 else 0
        trade_count += 1
        if trade_profit > 0:
            win_count += 1
        total_profit += trade_profit
        cash += position * stock_df.iloc[-1]['Close']
        position = 0

    return_pct = (cash - initial_cash) / initial_cash * 100
    avg_trade_profit = total_profit / trade_count if trade_count > 0 else 0
    win_rate = win_count / trade_count * 100 if trade_count > 0 else 0

    return return_pct, trade_count, avg_trade_profit, win_rate


def process_single_stock(ts_code, name, industry, oamv_state_dict, model, discretizer,
                          index_df, industry_j_cache):
    try:
        df = get_stock_daily(ts_code, DATA_START, DATA_END)
        if df is None or len(df) < 200:
            return None

        df = compute_indicators(df)
        if df is None:
            return None

        featured_df = discretizer.transform(df)
        industry_j = industry_j_cache.get(industry)
        featured_df = discretizer.add_market_context(featured_df, index_df, None, industry_j)

        j0_ret, j0_tc, j0_atp, j0_wr = backtest_ultimate_j0(df)
        oamv_ret, oamv_tc, oamv_atp, oamv_wr = backtest_oamv_catboost(
            df, featured_df, oamv_state_dict, model
        )

        del df, featured_df
        gc.collect()

        return {
            'code': ts_code,
            'name': name,
            'industry': industry,
            'j0': {'return': j0_ret, 'trades': j0_tc, 'avg_profit': j0_atp, 'win_rate': j0_wr},
            'oamv': {'return': oamv_ret, 'trades': oamv_tc, 'avg_profit': oamv_atp, 'win_rate': oamv_wr},
            'success': True
        }
    except Exception:
        return None


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


def run_backtest(stock_limit=500):
    print("=" * 120)
    print("0AMV迟滞状态机 + CatBoost + 双线出场 三层流水线量化系统 v4")
    print("=" * 120)

    print("\n[Stage 1] 0AMV迟滞状态机 (替代HMM)")
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

    print("\n[Stage 1.5] 预加载行业ETF数据")
    print("-" * 80)

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

    print("\n[Stage 2] 收集训练样本 (CatBoost)")
    print("-" * 80)

    stock_list = get_all_a_stocks(limit=stock_limit)
    if stock_list is None:
        return

    discretizer = FeatureDiscretizer()
    labeler = TripleBarrierLabeler()
    labeler.max_hold_days = 10
    all_samples = []
    train_stock_count = 0

    train_end_idx = int(len(stock_list) * 0.6)

    for idx, (_, row) in enumerate(stock_list.iterrows()):
        if idx >= train_end_idx:
            break

        ts_code = row['ts_code']
        industry = row.get('industry', '')

        if idx % 50 == 0:
            print(f"  Collecting: {idx}/{train_end_idx}, samples: {len(all_samples)}")

        try:
            df = get_stock_daily(ts_code, DATA_START, DATA_END)
            if df is None or len(df) < 200:
                continue

            df = compute_indicators(df)
            if df is None:
                continue

            featured_df = discretizer.transform(df)
            industry_j = industry_j_cache.get(industry)
            featured_df = discretizer.add_market_context(featured_df, index_df, None, industry_j)

            oamv_stock = pd.Series(0, index=df.index, dtype=int)
            for d, state in oamv_state_dict.items():
                if d in oamv_stock.index:
                    oamv_stock.loc[d] = state

            train_mask = (df.index >= TRAIN_START) & (df.index < EVAL_START)
            state_mask = oamv_stock == 1
            combined_mask = state_mask & train_mask

            candidate_indices = [i for i in range(len(df)) if combined_mask.iloc[i]]
            if len(candidate_indices) == 0:
                del df, featured_df
                gc.collect()
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

            train_stock_count += 1
            del df, featured_df
            gc.collect()
        except Exception:
            continue

    print(f"\nTotal training samples: {len(all_samples)} from {train_stock_count} stocks")

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

    print("\n[Stage 2] 训练CatBoost模型")
    print("-" * 80)

    model = CatBoostPredictor(buy_threshold=CATBOOST_BUY_THRESHOLD)
    model.train(X_train, y_train, X_val, y_val)

    importance = model.feature_importance()
    if importance:
        print("\nFeature Importance:")
        for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
            bar = '#' * (int(imp) // 2)
            print(f"  {name:>22s}: {imp:7.2f} {bar}")

    val_probs = model.predict_proba(X_val)
    val_preds = val_probs >= CATBOOST_BUY_THRESHOLD
    val_accuracy = (val_preds == y_val).mean()
    val_pos_precision = y_val[val_preds].mean() if val_preds.sum() > 0 else 0
    print(f"\nValidation: accuracy={val_accuracy:.2%}, precision@{CATBOOST_BUY_THRESHOLD}={val_pos_precision:.2%}")
    print(f"Val >= {CATBOOST_BUY_THRESHOLD}: {val_preds.sum()}/{len(val_preds)}")
    print(f"Prob dist: min={val_probs.min():.4f}, max={val_probs.max():.4f}, "
          f"mean={val_probs.mean():.4f}, std={val_probs.std():.4f}")
    print(f"  P>=0.50: {(val_probs >= 0.50).sum()}, P>=0.55: {(val_probs >= 0.55).sum()}, "
          f"P>=0.60: {(val_probs >= 0.60).sum()}, P>=0.65: {(val_probs >= 0.65).sum()}, "
          f"P>=0.70: {(val_probs >= 0.70).sum()}")

    if val_preds.sum() > 0:
        print(f"  When P>={CATBOOST_BUY_THRESHOLD}: actual win rate = {y_val[val_preds].mean():.2%}")

    print("\n[Stage 3] 全市场回测 (双线出场)")
    print("-" * 80)

    all_results = []
    start_time = time.time()

    def process_wrapper(args):
        ts_code, name, industry = args
        return process_single_stock(ts_code, name, industry, oamv_state_dict, model, discretizer,
                                    index_df, industry_j_cache)

    stock_args = [(row['ts_code'], row['name'], row.get('industry', '')) for _, row in stock_list.iterrows()]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_stock = {executor.submit(process_wrapper, args): args for args in stock_args}
        for i, future in enumerate(as_completed(future_to_stock), 1):
            result = future.result()
            if result is not None and result.get('success', False):
                all_results.append(result)
            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {i}/{len(stock_args)} ({i/len(stock_args)*100:.1f}%), "
                      f"Success: {len(all_results)}, Elapsed: {elapsed:.1f}s")
                gc.collect()

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 120)
    print("回测结果对比: ULTIMATE_J0 vs 0AMV+CatBoost+双线出场")
    print("=" * 120)

    j0_returns = [r['j0']['return'] for r in all_results]
    j0_win_rates = [r['j0']['win_rate'] for r in all_results if r['j0']['trades'] > 0]
    j0_avg_profits = [r['j0']['avg_profit'] for r in all_results if r['j0']['trades'] > 0]
    j0_active = sum(1 for r in all_results if r['j0']['trades'] > 0)

    oamv_returns = [r['oamv']['return'] for r in all_results]
    oamv_win_rates = [r['oamv']['win_rate'] for r in all_results if r['oamv']['trades'] > 0]
    oamv_avg_profits = [r['oamv']['avg_profit'] for r in all_results if r['oamv']['trades'] > 0]
    oamv_active = sum(1 for r in all_results if r['oamv']['trades'] > 0)

    total = len(all_results)

    print(f"\n{'Metric':<25} {'ULTIMATE_J0':>15} {'0AMV+CatBoost':>15}")
    print("-" * 60)

    j0_avg_ret = float(np.mean(j0_returns)) if j0_returns else 0
    oamv_avg_ret = float(np.mean(oamv_returns)) if oamv_returns else 0
    print(f"{'Avg Return':<25} {j0_avg_ret:>+14.2f}% {oamv_avg_ret:>+14.2f}%")

    j0_pos_rate = sum(1 for r in j0_returns if r > 0) / total * 100 if total > 0 else 0
    oamv_pos_rate = sum(1 for r in oamv_returns if r > 0) / total * 100 if total > 0 else 0
    print(f"{'Positive Rate':<25} {j0_pos_rate:>14.1f}% {oamv_pos_rate:>14.1f}%")

    j0_wr = float(np.mean(j0_win_rates)) if j0_win_rates else 0
    oamv_wr = float(np.mean(oamv_win_rates)) if oamv_win_rates else 0
    print(f"{'Avg Win Rate':<25} {j0_wr:>13.1f}% {oamv_wr:>13.1f}%")

    j0_ap = float(np.mean(j0_avg_profits)) if j0_avg_profits else 0
    oamv_ap = float(np.mean(oamv_avg_profits)) if oamv_avg_profits else 0
    print(f"{'Avg Trade Profit':<25} {j0_ap:>+14.2f}% {oamv_ap:>+14.2f}%")

    print(f"{'Active Stocks':<25} {j0_active:>14} {oamv_active:>14}")

    j0_sharpe = float(np.mean(j0_returns) / np.std(j0_returns)) if np.std(j0_returns) > 0 else 0
    oamv_sharpe = float(np.mean(oamv_returns) / np.std(oamv_returns)) if np.std(oamv_returns) > 0 else 0
    print(f"{'Sharpe-like':<25} {j0_sharpe:>14.3f} {oamv_sharpe:>14.3f}")

    oamv_better = sum(1 for r in all_results if r['oamv']['return'] > r['j0']['return'])
    print(f"\n0AMV+CatBoost outperforms J0: {oamv_better} stocks")

    if oamv_active > 0:
        oamv_active_results = [r for r in all_results if r['oamv']['trades'] > 0]
        oamv_active_rets = [r['oamv']['return'] for r in oamv_active_results]
        print(f"\n0AMV Active Stock Stats ({oamv_active} stocks):")
        print(f"  Avg return: {np.mean(oamv_active_rets):+.2f}%")
        print(f"  Median return: {np.median(oamv_active_rets):+.2f}%")
        print(f"  Best: {max(oamv_active_rets):+.2f}%, Worst: {min(oamv_active_rets):+.2f}%")
        print(f"  Win rate (active): {sum(1 for r in oamv_active_rets if r > 0)/len(oamv_active_rets)*100:.1f}%")

    output_dir = Path(__file__).parent / "double_line_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = {
        'strategy': '0AMV_Hysteresis_CatBoost_DualExit_v4',
        'config': {
            'oamv_upper': OAMV_UPPER,
            'oamv_lower': OAMV_LOWER,
            'catboost_buy_threshold': CATBOOST_BUY_THRESHOLD,
            'eval_start': str(EVAL_START),
            'train_start': str(TRAIN_START),
            'stock_limit': stock_limit,
            'exit_logic': {
                'individual_stop': 'close < white_line (zx_st)',
                'trailing_atr_stop': 'drawdown_from_peak >= 1.2 * ATR',
                'system_liquidation': 'oamv_state == 0',
            },
        },
        'j0_stats': {
            'avg_return': j0_avg_ret, 'positive_rate': j0_pos_rate,
            'avg_win_rate': j0_wr, 'avg_trade_profit': j0_ap,
            'active_stocks': j0_active, 'sharpe': j0_sharpe,
        },
        'oamv_stats': {
            'avg_return': oamv_avg_ret, 'positive_rate': oamv_pos_rate,
            'avg_win_rate': oamv_wr, 'avg_trade_profit': oamv_ap,
            'active_stocks': oamv_active, 'sharpe': oamv_sharpe,
        },
        'training_samples': len(all_samples),
        'feature_importance': importance,
        'total_stocks': total,
        'elapsed_seconds': elapsed_time,
    }

    report_file = output_dir / f"oamv_catboost_v4_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nResults saved: {report_file}")
    print("\n" + "=" * 120)
    print("系统运行完毕")
    print("=" * 120)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    run_backtest(stock_limit=limit)
