"""
MACD上穿0轴 对比测试
对比：ULTIMATE_J0 vs ULTIMATE_J0_NO_MACD (去掉MACD上穿0轴)
"""

import os
import sys
import gc
from pathlib import Path
from datetime import datetime
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent / "pip_libs"))

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import tushare as ts

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

STRATEGY_NAMES = {
    'ULTIMATE_J0': '有MACD上穿0轴',
    'ULTIMATE_J0_NO_MACD': '无MACD上穿0轴',
}

EVAL_START = pd.Timestamp('2024-05-17')


def get_all_a_stocks():
    print("Fetching all A-share stocks...")
    try:
        stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry,list_date')
        a_stocks = stock_basic[(stock_basic['ts_code'].str.endswith('.SH')) | (stock_basic['ts_code'].str.endswith('.SZ'))]
        print(f"Total: {len(a_stocks)} stocks")
        return a_stocks
    except Exception as e:
        print(f"Failed: {e}")
        return None


def get_stock_daily(ts_code, start_date='20210101', end_date='20260519'):
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        column_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'vol': 'Volume', 'trade_date': 'Date'
        }
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)
        return df
    except:
        return None


def calculate_all_indicators(df):
    df = df.copy()
    if 'Close' not in df.columns:
        df['Close'] = df['close']

    df['white_line'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['white_line'] = df['white_line'].ewm(span=10, adjust=False).mean()
    df['ma14'] = df['Close'].rolling(window=14).mean()
    df['ma28'] = df['Close'].rolling(window=28).mean()
    df['ma57'] = df['Close'].rolling(window=57).mean()
    df['ma114'] = df['Close'].rolling(window=114).mean()
    df['yellow_line'] = (df['ma14'] + df['ma28'] + df['ma57'] + df['ma114']) / 4

    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema_fast - ema_slow
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2

    df['MACD_cross_up'] = [False] + [(df['MACD'].iloc[i] >= 0) and (df['MACD'].iloc[i-1] < 0) for i in range(1, len(df))]
    df['MACD_cross_down'] = [False] + [(df['MACD'].iloc[i] < 0) and (df['MACD'].iloc[i-1] >= 0) for i in range(1, len(df))]
    df['white_above_yellow'] = [False] + [df['white_line'].iloc[i] > df['yellow_line'].iloc[i] for i in range(1, len(df))]

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    low_list = df['Low'].rolling(window=9, min_periods=1).min()
    high_list = df['High'].rolling(window=9, min_periods=1).max()
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)
    df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df['D_val'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D_val']

    df['J_below_0_recent5'] = df['J'].rolling(window=5, min_periods=1).min() < 0
    df['J_rising'] = [False] + [(df['J'].iloc[i] > df['J'].iloc[i-1]) for i in range(1, len(df))]
    df['yellow_rising'] = [False] + [df['yellow_line'].iloc[i] > df['yellow_line'].iloc[i-1] for i in range(1, len(df))]

    dist_pct = (df['Close'] - df['yellow_line']).abs() / df['yellow_line'] * 100
    near_yellow = (dist_pct < 2.0).astype(int)
    df['sideways'] = near_yellow.rolling(window=8, min_periods=1).sum() >= 6
    df['not_sideways'] = ~df['sideways']

    df['vol_ma5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
    df['vol_above_ma5'] = df['Volume'] > df['vol_ma5']
    df['low_above_yellow'] = df['Low'] > df['yellow_line']

    return df


def backtest_strategy(df, strategy='ULTIMATE_J0', initial_cash=1000000, position_size=0.95):
    cash = initial_cash
    position = 0
    entry_price = 0
    peak_price = 0
    trade_count = 0
    win_count = 0
    total_profit = 0

    three_cond = lambda row: row.get('yellow_rising', False) and row.get('not_sideways', True) and row.get('vol_above_ma5', False)

    for i in range(len(df)):
        row = df.iloc[i]
        if df.index[i] < EVAL_START:
            continue
        if any(pd.isna(row.get(col)) for col in ['Close', 'MACD', 'white_line', 'yellow_line']):
            continue

        current_price = row['Close']

        if position == 0:
            base_j0 = row['white_above_yellow'] and row.get('J_below_0_recent5', False)
            if strategy == 'ULTIMATE_J0':
                base_j0 = row['MACD_cross_up'] and base_j0
            tc = three_cond(row)
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
            base_sell = row['MACD_cross_down'] and not row['white_above_yellow']
            drawdown = (peak_price - current_price) / peak_price * 100 if peak_price > 0 else 0
            sell_signal = base_sell or (row['RSI'] > 70 and profit_pct > 5) or (drawdown > 8 and profit_pct > 3)

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
    eval_df = df[df.index >= EVAL_START]
    buy_hold_return = (eval_df.iloc[-1]['Close'] - eval_df.iloc[0]['Close']) / eval_df.iloc[0]['Close'] * 100 if len(eval_df) > 0 else 0
    avg_trade_profit = total_profit / trade_count if trade_count > 0 else 0
    win_rate = win_count / trade_count * 100 if trade_count > 0 else 0

    return return_pct, buy_hold_return, trade_count, avg_trade_profit, win_rate


def backtest_single_stock(ts_code, name, industry=None):
    try:
        df = get_stock_daily(ts_code, '20210101', '20260519')
        if df is None or len(df) < 200:
            return None
        df = calculate_all_indicators(df)
        results = {}
        for strategy in STRATEGY_NAMES.keys():
            return_pct, buy_hold_return, trade_count, avg_trade_profit, win_rate = backtest_strategy(df, strategy)
            results[strategy] = {
                'return_pct': return_pct,
                'buy_hold_return': buy_hold_return,
                'trade_count': trade_count,
                'avg_trade_profit': avg_trade_profit,
                'win_rate': win_rate,
            }
        del df
        gc.collect()
        return {'code': ts_code, 'name': name, 'industry': industry or '', 'strategies': results, 'success': True}
    except:
        return None


def run_backtest():
    print("=" * 150)
    print("MACD上穿0轴 对比测试")
    print("=" * 150)

    stock_list = get_all_a_stocks()
    if stock_list is None:
        return
    stock_list = stock_list.head(200)
    print(f"Quick test: using first {len(stock_list)} stocks")

    all_results = []
    start_time = time.time()
    strategy_keys = list(STRATEGY_NAMES.keys())
    print(f"\nBacktesting {len(stock_list)} stocks x {len(strategy_keys)} strategies...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_stock = {
            executor.submit(backtest_single_stock, row['ts_code'], row['name'], row.get('industry', '')): row
            for _, row in stock_list.iterrows()
        }
        for i, future in enumerate(as_completed(future_to_stock), 1):
            result = future.result()
            if result is not None and result.get('success', False):
                all_results.append(result)
            if i % 500 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {i}/{len(stock_list)} ({i/len(stock_list)*100:.1f}%), Success: {len(all_results)}, Elapsed: {elapsed:.1f}s")
                gc.collect()

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 150)
    print("对比结果")
    print("=" * 150)

    strategy_stats = {}
    for strategy in strategy_keys:
        returns = [r['strategies'][strategy]['return_pct'] for r in all_results]
        buy_hold_returns = [r['strategies'][strategy]['buy_hold_return'] for r in all_results]
        trade_counts = [r['strategies'][strategy]['trade_count'] for r in all_results]
        avg_trade_profits = [r['strategies'][strategy]['avg_trade_profit'] for r in all_results]
        win_rates = [r['strategies'][strategy]['win_rate'] for r in all_results]
        positive_count = sum(1 for r in returns if r > 0)
        outperform_count = sum(1 for r in all_results if r['strategies'][strategy]['return_pct'] > r['strategies'][strategy]['buy_hold_return'])
        total = len(all_results)
        active_stocks = sum(1 for tc in trade_counts if tc > 0)

        strategy_stats[strategy] = {
            'avg_return': float(np.mean(returns)),
            'median_return': float(np.median(returns)),
            'positive_rate': positive_count / total * 100,
            'outperform_rate': outperform_count / total * 100,
            'avg_buy_hold': float(np.mean(buy_hold_returns)),
            'avg_trade_count': float(np.mean(trade_counts)),
            'avg_trade_profit': float(np.mean([p for p in avg_trade_profits if p != 0])) if any(p != 0 for p in avg_trade_profits) else 0,
            'avg_win_rate': float(np.mean([w for w in win_rates if w != 0])) if any(w != 0 for w in win_rates) else 0,
            'sharpe_like': float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
            'active_stocks': active_stocks,
        }

    print(f"\n{'Strategy':<30} {'WinRate':>10} {'TradePft':>10} {'AvgRet':>10} {'PosRate':>10} {'Trades':>8} {'Active':>8} {'Sharpe':>10}")
    print("-" * 150)
    for k in strategy_keys:
        s = strategy_stats[k]
        print(f"{k:<30} {s['avg_win_rate']:>9.1f}% {s['avg_trade_profit']:>+9.2f}% {s['avg_return']:>+9.2f}% {s['positive_rate']:>9.1f}% {s['avg_trade_count']:>7.1f} {s['active_stocks']:>7} {s['sharpe_like']:>10.3f}")

    s1 = strategy_stats['ULTIMATE_J0']
    s2 = strategy_stats['ULTIMATE_J0_NO_MACD']

    print("\n" + "=" * 150)
    print("核心差异分析")
    print("=" * 150)
    print(f"  胜率: 有MACD {s1['avg_win_rate']:.1f}% vs 无MACD {s2['avg_win_rate']:.1f}% (差 {s2['avg_win_rate']-s1['avg_win_rate']:+.1f}%)")
    print(f"  单笔均利: 有MACD {s1['avg_trade_profit']:+.2f}% vs 无MACD {s2['avg_trade_profit']:+.2f}% (差 {s2['avg_trade_profit']-s1['avg_trade_profit']:+.2f}%)")
    print(f"  平均收益: 有MACD {s1['avg_return']:+.2f}% vs 无MACD {s2['avg_return']:+.2f}% (差 {s2['avg_return']-s1['avg_return']:+.2f}%)")
    print(f"  正收益率: 有MACD {s1['positive_rate']:.1f}% vs 无MACD {s2['positive_rate']:.1f}% (差 {s2['positive_rate']-s1['positive_rate']:+.1f}%)")
    print(f"  触发股票: 有MACD {s1['active_stocks']} vs 无MACD {s2['active_stocks']} (差 {s2['active_stocks']-s1['active_stocks']:+d})")
    print(f"  夏普比: 有MACD {s1['sharpe_like']:.3f} vs 无MACD {s2['sharpe_like']:.3f} (差 {s2['sharpe_like']-s1['sharpe_like']:+.3f})")

    output_dir = Path(__file__).parent / "double_line_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = {
        'strategy_comparison': 'MACD上穿0轴对比测试',
        'strategy_stats': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()} for k, v in strategy_stats.items()},
        'elapsed_seconds': elapsed_time,
        'total_stocks': len(all_results),
    }

    report_file = output_dir / f"macd_crossup_compare_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved: {report_file}")


if __name__ == "__main__":
    run_backtest()
