
import os
import sys
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import tushare as ts

load_dotenv(Path(__file__).parent.parent / ".env")

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
pro = None
if TUSHARE_TOKEN:
    try:
        pro = ts.pro_api(TUSHARE_TOKEN)
        print("Tushare init OK")
    except Exception as e:
        print(f"Tushare init failed: {e}")
        pro = None
else:
    print("Tushare Token not found")
    sys.exit(1)

STOCK_LIST = [
    ('600570.SH', '恒生电子'),
    ('600519.SH', '贵州茅台'),
    ('000001.SZ', '平安银行'),
    ('300750.SZ', '宁德时代'),
    ('601318.SH', '中国平安'),
    ('000333.SZ', '美的集团'),
    ('002415.SZ', '海康威视'),
    ('300059.SZ', '东方财富'),
]


def convert_ticker_format(ticker):
    if ticker.endswith('.SS'):
        return ticker.replace('.SS', '.SH')
    return ticker


def get_stock_daily(ticker, start_date='20210101', end_date='20241231'):
    if not pro:
        print("Tushare not initialized")
        return None

    ts_code = convert_ticker_format(ticker)

    try:
        print(f"Getting {ticker} ({ts_code}) data...")
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        if df is not None and not df.empty:
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume',
                'trade_date': 'Date'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = df[old_col]
            
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            df.set_index('Date', inplace=True)
            
            print(f"Got {len(df)} records")
            return df
        else:
            print(f"No data for {ticker}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def calculate_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    
    if 'Close' not in df.columns:
        df['Close'] = df['close']
    
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    
    df['DIF'] = ema_fast - ema_slow
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    
    cross_up = []
    cross_down = []
    
    for i in range(len(df)):
        if i == 0:
            cross_up.append(False)
            cross_down.append(False)
        else:
            macd_now = df['MACD'].iloc[i]
            macd_prev = df['MACD'].iloc[i-1]
            cross_up_val = (macd_now >= 0) and (macd_prev < 0)
            cross_down_val = (macd_now < 0) and (macd_prev >= 0)
            cross_up.append(cross_up_val)
            cross_down.append(cross_down_val)
    
    df['MACD_cross_up'] = cross_up
    df['MACD_cross_down'] = cross_down
    
    return df


def backtest_macd_strategy(df, initial_cash=1000000, position_size=0.95):
    df = calculate_macd(df)
    
    cash = initial_cash
    position = 0
    entry_price = 0
    trades = []
    equity_curve = []
    
    for date, row in df.iterrows():
        if pd.isna(row['Close']) or pd.isna(row['MACD']):
            continue
        
        current_price = row['Close']
        
        if position == 0 and row['MACD_cross_up'] and row['DIF'] > row['DEA']:
            shares = int(cash * position_size / current_price)
            if shares > 0:
                position = shares
                entry_price = current_price
                cash -= shares * current_price
                trades.append({
                    'date': date,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'reason': 'MACD up + DIF above DEA'
                })
        
        elif position > 0 and row['MACD_cross_down']:
            cash += position * current_price
            pnl = (current_price - entry_price) * position
            pnl_pct = (current_price - entry_price) / entry_price * 100
            trades.append({
                'date': date,
                'type': 'SELL',
                'price': current_price,
                'shares': position,
                'reason': 'MACD down',
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
            position = 0
            entry_price = 0
        
        equity = cash + position * current_price
        equity_curve.append({'date': date, 'equity': equity})
    
    if position > 0:
        final_price = df.iloc[-1]['Close']
        cash += position * final_price
        pnl = (final_price - entry_price) * position
        pnl_pct = (final_price - entry_price) / entry_price * 100
        trades.append({
            'date': df.index[-1],
            'type': 'SELL',
            'price': final_price,
            'shares': position,
            'reason': 'Final sell',
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })
        position = 0
    
    final_cash = cash
    return_pct = (final_cash - initial_cash) / initial_cash * 100
    
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] in ['SELL']]
    
    winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in sell_trades if t.get('pnl', 0) <= 0]
    
    return {
        'final_cash': final_cash,
        'return_pct': return_pct,
        'total_trades': len(buy_trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0,
        'trades': trades,
        'equity_curve': equity_curve
    }


def run_stock_backtest():
    print("=" * 100)
    print("Stock MACD Strategy Backtest")
    print("Strategy: MACD cross up 0 + DIF above DEA buy, MACD cross down 0 sell")
    print("=" * 100)
    
    results = []
    
    for ts_code, name in STOCK_LIST:
        df = get_stock_daily(ts_code, '20210101', '20241231')
        
        if df is None or len(df) < 60:
            print(f"{name} data insufficient")
            continue
        
        df = calculate_macd(df)
        result = backtest_macd_strategy(df)
        
        results.append({
            'code': ts_code,
            'name': name,
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'start_price': df['Close'].iloc[0],
            'end_price': df['Close'].iloc[-1],
            'return_pct': result['return_pct'],
            'total_trades': result['total_trades'],
            'win_rate': result['win_rate'],
            'winning_trades': result['winning_trades'],
            'losing_trades': result['losing_trades'],
            'final_cash': result['final_cash']
        })
        
        print(f"{name} ({ts_code}): {result['return_pct']:.2f}% return, {result['total_trades']} trades, {result['win_rate']:.1f}% win rate")
    
    if not results:
        print("No valid data")
        return []
    
    results.sort(key=lambda x: x['return_pct'], reverse=True)
    
    print("\n" + "=" * 100)
    print("Results Summary (Sorted by Return)")
    print("=" * 100)
    print(f"{'Rank':<6}{'Name':<16}{'Code':<12}{'Return':>12}{'Trades':>8}{'Win Rate':>10}")
    print("-" * 100)
    
    for i, r in enumerate(results, 1):
        print(f"{i:<6}{r['name']:<16}{r['code']:<12}{r['return_pct']:>11.2f}%{r['total_trades']:>8}{r['win_rate']:>9.1f}%")
    
    total_return = [r['return_pct'] for r in results]
    avg_return = np.mean(total_return)
    win_count = len([r for r in results if r['return_pct'] > 0])
    total_count = len(results)
    
    print("\n" + "=" * 100)
    print("Overall Statistics")
    print("=" * 100)
    print(f"Total Stocks: {total_count}")
    print(f"Positive return Stocks: {win_count} ({win_count/total_count*100:.1f}%)")
    print(f"Average return: {avg_return:.2f}%")
    print(f"Best return: {max(total_return):.2f}%")
    print(f"Worst return: {min(total_return):.2f}%")
    
    output_dir = Path(__file__).parent.parent / "results" / "double_line_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"stock_macd_{timestamp}.json"
    
    data = {
        'strategy': 'MACD Strategy',
        'description': 'MACD cross up 0 + DIF above DEA buy, cross down 0 sell',
        'backtest_period': '2021-01-01 to 2024-12-31',
        'timestamp': timestamp,
        'summary': {
            'total_stocks': total_count,
            'winning_stocks': win_count,
            'win_rate_stocks': win_count / total_count * 100,
            'average_return': avg_return
        },
        'results': results
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved: {report_file}")
    
    return results


if __name__ == "__main__":
    run_stock_backtest()

