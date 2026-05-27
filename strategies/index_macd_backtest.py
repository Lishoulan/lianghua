
"""
指数MACD策略回测系统（代替ETF）
策略：MACD柱上穿0轴买入，跌破0轴卖出
"""

import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import tushare as ts
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
pro = ts.pro_api(TUSHARE_TOKEN)


# 主要指数列表（代替ETF）
INDEX_LIST = [
    # 宽基指数
    ("000300.SH", "沪深300"),
    ("000905.SH", "中证500"),
    ("399006.SZ", "创业板指"),
    ("000016.SH", "上证50"),
    ("000852.SH", "中证1000"),
    ("399673.SZ", "创业板50"),
    
    # 行业指数
    ("399971.SZ", "中证传媒"),
    ("399932.SZ", "中证医药"),
    ("399986.SZ", "中证银行"),
    ("399970.SZ", "中证军工"),
    ("399975.SZ", "中证证券"),
    ("399998.SZ", "中证煤炭"),
    ("399936.SZ", "中证电子"),
    ("399997.SZ", "中证白酒"),
    ("399941.SZ", "中证新能源"),
    ("399806.SZ", "中证环保"),
    ("399939.SZ", "中证信息"),
    ("399959.SZ", "中证军工"),
]


def calculate_macd(df, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    df = df.copy()
    
    if 'Close' not in df.columns:
        df['Close'] = df['close']
    
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    
    df['DIF'] = ema_fast - ema_slow
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    
    df['MACD_cross_up'] = (df['MACD'] >= 0) & (df['MACD'].shift(1) < 0)
    df['MACD_cross_down'] = (df['MACD'] < 0) & (df['MACD'].shift(1) >= 0)
    
    return df


def get_index_data(ts_code, start_date='20210101', end_date='20260516'):
    """获取指数数据"""
    try:
        df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        if df is None or len(df) < 60:
            return None
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)
        
        return df
    except Exception as e:
        return None


def backtest_macd_strategy(df, initial_cash=1000000, position_size=0.95):
    """MACD策略回测"""
    df = calculate_macd(df)
    
    cash = initial_cash
    position = 0
    entry_price = 0
    entry_date = None
    trades = []
    equity_curve = []
    
    for date, row in df.iterrows():
        if pd.isna(row['Close']) or pd.isna(row['MACD']):
            continue
        
        current_price = row['Close']
        
        # 买入信号
        if position == 0 and row['MACD_cross_up']:
            shares = int(cash * position_size / current_price)
            if shares > 0:
                position = shares
                entry_price = current_price
                entry_date = date
                cash -= shares * current_price
                trades.append({
                    'date': date,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'reason': 'MACD上穿0轴'
                })
        
        # 卖出信号
        elif position > 0 and row['MACD_cross_down']:
            cash += position * current_price
            pnl = (current_price - entry_price) * position
            pnl_pct = (current_price - entry_price) / entry_price * 100
            trades.append({
                'date': date,
                'type': 'SELL',
                'price': current_price,
                'shares': position,
                'reason': 'MACD下穿0轴',
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
            position = 0
            entry_price = 0
            entry_date = None
        
        equity = cash + position * current_price
        equity_curve.append({'date': date, 'equity': equity})
    
    # 期末清仓
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
            'reason': '期末清仓',
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


def run_index_backtest():
    """运行指数MACD策略回测"""
    print("=" * 100)
    print("📊 指数MACD策略回测系统（代替ETF）")
    print("策略：MACD柱上穿0轴买入，跌破0轴卖出")
    print("=" * 100)
    
    results = []
    
    for ts_code, name in INDEX_LIST:
        print(f"\n正在回测: {name} ({ts_code})...", end=" ")
        
        df = get_index_data(ts_code, '20210101', '20260516')
        
        if df is None:
            print("❌ 数据不足")
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
        
        print(f"✅ 收益: {result['return_pct']:+.2f}%, 交易次数: {result['total_trades']}次, 胜率: {result['win_rate']:.1f}%")
    
    if len(results) == 0:
        print("\n❌ 没有获取到任何数据")
        return
    
    # 按收益率排序
    results.sort(key=lambda x: x['return_pct'], reverse=True)
    
    print("\n" + "=" * 100)
    print("📊 指数MACD策略回测结果汇总（按收益率排序）")
    print("=" * 100)
    print(f"{'排名':>4} {'指数名称':<16} {'代码':<12} {'期初点位':>10} {'期末点位':>10} {'MACD策略':>12} {'交易次数':>8} {'胜率':>8}")
    print("-" * 100)
    
    for i, r in enumerate(results, 1):
        print(f"{i:>4}. {r['name']:<16} {r['code']:<12} "
              f"{r['start_price']:>10.2f} {r['end_price']:>10.2f} "
              f"{r['return_pct']:>+11.2f}% {r['total_trades']:>8}次 {r['win_rate']:>7.1f}%")
    
    # 统计
    total_return = [r['return_pct'] for r in results]
    avg_return = np.mean(total_return)
    win_count = len([r for r in results if r['return_pct'] > 0])
    total_count = len(results)
    
    print("\n" + "=" * 100)
    print("📈 总体统计")
    print("=" * 100)
    print(f"指数总数: {total_count}")
    print(f"正收益指数: {win_count} ({win_count/total_count*100:.1f}%)")
    print(f"负收益指数: {total_count - win_count} ({(total_count-win_count)/total_count*100:.1f}%)")
    print(f"平均收益率: {avg_return:+.2f}%")
    print(f"最高收益率: {max(total_return):+.2f}%")
    print(f"最低收益率: {min(total_return):+.2f}%")
    
    # 保存结果
    save_results(results, avg_return, win_count, total_count)
    
    return results


def save_results(results, avg_return, win_count, total_count):
    """保存结果"""
    output_dir = Path(__file__).parent / "double_line_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    data = {
        'strategy': 'MACD策略',
        'description': 'MACD柱上穿0轴买入，跌破0轴卖出',
        'backtest_period': '2021-01-01 to 2026-05-16',
        'timestamp': timestamp,
        'summary': {
            'total_indexes': total_count,
            'winning_indexes': win_count,
            'win_rate_indexes': win_count / total_count * 100,
            'average_return': avg_return
        },
        'results': results
    }
    
    report_file = output_dir / f"index_macd_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 回测结果已保存: {report_file}")


if __name__ == "__main__":
    run_index_backtest()
