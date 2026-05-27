"""
MACD策略回测
策略逻辑：
- MACD值从负变正（金叉）时买入
- 白色线(MACD)在紫色线(Signal)上方
- MACD值从正变负（死叉）时清仓
"""
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from data.tushare_data import get_stock_daily


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    计算MACD指标
    
    MACD = EMA(close, fast) - EMA(close, slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    # 计算EMA
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    
    # MACD线
    df['macd'] = ema_fast - ema_slow
    
    # Signal线
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    
    # Histogram
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 金叉死叉信号
    # MACD从负变正 = 金叉（买入信号）
    # MACD从正变负 = 死叉（卖出信号）
    df['macd_above_zero'] = df['macd'] > 0
    df['macd_cross_up'] = (df['macd'] > 0) & (df['macd'].shift(1) <= 0)
    df['macd_cross_down'] = (df['macd'] < 0) & (df['macd'].shift(1) >= 0)
    
    return df


def backtest_macd_strategy(df, initial_cash=1000000):
    """
    MACD策略回测
    
    买入条件：MACD从负变正（金叉）且MACD在Signal上方
    卖出条件：MACD从正变负（死叉）
    """
    cash = initial_cash
    position = 0
    entry_price = 0
    trades = []
    portfolio_values = []
    
    for i, row in df.iterrows():
        price = row['Close']
        
        if pd.isna(price):
            portfolio_values.append(cash)
            continue
        
        # 金叉买入（MACD从负变正）且白色线在紫色线上方
        if row['macd_cross_up'] and row['macd'] > row['macd_signal'] and position == 0:
            shares = int(cash / price)
            if shares > 0:
                position = shares
                entry_price = price
                cash -= shares * price
                trades.append({
                    'date': i,
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'macd': row['macd'],
                    'cash_used': shares * price
                })
        
        # 死叉卖出（MACD从正变负）
        elif row['macd_cross_down'] and position > 0:
            cash += position * price
            profit = (price - entry_price) * position
            profit_pct = (price - entry_price) / entry_price * 100
            trades.append({
                'date': i,
                'type': 'SELL',
                'price': price,
                'shares': position,
                'macd': row['macd'],
                'cash_gained': position * price,
                'profit': profit,
                'profit_pct': profit_pct
            })
            position = 0
            entry_price = 0
        
        # 更新组合价值
        portfolio_values.append(cash + position * price)
    
    # 最后清仓
    if position > 0:
        final_price = df.iloc[-1]['Close']
        cash += position * final_price
        profit = (final_price - entry_price) * position
        profit_pct = (final_price - entry_price) / entry_price * 100
        trades.append({
            'date': df.index[-1],
            'type': 'FINAL_SELL',
            'price': final_price,
            'shares': position,
            'macd': df.iloc[-1]['macd'],
            'cash_gained': position * final_price,
            'profit': profit,
            'profit_pct': profit_pct
        })
    
    return {
        'final_cash': cash,
        'return_pct': (cash - initial_cash) / initial_cash * 100,
        'trades': trades,
        'portfolio_values': portfolio_values
    }


def analyze_macd_trades(trades):
    """分析MACD交易"""
    sell_trades = [t for t in trades if t['type'] in ('SELL', 'FINAL_SELL')]
    
    if not sell_trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_profit_pct': 0
        }
    
    winning = len([t for t in sell_trades if t.get('profit', 0) > 0])
    losing = len([t for t in sell_trades if t.get('profit', 0) < 0])
    avg_profit = np.mean([t.get('profit_pct', 0) for t in sell_trades])
    
    return {
        'total_trades': len(sell_trades),
        'winning_trades': winning,
        'losing_trades': losing,
        'win_rate': winning / len(sell_trades) * 100,
        'avg_profit_pct': avg_profit
    }


def run_macd_backtest():
    """运行MACD策略回测"""
    
    stocks = [
        ('600570.SS', '恒生电子'),
        ('600519.SS', '贵州茅台'),
        ('000001.SZ', '平安银行'),
        ('300750.SZ', '宁德时代'),
        ('601318.SS', '中国平安'),
        ('000333.SZ', '美的集团'),
        ('002415.SZ', '海康威视'),
        ('300059.SZ', '东方财富'),
    ]
    
    print("=" * 100)
    print("📊 MACD策略回测")
    print("   策略逻辑：MACD从负变正买入 + 白色线在紫色线上方，从正变负清仓")
    print("   回测区间：2022-01-01 ~ 2024-12-31")
    print("=" * 100)
    
    results = []
    
    for ticker, name in stocks:
        print(f"\n[{ticker}] {name}...")
        
        df = get_stock_daily(ticker, '20220101', '20241231')
        if df is None or df.empty:
            print(f"  ❌ 数据获取失败")
            continue
        
        df = calculate_macd(df)
        result = backtest_macd_strategy(df)
        analysis = analyze_macd_trades(result['trades'])
        
        # 买入持有基准
        bh_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
        
        results.append({
            'ticker': ticker,
            'name': name,
            'macd_return': result['return_pct'],
            'buy_hold_return': bh_return,
            'trade_count': analysis['total_trades'],
            'win_rate': analysis['win_rate'],
            'avg_profit': analysis['avg_profit_pct'],
            'improvement': result['return_pct'] - bh_return
        })
        
        print(f"  MACD策略: {result['return_pct']:+.2f}% (交易{analysis['total_trades']}次, 胜率{analysis['win_rate']:.0f}%)")
        print(f"  买入持有: {bh_return:+.2f}%")
        print(f"  相对收益: {result['return_pct'] - bh_return:+.2f}%")
    
    # 汇总
    print("\n" + "=" * 100)
    print("📋 MACD策略汇总")
    print("=" * 100)
    print(f"{'股票':<12} | {'MACD策略':>10} | {'买入持有':>10} | {'交易次数':>8} | {'胜率':>6} | {'平均收益':>8} | {'改善':>10}")
    print("-" * 90)
    
    for r in results:
        imp_str = f"{r['improvement']:+.2f}%"
        if r['improvement'] > 0:
            imp_str = f"✅ {r['improvement']:+.2f}%"
        
        print(f"{r['name']:<12} | {r['macd_return']:>+10.2f}% | {r['buy_hold_return']:>+10.2f}% | {r['trade_count']:>8} | {r['win_rate']:>5.0f}% | {r['avg_profit']:>7.2f}% | {imp_str:>10}")
    
    # 平均值
    avg_macd = np.mean([r['macd_return'] for r in results])
    avg_bh = np.mean([r['buy_hold_return'] for r in results])
    avg_improvement = np.mean([r['improvement'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results])
    
    print("-" * 90)
    print(f"{'平均':<12} | {avg_macd:>+10.2f}% | {avg_bh:>+10.2f}% | {'':>8} | {avg_win_rate:>5.0f}% | {'':>8} | {avg_improvement:>+10.2f}%")
    
    # 统计
    better_count = len([r for r in results if r['improvement'] > 0])
    print(f"\nMACD策略胜出: {better_count}/{len(results)} 只股票")
    
    print("\n" + "=" * 100)
    if avg_improvement > 0:
        print(f"✅ MACD策略平均跑赢买入持有 {avg_improvement:+.2f}%")
    else:
        print(f"❌ MACD策略平均跑输买入持有 {abs(avg_improvement):.2f}%")
    print("=" * 100)
    
    # 保存结果
    save_results(results)
    
    return results


def save_results(results):
    """保存结果"""
    output_dir = Path(__file__).parent.parent / "results" / "double_line_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    data = {
        'strategy': 'MACD',
        'description': 'MACD从负变正买入 + 白色线在紫色线上方，从正变负清仓',
        'timestamp': timestamp,
        'results': results,
        'summary': {
            'avg_macd_return': float(np.mean([r['macd_return'] for r in results])),
            'avg_buy_hold_return': float(np.mean([r['buy_hold_return'] for r in results])),
            'avg_improvement': float(np.mean([r['improvement'] for r in results])),
            'avg_win_rate': float(np.mean([r['win_rate'] for r in results])),
            'win_count': len([r for r in results if r['improvement'] > 0])
        }
    }
    
    report_file = output_dir / f"macd_strategy_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 报告已保存: {report_file}")


if __name__ == "__main__":
    run_macd_backtest()
