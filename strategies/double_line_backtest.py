"""
双线战法回测系统主程序
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from strategies.double_line_strategy import calculate_double_line_strategy, backtest_strategy
from data.tushare_data import get_stock_daily


def run_backtest_for_stock(ticker, ticker_name, start_date='20220101', end_date='20241231'):
    """
    运行单只股票的回测
    """
    print(f"\n{'='*70}")
    print(f"📊 开始回测: {ticker_name} ({ticker})")
    print(f"{'='*70}")
    
    # 1. 获取数据
    print("\n[1/4] 获取历史数据...")
    df = get_stock_daily(ticker, start_date, end_date)
    
    if df is None or df.empty:
        print(f"❌ 无法获取 {ticker_name} 的数据")
        return None
    
    print(f"✅ 获取到 {len(df)} 条数据")
    print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
    
    # 2. 计算指标
    print("\n[2/4] 计算双线战法指标...")
    df = calculate_double_line_strategy(df)
    print(f"✅ 指标计算完成")
    
    # 3. 回测所有玩法
    print("\n[3/4] 运行5种玩法回测...")
    
    results = {}
    play_names = {
        'play1_signal': '玩法1: 金叉买入，死叉卖出（无脑玩法）',
        'play2_signal': '玩法2: 死叉时强制离场',
        'play3_signal': '玩法3: 死叉多反而是极限买点',
        'play4_signal': '玩法4: 白黄区间都是容错率高的买入区',
        'play5_signal': '玩法5: 放量金叉后缩量回踩黄线',
    }
    
    for signal_col, play_name in play_names.items():
        print(f"\n运行 {play_name}...")
        df_result, trades, final_cash = backtest_strategy(df.copy(), signal_col)
        results[signal_col] = {
            'name': play_name,
            'trades': trades,
            'final_cash': final_cash,
            'total_return': (final_cash - 1000000) / 1000000 * 100,
            'trade_count': len(trades)
        }
        print(f"✅ 完成，最终资金: {final_cash:,.0f} 元，收益率: {results[signal_col]['total_return']:+.2f}%")
    
    # 4. 保存结果
    print("\n[4/4] 保存回测结果...")
    save_results(ticker, ticker_name, df, results)
    
    return {
        'ticker': ticker,
        'name': ticker_name,
        'results': results,
        'data': df
    }


def save_results(ticker, ticker_name, df, results):
    """
    保存回测结果
    """
    output_dir = Path(__file__).parent.parent / "results" / "double_line_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = ticker.replace('.', '_')
    
    # 1. 保存完整数据
    data_file = output_dir / f"{safe_ticker}_data_{timestamp}.csv"
    df.to_csv(data_file, encoding='utf-8-sig')
    print(f"📊 数据保存到: {data_file}")
    
    # 2. 保存回测结果
    summary = {
        'ticker': ticker,
        'name': ticker_name,
        'analysis_date': datetime.now().isoformat(),
        'strategies': {}
    }
    
    for signal_col, result in results.items():
        summary['strategies'][signal_col] = {
            'name': result['name'],
            'final_cash': result['final_cash'],
            'total_return_pct': result['total_return'],
            'trade_count': result['trade_count'],
            'trades': result['trades']
        }
    
    summary_file = output_dir / f"{safe_ticker}_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📋 回测总结保存到: {summary_file}")


def print_summary(result):
    """
    打印回测总结
    """
    if result is None:
        return
    
    print(f"\n{'='*70}")
    print(f"📋 {result['name']} ({result['ticker']}) 回测总结")
    print(f"{'='*70}")
    
    for signal_col, res in result['results'].items():
        print(f"\n{res['name']}")
        print(f"  最终资金: ¥{res['final_cash']:,.0f}")
        print(f"  总收益率: {res['total_return']:+.2f}%")
        print(f"  交易次数: {res['trade_count']}")


def main():
    """
    主程序
    """
    # 要回测的股票列表
    stocks = [
        ('600570.SS', '恒生电子'),
        ('600519.SS', '贵州茅台'),
        ('000001.SZ', '平安银行'),
        ('300750.SZ', '宁德时代'),
    ]
    
    print("="*70)
    print("🚀 双线战法回测系统")
    print("="*70)
    print(f"回测股票数: {len(stocks)}")
    print(f"初始资金: ¥1,000,000")
    print(f"策略数: 5种玩法")
    
    all_results = []
    for ticker, name in stocks:
        try:
            result = run_backtest_for_stock(
                ticker, name,
                start_date='20220101',
                end_date='20241231'
            )
            if result:
                all_results.append(result)
                print_summary(result)
        except Exception as e:
            print(f"❌ {name} 回测失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("✅ 回测完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
