
import json
from pathlib import Path

def main():
    results_dir = Path(__file__).parent.parent / "results" / "double_line_backtest_results"
    
    # Load results
    v1_path = results_dir / "brick_strategy_summary_20260520_102419.json"
    v2_path = results_dir / "brick_strategy_summary_20260520_152426.json"
    
    with open(v1_path, 'r', encoding='utf-8') as f:
        v1 = json.load(f)
    
    with open(v2_path, 'r', encoding='utf-8') as f:
        v2 = json.load(f)
    
    # Generate comparison table
    print("=" * 100)
    print("砖型图选股策略 v1 vs v2 对比报告")
    print("=" * 100)
    print()
    
    print("指标                v1                   v2                   变化")
    print("-" * 80)
    sig_diff = v2['total_signals'] - v1['total_signals']
    sig_pct = (v2['total_signals']/v1['total_signals']-1)*100
    print("信号总数            %-20d %-20d %-20d (%.1f%%)" % (
        v1['total_signals'], 
        v2['total_signals'], 
        sig_diff, 
        sig_pct
    ))
    stock_diff = v2['unique_stocks'] - v1['unique_stocks']
    print("覆盖股票数          %-20d %-20d %-20d" % (
        v1['unique_stocks'], 
        v2['unique_stocks'], 
        stock_diff
    ))
    print()
    
    print("=" * 100)
    print("前瞻收益对比")
    print("=" * 100)
    print()
    
    for period in ['1d', '3d', '5d', '10d', '20d']:
        v1_ret = v1['forward_returns'][period]
        v2_ret = v2['forward_returns'][period]
        print("%s 周期:" % period.upper())
        print("  胜率:           %.2f%%              %.2f%%              %+.2f%%" % (
            v1_ret['win_rate'], 
            v2_ret['win_rate'], 
            v2_ret['win_rate']-v1_ret['win_rate']
        ))
        print("  平均收益:       %.2f%%              %.2f%%              %+.2f%%" % (
            v1_ret['avg_return'], 
            v2_ret['avg_return'], 
            v2_ret['avg_return']-v1_ret['avg_return']
        ))
        print("  最大收益:       %.2f%%              %.2f%%              %+.2f%%" % (
            v1_ret['max_return'], 
            v2_ret['max_return'], 
            v2_ret['max_return']-v1_ret['max_return']
        ))
        print("  最大亏损:       %.2f%%              %.2f%%              %+.2f%%" % (
            v1_ret['min_return'], 
            v2_ret['min_return'], 
            v2_ret['min_return']-v1_ret['min_return']
        ))
        print("  盈亏比:         %.2f               %.2f               %+.2f" % (
            v1_ret['profit_loss_ratio'], 
            v2_ret['profit_loss_ratio'], 
            v2_ret['profit_loss_ratio']-v1_ret['profit_loss_ratio']
        ))
        print()
    
    print("=" * 100)
    print("策略条件对比")
    print("=" * 100)
    print()
    print("v1 条件:")
    for i, cond in enumerate(v1['conditions'], 1):
        print("  %d. %s" % (i, cond))
    print()
    print("v2 条件 (新增止损防守线):")
    for i, cond in enumerate(v2['conditions'], 1):
        print("  %d. %s" % (i, cond))
    print()
    
    print("=" * 100)
    print("总结与建议")
    print("=" * 100)
    print()
    print("v2 策略通过新增止损防守线，显著减少了信号数量（-63%），")
    print("同时大幅收窄了最大亏损（20日从 -63.88% 降至 -52.13%）。")
    print()
    print("建议：")
    print("  1. 如果追求稳健，可使用 v2 策略")
    print("  2. 如果追求更多机会，可微调止损系数至 0.97 或 0.99")
    print("  3. 可进一步测试不同的距离止损线阈值（3%、7%）")
    print()

if __name__ == "__main__":
    main()

