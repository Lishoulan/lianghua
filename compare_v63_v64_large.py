"""V6.3 vs V6.4 大规模对比回测（随机抽样200只缓存股票）"""
import sys, os, time, random, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classic_ta.stock_data_cache import get_stock_data_cached
from classic_ta.v60_ambush_model import IndicatorCalcBase
from classic_ta.v63_ambush_model import (
    add_micro_confirm_indicators, Detect_AmbushSignal_V63,
    StatefulTradeBacktester_V63, V63_PARAMS,
)
from classic_ta.v64_ambush_model import (
    add_inst_support_indicators, add_entry_quality_indicators,
    Detect_AmbushSignal_V64,
    StatefulTradeBacktester_V64, analyze_support_score_impact,
    V64_PARAMS,
)

def get_cached_stock_codes():
    cache_dir = Path("results/stock_cache")
    codes = []
    for f in os.listdir(cache_dir):
        if f.endswith(".csv"):
            code = f.replace(".csv", "")
            # 排除北交所
            if code.startswith("8") or code.startswith("9"):
                continue
            codes.append(code)
    return codes

def run_large_comparison(sample_size=200):
    all_codes = get_cached_stock_codes()
    random.seed(42)
    selected = random.sample(all_codes, min(sample_size, len(all_codes)))
    print(f"从 {len(all_codes)} 只缓存股票中随机抽取 {len(selected)} 只")
    print("=" * 80)

    v63_trades = []
    v64_trades = []
    loaded = 0
    errors = 0
    t0 = time.time()

    for i, code in enumerate(selected, 1):
        df = get_stock_data_cached(code, min_rows=130)
        if df is None:
            errors += 1
            continue
        loaded += 1

        # V6.3
        try:
            df3 = IndicatorCalcBase(df.copy())
            df3 = add_micro_confirm_indicators(df3)
            df3 = Detect_AmbushSignal_V63(df3, V63_PARAMS)
            t63 = StatefulTradeBacktester_V63(df3, params=V63_PARAMS, ts_code=code)
        except:
            t63 = []

        # V6.4.8
        try:
            df4 = IndicatorCalcBase(df.copy())
            df4 = add_micro_confirm_indicators(df4)
            df4 = add_entry_quality_indicators(df4, V64_PARAMS)
            df4 = Detect_AmbushSignal_V64(df4, V64_PARAMS)
            t64 = StatefulTradeBacktester_V64(df4, params=V64_PARAMS, ts_code=code)
        except:
            t64 = []

        v63_trades.extend(t63)
        v64_trades.extend(t64)

        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  进度: {i}/{len(selected)} | 已加载{loaded} | "
                  f"V6.3:{len(v63_trades)}笔 V6.4:{len(v64_trades)}笔 | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\n完成! 耗时{elapsed:.0f}s | 加载{loaded}只 | 错误{errors}")

    # ── 汇总 ──
    print("\n" + "=" * 80)
    print(f"  V6.3 vs V6.4.9(+趋势方向) 对比汇总（{loaded}只股票, 2024~2026缓存数据）")
    print("=" * 80)

    for label, trades in [("V6.3", v63_trades), ("V6.4", v64_trades)]:
        if not trades:
            print(f"\n  【{label}】无交易")
            continue
        total = len(trades)
        wins = [t for t in trades if t.profit_pct > 0]
        losses = [t for t in trades if t.profit_pct <= 0]
        wr = len(wins) / total * 100
        avg_p = np.mean([t.profit_pct for t in trades])
        avg_w = np.mean([t.profit_pct for t in wins]) if wins else 0
        avg_l = np.mean([t.profit_pct for t in losses]) if losses else 0
        pf = abs(avg_w / avg_l) if avg_l != 0 else 999
        avg_hold = np.mean([t.hold_days for t in trades])
        total_p = sum(t.profit_pct for t in trades)

        exit_counts = {}
        for t in trades:
            exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

        print(f"\n  【{label}】")
        print(f"    总交易: {total}笔 | 盈利: {len(wins)} | 亏损: {len(losses)}")
        print(f"    胜率: {wr:.1f}%")
        print(f"    平均收益: {avg_p:+.2f}% | 累计收益: {total_p:+.1f}%")
        print(f"    平均盈利: {avg_w:+.2f}% | 平均亏损: {avg_l:+.2f}%")
        print(f"    盈亏比: {pf:.2f}")
        print(f"    平均持仓: {avg_hold:.1f}天")
        print(f"    退出分布: {exit_counts}")

    # V6.4 评分分析
    if v64_trades:
        print("\n  【V6.4.9 入场质量评分+趋势方向 → 胜率分析】")
        print("  （评分0~8 + 黄线上升二级过滤）")
        score_analysis = analyze_support_score_impact(v64_trades)
        for score_key, stats in score_analysis.items():
            sv = score_key.replace("score_", "")
            bar = "█" * max(1, int(stats["win_rate"] / 5))
            print(f"    评分={sv}/8: {stats['trades']:>3}笔 | "
                  f"胜率{stats['win_rate']:>5.1f}% | 均收益{stats['avg_profit']:>+6.2f}% | {bar}")
            # 显示因子组合模式
            if stats.get("factor_pattern"):
                for pattern, count in stats["factor_pattern"].items():
                    pw = [t for t in v64_trades if t.stock_name == pattern and t.profit_pct > 0]
                    pt = [t for t in v64_trades if t.stock_name == pattern]
                    pwr = len(pw) / len(pt) * 100 if pt else 0
                    pap = np.mean([t.profit_pct for t in pt]) if pt else 0
                    print(f"      └ {pattern}: {count}笔 胜率{pwr:.0f}% 均收益{pap:+.2f}%")

    # 差异
    if v63_trades and v64_trades:
        wr63 = len([t for t in v63_trades if t.profit_pct > 0]) / len(v63_trades) * 100
        wr64 = len([t for t in v64_trades if t.profit_pct > 0]) / len(v64_trades) * 100
        ap63 = np.mean([t.profit_pct for t in v63_trades])
        ap64 = np.mean([t.profit_pct for t in v64_trades])
        tp63 = sum(t.profit_pct for t in v63_trades)
        tp64 = sum(t.profit_pct for t in v64_trades)
        filtered = len(v63_trades) - len(v64_trades)
        # 被过滤的交易质量
        if filtered > 0:
            # 找出V6.3有但V6.4没有的交易（通过日期+代码匹配）
            v64_keys = set((t.ts_code, t.buy_date) for t in v64_trades)
            filtered_trades = [t for t in v63_trades if (t.ts_code, t.buy_date) not in v64_keys]
            if filtered_trades:
                fw = len([t for t in filtered_trades if t.profit_pct > 0])
                fwr = fw / len(filtered_trades) * 100
                fap = np.mean([t.profit_pct for t in filtered_trades])
                print(f"\n  【被V6.4.9过滤掉的交易分析】")
                print(f"    共{len(filtered_trades)}笔 | 胜率{fwr:.1f}% | 平均收益{fap:+.2f}%")
                print(f"    → 如果这些交易质量差，说明过滤有效")

        print(f"\n  【最终差异】")
        print(f"    胜率: {wr63:.1f}% → {wr64:.1f}% ({wr64 - wr63:+.1f}pp)")
        print(f"    平均收益: {ap63:+.2f}% → {ap64:+.2f}% ({ap64 - ap63:+.2f}pp)")
        print(f"    累计收益: {tp63:+.1f}% → {tp64:+.1f}% ({tp64 - tp63:+.1f}pp)")
        print(f"    交易数: {len(v63_trades)} → {len(v64_trades)} (过滤{filtered}笔)")


if __name__ == "__main__":
    run_large_comparison(200)
