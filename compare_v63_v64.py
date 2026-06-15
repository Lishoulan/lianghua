"""V6.3 vs V6.4 快速对比回测（使用缓存数据）"""
import sys, os, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classic_ta.stock_data_cache import get_stock_data_cached
from classic_ta.v60_ambush_model import IndicatorCalcBase
from classic_ta.v63_ambush_model import (
    add_micro_confirm_indicators, Detect_AmbushSignal_V63,
    StatefulTradeBacktester_V63, V63_PARAMS,
)
from classic_ta.v64_ambush_model import (
    add_inst_support_indicators, Detect_AmbushSignal_V64,
    StatefulTradeBacktester_V64, analyze_support_score_impact,
    V64_PARAMS,
)
from classic_ta.v61_ambush_model import TradeRecord

# 选20只代表性股票
TEST_STOCKS = [
    ("600036.SH", "招商银行"), ("601318.SH", "中国平安"), ("000001.SZ", "平安银行"),
    ("600030.SH", "中信证券"), ("300059.SZ", "东方财富"),
    ("600519.SH", "贵州茅台"), ("000858.SZ", "五粮液"), ("000651.SZ", "格力电器"),
    ("000333.SZ", "美的集团"), ("600887.SH", "伊利股份"),
    ("600276.SH", "恒瑞医药"), ("300760.SZ", "迈瑞医疗"), ("000538.SZ", "云南白药"),
    ("300750.SZ", "宁德时代"), ("002594.SZ", "比亚迪"),
    ("002415.SZ", "海康威视"), ("002230.SZ", "科大讯飞"),
    ("600031.SH", "三一重工"), ("601899.SH", "紫金矿业"), ("600900.SH", "长江电力"),
]


def run_comparison():
    v63_all_trades = []
    v64_all_trades = []
    errors = 0

    for i, (code, name) in enumerate(TEST_STOCKS, 1):
        df = get_stock_data_cached(code, min_rows=130)
        if df is None:
            errors += 1
            print(f"  [{i}] {name} 无数据")
            continue

        # V6.3
        try:
            df3 = IndicatorCalcBase(df.copy())
            df3 = add_micro_confirm_indicators(df3)
            df3 = Detect_AmbushSignal_V63(df3, V63_PARAMS)
            t63 = StatefulTradeBacktester_V63(df3, params=V63_PARAMS, ts_code=code)
        except Exception as e:
            print(f"  [{i}] {name} V6.3异常: {e}")
            t63 = []

        # V6.4
        try:
            df4 = IndicatorCalcBase(df.copy())
            df4 = add_micro_confirm_indicators(df4)
            df4 = add_inst_support_indicators(df4, V64_PARAMS)
            df4 = Detect_AmbushSignal_V64(df4, V64_PARAMS)
            t64 = StatefulTradeBacktester_V64(df4, params=V64_PARAMS, ts_code=code)
        except Exception as e:
            print(f"  [{i}] {name} V6.4异常: {e}")
            t64 = []

        # 信号数
        sig63 = int(df3["ambush_signal"].sum()) if "ambush_signal" in df3.columns else 0
        sig64 = int(df4["ambush_signal"].sum()) if "ambush_signal" in df4.columns else 0

        v63_all_trades.extend(t63)
        v64_all_trades.extend(t64)

        wr63 = f"{len([t for t in t63 if t.profit_pct > 0])}/{len(t63)}" if t63 else "0/0"
        wr64 = f"{len([t for t in t64 if t.profit_pct > 0])}/{len(t64)}" if t64 else "0/0"
        print(f"  [{i:2d}] {name:<8} 信号 {sig63:>3}→{sig64:>3} | "
              f"V6.3交易 {wr63:>7} | V6.4交易 {wr64:>7}")

    # ── 汇总对比 ──
    print("\n" + "=" * 80)
    print("  V6.3 vs V6.4 对比汇总（缓存数据 2024~2026）")
    print("=" * 80)

    for label, trades in [("V6.3", v63_all_trades), ("V6.4", v64_all_trades)]:
        if not trades:
            print(f"  {label}: 无交易")
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

        exit_counts = {}
        for t in trades:
            exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

        print(f"\n  【{label}】")
        print(f"    总交易: {total}笔 | 盈利: {len(wins)} | 亏损: {len(losses)}")
        print(f"    胜率: {wr:.1f}%")
        print(f"    平均收益: {avg_p:+.2f}% | 平均盈利: {avg_w:+.2f}% | 平均亏损: {avg_l:+.2f}%")
        print(f"    盈亏比: {pf:.2f}")
        print(f"    平均持仓: {avg_hold:.1f}天")
        print(f"    退出分布: {exit_counts}")

    # V6.4 评分分析
    if v64_all_trades:
        print("\n  【V6.4 主力托底评分分析】")
        score_analysis = analyze_support_score_impact(v64_all_trades)
        for score_key, stats in score_analysis.items():
            sv = score_key.replace("score_", "")
            print(f"    评分={sv}: 交易{stats['trades']}笔 | "
                  f"胜率{stats['win_rate']:.1f}% | 平均收益{stats['avg_profit']:+.2f}%")

    # 差异
    if v63_all_trades and v64_all_trades:
        wr63 = len([t for t in v63_all_trades if t.profit_pct > 0]) / len(v63_all_trades) * 100
        wr64 = len([t for t in v64_all_trades if t.profit_pct > 0]) / len(v64_all_trades) * 100
        ap63 = np.mean([t.profit_pct for t in v63_all_trades])
        ap64 = np.mean([t.profit_pct for t in v64_all_trades])
        print(f"\n  【差异】")
        print(f"    胜率变化: {wr63:.1f}% → {wr64:.1f}% ({wr64 - wr63:+.1f}pp)")
        print(f"    平均收益变化: {ap63:+.2f}% → {ap64:+.2f}% ({ap64 - ap63:+.2f}pp)")
        print(f"    信号过滤: {len(v63_all_trades)}笔 → {len(v64_all_trades)}笔 "
              f"(过滤{len(v63_all_trades) - len(v64_all_trades)}笔)")


if __name__ == "__main__":
    print("V6.3 vs V6.4 快速对比回测")
    print(f"股票池: {len(TEST_STOCKS)}只 | 数据源: 本地缓存")
    print("=" * 80)
    run_comparison()
