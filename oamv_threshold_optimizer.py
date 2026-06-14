"""
OAMV活跃市值择时参数优化器
============================
目标：找到活跃市值X指标的最优做多/做空阈值，使得：
  - 做多（state=1）期间持有指数的累计收益最大化
  - 做空（state=0）期间避开下跌的规避收益最大化
  - 综合评估：BULL期间年化收益 vs BEAR期间年化收益

数据源：
  - 全市场活跃市值：results/oamv_cache/market_amv_cache.csv
    公式：AMV = Σ(每只股票 circ_mv × turnover_rate_f / 100)
  - 上证指数日线：tushare index_daily（用于评估择时效果）

优化搜索空间：
  - smooth_period: [5, 8, 10, 13, 15, 20]（AMV平滑周期）
  - cost_ma_period: [13, 17, 21, 26, 34, 42]（成本均线周期）
  - upper_threshold: [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]（做多阈值）
  - lower_threshold: [-1.0, -1.5, -2.0, -2.3, -2.5, -3.0, -3.5, -4.0]（做空阈值）

评估指标：
  1. 做多期间年化收益率
  2. 做空期间年化收益率（负值越大说明成功避开下跌）
  3. 收益差（做多 - 做空）
  4. 做多胜率（做多期间正收益天数占比）
  5. 最大回撤（做多期间）
  6. 信号切换次数（太频繁不好操作）

用法：
    python oamv_threshold_optimizer.py
"""

import os
import sys
import time
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
import tushare as ts

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
pro = ts.pro_api(TUSHARE_TOKEN) if TUSHARE_TOKEN else None


# ══════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════

def load_amv_cache():
    """加载全市场活跃市值缓存"""
    cache_file = Path(__file__).parent / "results" / "oamv_cache" / "market_amv_cache.csv"
    if not cache_file.exists():
        print("❌ 活跃市值缓存文件不存在，请先运行 market_amv_cache.py")
        return None
    df = pd.read_csv(cache_file, index_col='trade_date', parse_dates=True)
    # 合并列
    if 'amv_circ' in df.columns and 'amv' not in df.columns:
        df['amv'] = df['amv_circ']
    elif 'amv_circ' in df.columns:
        df['amv'] = df['amv'].fillna(df['amv_circ'])
    amv_series = df['amv'].dropna()
    print(f"✅ 活跃市值缓存: {len(amv_series)}天 ({amv_series.index[0].strftime('%Y-%m-%d')} ~ {amv_series.index[-1].strftime('%Y-%m-%d')})")
    return amv_series


def load_index_daily(start_date='20191201', end_date=None):
    """加载上证指数日线数据"""
    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y%m%d')
    try:
        df = pro.index_daily(ts_code='000001.SH', start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            print("❌ 无法获取上证指数数据")
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)
        df['Close'] = df['close'].astype(float)
        df['daily_return'] = df['Close'].pct_change()
        print(f"✅ 上证指数: {len(df)}天 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
        return df
    except Exception as e:
        print(f"❌ 获取上证指数失败: {e}")
        return None


# ══════════════════════════════════════════════════════════
#  OAMV计算（向量化，支持参数化）
# ══════════════════════════════════════════════════════════

def compute_oamv_x(amv_series, smooth_method='sma', smooth_period=8,
                    cost_ma_method='sma', cost_ma_period=21):
    """
    计算OAMV X指标值
    X = (平滑AMV - 成本均线) / 成本均线 * 100%

    参数:
        amv_series: 全市场活跃市值时间序列
        smooth_method: 平滑方式 ('sma', 'ema', 'hybrid')
        smooth_period: 平滑周期
        cost_ma_method: 成本均线方式 ('sma', 'ema')
        cost_ma_period: 成本均线周期

    返回:
        x_series: X指标值序列（百分比）
    """
    # 平滑
    if smooth_method == 'sma':
        smoothed = amv_series.rolling(window=smooth_period, min_periods=smooth_period).mean()
    elif smooth_method == 'ema':
        smoothed = amv_series.ewm(span=smooth_period, adjust=False).mean()
    elif smooth_method == 'hybrid':
        ma5 = amv_series.rolling(window=5, min_periods=5).mean()
        ma20 = amv_series.rolling(window=20, min_periods=20).mean()
        smoothed = 0.6 * ma5 + 0.4 * ma20
    else:
        smoothed = amv_series.copy()

    # 成本均线
    if cost_ma_method == 'ema':
        cost_ma = smoothed.ewm(span=cost_ma_period, adjust=False).mean()
    else:
        cost_ma = smoothed.rolling(window=cost_ma_period, min_periods=cost_ma_period).mean()

    # X指标
    x_series = (smoothed - cost_ma) / cost_ma * 100.0

    return x_series


def apply_hysteresis(x_series, upper_threshold, lower_threshold):
    """
    施加迟滞滤波（状态机）

    规则：
      - X >= upper → state=1 (做多)
      - X <= lower → state=0 (不做多/做空)
      - 中间地带保持前一状态

    返回:
        state_series: 0/1状态序列
    """
    state = np.zeros(len(x_series), dtype=int)
    current_state = 0

    x_values = x_series.values
    for i in range(len(x_values)):
        val = x_values[i]
        if np.isnan(val):
            state[i] = current_state
            continue
        if val >= upper_threshold:
            current_state = 1
        elif val <= lower_threshold:
            current_state = 0
        state[i] = current_state

    return pd.Series(state, index=x_series.index, dtype=int)


# ══════════════════════════════════════════════════════════
#  评估函数
# ══════════════════════════════════════════════════════════

def evaluate_timing(state_series, index_returns, min_transitions=4):
    """
    评估择时效果

    参数:
        state_series: 0/1状态序列
        index_returns: 指数日收益率
        min_transitions: 最少切换次数（太少说明参数太宽或太窄）

    返回:
        metrics: 评估指标字典，或None（参数无效）
    """
    # 对齐
    common_idx = state_series.index.intersection(index_returns.index)
    if len(common_idx) < 100:
        return None

    states = state_series.reindex(common_idx)
    returns = index_returns.reindex(common_idx)

    # 过滤NaN
    valid = states.notna() & returns.notna()
    states = states[valid]
    returns = returns[valid]

    if len(states) == 0:
        return None

    # 切换次数
    transitions = (states.diff() != 0).sum() - 1
    if transitions < min_transitions:
        return None

    # 分离做多/做空期间
    bull_mask = states == 1
    bear_mask = states == 0

    bull_returns = returns[bull_mask]
    bear_returns = returns[bear_mask]

    bull_days = len(bull_returns)
    bear_days = len(bear_returns)

    if bull_days < 20 or bear_days < 20:
        return None

    # 做多期间累计收益
    bull_cum = (1 + bull_returns).prod() - 1
    # 做空期间累计收益（如果持有的话会亏多少）
    bear_cum = (1 + bear_returns).prod() - 1

    # 年化
    total_days = len(states)
    bull_annual = (1 + bull_cum) ** (252 / bull_days) - 1 if bull_days > 0 else 0
    bear_annual = (1 + bear_cum) ** (252 / bear_days) - 1 if bear_days > 0 else 0

    # 策略总收益（做多期间持有，做空期间空仓）
    strategy_cum = (1 + returns * states).prod() - 1
    strategy_annual = (1 + strategy_cum) ** (252 / total_days) - 1

    # 买入持有收益
    bh_cum = (1 + returns).prod() - 1
    bh_annual = (1 + bh_cum) ** (252 / total_days) - 1

    # 做多胜率（正收益天数占比）
    bull_win_rate = (bull_returns > 0).mean() * 100 if bull_days > 0 else 0

    # 做多期间最大回撤
    bull_equity = (1 + returns * states).cumprod()
    peak = bull_equity.cummax()
    drawdown = (bull_equity - peak) / peak
    max_drawdown = drawdown.min() * 100  # 百分比

    # 做多时间占比
    bull_ratio = bull_days / total_days * 100

    # 综合评分：策略年化 - 做空期间年化（越大说明择时越好）
    timing_score = strategy_annual - bh_annual  # 超额收益

    return {
        'bull_cum_pct': bull_cum * 100,
        'bear_cum_pct': bear_cum * 100,
        'bull_annual_pct': bull_annual * 100,
        'bear_annual_pct': bear_annual * 100,
        'strategy_annual_pct': strategy_annual * 100,
        'bh_annual_pct': bh_annual * 100,
        'excess_annual_pct': (strategy_annual - bh_annual) * 100,
        'bull_win_rate': bull_win_rate,
        'max_drawdown_pct': max_drawdown,
        'bull_ratio_pct': bull_ratio,
        'transitions': transitions,
        'bull_days': bull_days,
        'bear_days': bear_days,
        'total_days': total_days,
    }


# ══════════════════════════════════════════════════════════
#  参数网格搜索
# ══════════════════════════════════════════════════════════

def run_optimization(amv_series, index_returns):
    """
    穷搜最优参数组合

    搜索空间（共 6×6×8×8 = 2304 种组合）:
      - smooth_period: [5, 8, 10, 13, 15, 20]
      - cost_ma_period: [13, 17, 21, 26, 34, 42]
      - upper_threshold: [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
      - lower_threshold: [-1.0, -1.5, -2.0, -2.3, -2.5, -3.0, -3.5, -4.0]
    """
    # 搜索空间
    smooth_periods = [5, 8, 10, 13, 15, 20]
    cost_ma_periods = [13, 17, 21, 26, 34, 42]
    upper_thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
    lower_thresholds = [-1.0, -1.5, -2.0, -2.3, -2.5, -3.0, -3.5, -4.0]
    smooth_methods = ['sma']  # 指南针用SMA，先固定
    cost_ma_methods = ['sma']  # 指南针用SMA，先固定

    total_combos = (len(smooth_periods) * len(cost_ma_periods) *
                    len(upper_thresholds) * len(lower_thresholds))
    print(f"\n{'='*80}")
    print(f"OAMV 活跃市值择时参数优化器")
    print(f"{'='*80}")
    print(f"搜索空间: {total_combos} 种参数组合")
    print(f"  smooth_period: {smooth_periods}")
    print(f"  cost_ma_period: {cost_ma_periods}")
    print(f"  upper_threshold: {upper_thresholds}")
    print(f"  lower_threshold: {lower_thresholds}")
    print(f"{'='*80}\n")

    results = []
    start_time = time.time()
    count = 0

    for sm in smooth_methods:
        for cm in cost_ma_methods:
            for sp in smooth_periods:
                # 预计算X序列（smooth_period固定时只算一次）
                for cp in cost_ma_periods:
                    x_series = compute_oamv_x(amv_series, smooth_method=sm,
                                             smooth_period=sp, cost_ma_method=cm,
                                             cost_ma_period=cp)

                    for upper in upper_thresholds:
                        for lower in lower_thresholds:
                            count += 1
                            if count % 200 == 0:
                                elapsed = time.time() - start_time
                                eta = elapsed / count * (total_combos - count)
                                print(f"  进度: {count}/{total_combos} ({count/total_combos*100:.1f}%) "
                                      f"| 有效: {len(results)} | ETA: {eta:.0f}s")

                            # 上阈值必须大于下阈值
                            if upper <= abs(lower):
                                continue

                            state_series = apply_hysteresis(x_series, upper, lower)
                            metrics = evaluate_timing(state_series, index_returns)

                            if metrics is None:
                                continue

                            metrics.update({
                                'smooth_method': sm,
                                'smooth_period': sp,
                                'cost_ma_method': cm,
                                'cost_ma_period': cp,
                                'upper_threshold': upper,
                                'lower_threshold': lower,
                            })
                            results.append(metrics)

    elapsed = time.time() - start_time
    print(f"\n搜索完成! 耗时: {elapsed:.1f}s | 有效组合: {len(results)}/{total_combos}")

    return results


def print_top_results(results, top_n=20):
    """打印最优参数结果"""
    if not results:
        print("❌ 无有效结果")
        return

    df = pd.DataFrame(results)

    # ── 排序指标1：策略年化收益最高 ──
    print(f"\n{'='*80}")
    print(f"🏆 TOP {top_n} 最优参数（按策略年化收益排序）")
    print(f"{'='*80}")
    top_strategy = df.nlargest(top_n, 'strategy_annual_pct')
    for i, row in top_strategy.iterrows():
        print(f"\n  #{top_strategy.index.get_loc(i)+1}  "
              f"SMA({row['smooth_period']}) + CostMA({row['cost_ma_period']}) "
              f"| 做多>{row['upper_threshold']:+.1f}% 做空<{row['lower_threshold']:.1f}%")
        print(f"       策略年化: {row['strategy_annual_pct']:+.2f}% "
              f"| 买持年化: {row['bh_annual_pct']:+.2f}% "
              f"| 超额: {row['excess_annual_pct']:+.2f}%")
        print(f"       做多年化: {row['bull_annual_pct']:+.2f}% "
              f"| 做空年化: {row['bear_annual_pct']:+.2f}% "
              f"| 做多占比: {row['bull_ratio_pct']:.1f}%")
        print(f"       最大回撤: {row['max_drawdown_pct']:.1f}% "
              f"| 切换次数: {int(row['transitions'])} "
              f"| 做多胜率: {row['bull_win_rate']:.1f}%")

    # ── 排序指标2：超额收益最高 ──
    print(f"\n{'='*80}")
    print(f"🎯 TOP {top_n} 最优参数（按超额收益排序 = 策略 - 买持）")
    print(f"{'='*80}")
    top_excess = df.nlargest(top_n, 'excess_annual_pct')
    for i, row in top_excess.iterrows():
        print(f"\n  #{top_excess.index.get_loc(i)+1}  "
              f"SMA({row['smooth_period']}) + CostMA({row['cost_ma_period']}) "
              f"| 做多>{row['upper_threshold']:+.1f}% 做空<{row['lower_threshold']:.1f}%")
        print(f"       超额年化: {row['excess_annual_pct']:+.2f}% "
              f"| 策略年化: {row['strategy_annual_pct']:+.2f}% "
              f"| 买持年化: {row['bh_annual_pct']:+.2f}%")
        print(f"       做多年化: {row['bull_annual_pct']:+.2f}% "
              f"| 做空年化: {row['bear_annual_pct']:+.2f}% "
              f"| 回撤: {row['max_drawdown_pct']:.1f}%")
        print(f"       做多占比: {row['bull_ratio_pct']:.1f}% "
              f"| 切换: {int(row['transitions'])}次 "
              f"| 做多{int(row['bull_days'])}天/做空{int(row['bear_days'])}天")

    # ── 排序指标3：做空期间跌幅最大（避开下跌能力） ──
    print(f"\n{'='*80}")
    print(f"🛡️ TOP 10 最优避险参数（做空期间跌幅最大 = 成功避开的下跌）")
    print(f"{'='*80}")
    top_avoid = df.nsmallest(10, 'bear_annual_pct')
    for i, row in top_avoid.iterrows():
        print(f"\n  #{top_avoid.index.get_loc(i)+1}  "
              f"SMA({row['smooth_period']}) + CostMA({row['cost_ma_period']}) "
              f"| 做多>{row['upper_threshold']:+.1f}% 做空<{row['lower_threshold']:.1f}%")
        print(f"       做空期间年化: {row['bear_annual_pct']:+.2f}%（成功避开！）"
              f"| 做多年化: {row['bull_annual_pct']:+.2f}%")
        print(f"       策略年化: {row['strategy_annual_pct']:+.2f}% "
              f"| 超额: {row['excess_annual_pct']:+.2f}% "
              f"| 做多占比: {row['bull_ratio_pct']:.1f}%")

    # ── 综合最优（平衡收益和操作频率） ──
    print(f"\n{'='*80}")
    print(f"⭐ 综合最优参数（超额>3% + 切换<30次 + 做多占比40-80%）")
    print(f"{'='*80}")
    balanced = df[
        (df['excess_annual_pct'] > 3.0) &
        (df['transitions'] < 30) &
        (df['bull_ratio_pct'] > 40) &
        (df['bull_ratio_pct'] < 80)
    ].nlargest(10, 'excess_annual_pct')

    if len(balanced) == 0:
        # 放宽条件
        balanced = df[
            (df['excess_annual_pct'] > 1.0) &
            (df['transitions'] < 40) &
            (df['bull_ratio_pct'] > 30) &
            (df['bull_ratio_pct'] < 85)
        ].nlargest(10, 'excess_annual_pct')

    for i, row in balanced.iterrows():
        print(f"\n  #{balanced.index.get_loc(i)+1}  "
              f"SMA({row['smooth_period']}) + CostMA({row['cost_ma_period']}) "
              f"| ✅ 做多>{row['upper_threshold']:+.1f}% ✅ 做空<{row['lower_threshold']:.1f}%")
        print(f"       超额年化: {row['excess_annual_pct']:+.2f}% "
              f"| 策略年化: {row['strategy_annual_pct']:+.2f}% "
              f"| 买持年化: {row['bh_annual_pct']:+.2f}%")
        print(f"       做多年化: {row['bull_annual_pct']:+.2f}% "
              f"| 做空年化: {row['bear_annual_pct']:+.2f}%")
        print(f"       做多占比: {row['bull_ratio_pct']:.1f}% "
              f"| 回撤: {row['max_drawdown_pct']:.1f}% "
              f"| 切换: {int(row['transitions'])}次")

    # ── 你当前用的参数评估 ──
    print(f"\n{'='*80}")
    print(f"📊 你当前参数的表现 (SMA(8) + CostMA(21) | 做多>+4.0% 做空<-2.3%)")
    print(f"{'='*80}")
    current = df[
        (df['smooth_period'] == 8) &
        (df['cost_ma_period'] == 21) &
        (df['upper_threshold'] == 4.0) &
        (df['lower_threshold'] == -2.3)
    ]
    if len(current) > 0:
        row = current.iloc[0]
        print(f"  策略年化: {row['strategy_annual_pct']:+.2f}% "
              f"| 买持年化: {row['bh_annual_pct']:+.2f}% "
              f"| 超额: {row['excess_annual_pct']:+.2f}%")
        print(f"  做多年化: {row['bull_annual_pct']:+.2f}% "
              f"| 做空年化: {row['bear_annual_pct']:+.2f}%")
        print(f"  做多占比: {row['bull_ratio_pct']:.1f}% "
              f"| 回撤: {row['max_drawdown_pct']:.1f}% "
              f"| 切换: {int(row['transitions'])}次")

        # 在所有结果中的排名
        rank_strategy = (df['strategy_annual_pct'] > row['strategy_annual_pct']).sum() + 1
        rank_excess = (df['excess_annual_pct'] > row['excess_annual_pct']).sum() + 1
        print(f"  策略年化排名: {rank_strategy}/{len(df)} "
              f"| 超额排名: {rank_excess}/{len(df)}")
    else:
        print("  ⚠️ 当前参数未在搜索空间中")

    return df


# ══════════════════════════════════════════════════════════
#  附加：固定平滑参数，细粒度搜索阈值
# ══════════════════════════════════════════════════════════

def fine_tune_thresholds(amv_series, index_returns, smooth_period=8, cost_ma_period=21):
    """
    固定平滑参数后，对阈值进行细粒度搜索

    搜索空间：
      - upper: 1.0 ~ 8.0, step=0.5
      - lower: -0.5 ~ -5.0, step=0.5
    """
    print(f"\n{'='*80}")
    print(f"🔍 细粒度阈值搜索 (固定 SMA({smooth_period}) + CostMA({cost_ma_period}))")
    print(f"{'='*80}")

    x_series = compute_oamv_x(amv_series, smooth_method='sma',
                              smooth_period=smooth_period, cost_ma_method='sma',
                              cost_ma_period=cost_ma_period)

    uppers = np.arange(1.0, 8.5, 0.5)
    lowers = np.arange(-0.5, -5.5, -0.5)

    results = []
    for upper in uppers:
        for lower in lowers:
            if upper <= abs(lower) * 0.5:  # 上阈值不能太小
                continue
            state_series = apply_hysteresis(x_series, upper, lower)
            metrics = evaluate_timing(state_series, index_returns, min_transitions=3)
            if metrics is None:
                continue
            metrics['upper_threshold'] = upper
            metrics['lower_threshold'] = lower
            results.append(metrics)

    if not results:
        print("  无有效结果")
        return

    df = pd.DataFrame(results)

    # 打印热力图（超额收益）
    print(f"\n  超额年化收益热力图 (行=做多阈值, 列=做空阈值):")
    print(f"  {'':>6}", end="")
    for lower in sorted(df['lower_threshold'].unique()):
        print(f"  {lower:>5.1f}", end="")
    print()

    for upper in sorted(df['upper_threshold'].unique()):
        print(f"  {upper:>5.1f}", end="")
        for lower in sorted(df['lower_threshold'].unique()):
            cell = df[(df['upper_threshold'] == upper) & (df['lower_threshold'] == lower)]
            if len(cell) > 0:
                val = cell.iloc[0]['excess_annual_pct']
                if val > 5:
                    marker = f" {val:>4.1f}★"
                elif val > 3:
                    marker = f" {val:>4.1f}●"
                elif val > 0:
                    marker = f" {val:>4.1f} "
                else:
                    marker = f" {val:>4.1f}×"
                print(marker, end="")
            else:
                print(f"  {'---':>5}", end="")
        print()

    # Top 5
    top5 = df.nlargest(5, 'excess_annual_pct')
    print(f"\n  🏆 细粒度搜索 Top 5:")
    for i, row in top5.iterrows():
        print(f"    做多>{row['upper_threshold']:+.1f}% 做空<{row['lower_threshold']:.1f}% "
              f"→ 超额{row['excess_annual_pct']:+.2f}% "
              f"策略{row['strategy_annual_pct']:+.2f}% "
              f"切换{int(row['transitions'])}次 "
              f"做多{row['bull_ratio_pct']:.0f}%时间")


# ══════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("OAMV活跃市值择时参数优化器 — 寻找最优做多/做空阈值")
    print("=" * 80)
    print()
    print("原理：全市场活跃市值 = Σ(流通市值 × 自由流通换手率)")
    print("      X = (SMA平滑 - 成本均线) / 成本均线 × 100%")
    print("      X > 做多阈值 → 进入做多状态")
    print("      X < 做空阈值 → 退出做多状态（空仓）")
    print("      中间地带 → 保持前一状态（迟滞）")
    print()

    # 1. 加载数据
    print("[1/4] 加载数据...\n")
    amv_series = load_amv_cache()
    if amv_series is None:
        sys.exit(1)

    index_df = load_index_daily()
    if index_df is None:
        sys.exit(1)

    index_returns = index_df['daily_return'].dropna()

    # 2. 全参数网格搜索
    print("\n[2/4] 全参数网格搜索...")
    results = run_optimization(amv_series, index_returns)

    # 3. 打印结果
    print("\n[3/4] 结果分析...")
    results_df = print_top_results(results)

    # 4. 细粒度搜索（基于最优平滑参数）
    print("\n[4/4] 细粒度阈值搜索...")
    if results_df is not None and len(results_df) > 0:
        # 找最优平滑参数
        best = results_df.nlargest(1, 'excess_annual_pct').iloc[0]
        best_sp = int(best['smooth_period'])
        best_cp = int(best['cost_ma_period'])
        fine_tune_thresholds(amv_series, index_returns, best_sp, best_cp)

        # 同时用你当前的平滑参数做细粒度搜索
        if best_sp != 8 or best_cp != 21:
            fine_tune_thresholds(amv_series, index_returns, 8, 21)

    # 保存完整结果
    if results_df is not None:
        output_file = Path(__file__).parent / "results" / "oamv_optimizer_results.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print(f"\n完整结果已保存: {output_file}")

    print(f"\n{'='*80}")
    print("优化完成!")
    print("='*80}")
