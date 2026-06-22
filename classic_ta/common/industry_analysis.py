"""
行业分析模块

从 v63_daily_push.py 提取的公共行业分析逻辑。
计算行业动量、冷热分布、轮动信号、行业滞涨股识别等。
"""
import logging
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)


def compute_industry_analysis(signals_data, industry_map, best_params):
    """计算行业间分析：动量、信号数量、热度排名、轮动信号

    Args:
        signals_data: {ts_code: df} 已计算的指标+信号数据
        industry_map: {ts_code: industry_name} 股票→行业映射
        best_params: dict 策略参数

    Returns:
        list[dict]: 行业统计列表，按动量降序排列
    """
    from classic_ta.v62_ambush_model import compute_industry_momentum

    mom_days = best_params.get("industry_momentum_days", 10)

    # 1. 计算行业动量
    mom_df = compute_industry_momentum(signals_data, industry_map, mom_days)

    # 2. 获取最新动量值和动量变化
    if mom_df.empty or len(mom_df) < 6:
        return []

    latest_mom = mom_df.iloc[-1]
    lookback = min(5, len(mom_df) - 1)
    prev_mom = mom_df.iloc[-1 - lookback]

    # 3. 统计每个行业的信号数量
    industry_signal_count = defaultdict(int)
    industry_stock_count = defaultdict(int)

    for ts_code, df in signals_data.items():
        industry = industry_map.get(ts_code, "")
        if not industry:
            continue
        industry_stock_count[industry] += 1
        if len(df) > 0 and df.iloc[-1].get("ambush_signal", False):
            industry_signal_count[industry] += 1

    # 4. 构建行业统计
    industry_stats = []
    for industry in mom_df.columns:
        momentum = float(latest_mom.get(industry, 0))
        prev_momentum = float(prev_mom.get(industry, 0))
        momentum_change = momentum - prev_momentum
        signal_count = industry_signal_count.get(industry, 0)
        stock_count = industry_stock_count.get(industry, 0)

        # 热度分类
        if momentum > 0.05:
            hot_cold = "火热"
        elif momentum > 0.02:
            hot_cold = "偏热"
        elif momentum > 0:
            hot_cold = "微热"
        elif momentum > -0.02:
            hot_cold = "微冷"
        elif momentum > -0.05:
            hot_cold = "偏冷"
        else:
            hot_cold = "冰冷"

        # 轮动信号
        was_hot = prev_momentum > best_params.get("industry_momentum_threshold", 0.02)
        is_hot = momentum > best_params.get("industry_momentum_threshold", 0.02)
        if not was_hot and is_hot:
            rotation = "轮入"
        elif was_hot and not is_hot:
            rotation = "轮出"
        elif is_hot and momentum_change > 0.01:
            rotation = "加速"
        elif is_hot and momentum_change < -0.01:
            rotation = "减速"
        elif not is_hot and momentum_change > 0.01:
            rotation = "回暖"
        elif not is_hot and momentum_change < -0.01:
            rotation = "恶化"
        else:
            rotation = "平稳"

        industry_stats.append({
            "name": industry,
            "momentum": round(momentum * 100, 2),
            "momentum_change": round(momentum_change * 100, 2),
            "rotation": rotation,
            "signal_count": signal_count,
            "stock_count": stock_count,
            "hot_cold": hot_cold,
        })

    # 按动量排序
    industry_stats.sort(key=lambda x: x["momentum"], reverse=True)
    return industry_stats


def compute_industry_ma_lines(all_signals_data, industry_map):
    """计算行业级别的黄白线（平均各成分股的黄白线）

    通过平均行业内所有股票的 white_line 和 yellow_line，得到行业级别的趋势线。
    白线>黄线 = 行业多头（短期趋势强于中期趋势）

    参数:
        all_signals_data: {ts_code: df} 含 white_line/yellow_line 列
        industry_map: {ts_code: industry_name}

    返回:
        DataFrame: MultiIndex columns (industry, 'white'/'yellow'), index=Date
    """
    industry_whites = defaultdict(list)
    industry_yellows = defaultdict(list)

    for ts_code, df in all_signals_data.items():
        industry = industry_map.get(ts_code, "")
        if not industry or df is None:
            continue
        if "white_line" not in df.columns or "yellow_line" not in df.columns:
            continue
        w = df["white_line"].copy()
        y = df["yellow_line"].copy()
        w.name = ts_code
        y.name = ts_code
        industry_whites[industry].append(w)
        industry_yellows[industry].append(y)

    result = {}
    for industry in industry_whites:
        if len(industry_whites[industry]) < 3:
            continue
        avg_w = pd.concat(industry_whites[industry], axis=1).mean(axis=1)
        avg_y = pd.concat(industry_yellows[industry], axis=1).mean(axis=1)
        result[(industry, "white")] = avg_w
        result[(industry, "yellow")] = avg_y

    if not result:
        return pd.DataFrame()
    return pd.DataFrame(result)


def compute_industry_lag_signals(
    signals: list,
    mom_df: pd.DataFrame,
    best_params: dict,
    industry_ma=None,
) -> list:
    """为每个信号计算个股动量，标记动量达标信号

    核心逻辑（纯个股动量过滤，行业仅作参考展示）：
      动量达标 = 个股N日收益 > 阈值（甜蜜点 -5%）
      含义: 个股近N日跌幅可控，非深度下跌中的接飞刀

    参数:
        signals: 信号列表（含 stock_ret_n 字段）
        mom_df: 行业动量DataFrame（仅用于展示）
        best_params: 策略参数
        industry_ma: 行业黄白线DataFrame（仅用于展示）

    返回:
        list: 增强后的信号列表，每个dict新增:
            - stock_momentum_ok: bool 个股动量是否达标
            - momentum_tag: str 动量标签
            - industry_mom: 行业动量（展示用）
    """
    if not best_params.get("lag_filter_enabled", False):
        return signals

    mover_min_return = best_params.get("lag_stock_max_return", -0.03)  # 甜蜜点 -3%
    score_boost = best_params.get("lag_score_boost", 1)

    for s in signals:
        industry = s.get("industry", "")
        stock_ret_n = s.get("stock_ret_n")

        # 获取信号日的行业动量（仅用于展示）
        industry_mom = None
        if industry and industry in mom_df.columns:
            try:
                signal_date = pd.Timestamp(s["signal_date"])
                aligned = mom_df[industry].reindex([signal_date], method="nearest", tolerance="1D")
                if not aligned.empty and pd.notna(aligned.iloc[0]):
                    industry_mom = float(aligned.iloc[0])
            except Exception:
                pass

        # 个股动量判定（核心过滤条件）
        stock_momentum_ok = False
        momentum_tag = ""

        if stock_ret_n is not None:
            stock_momentum_ok = stock_ret_n > mover_min_return
            if stock_momentum_ok:
                momentum_tag = f"动量{stock_ret_n * 100:+.1f}%"
                # 动量达标信号获得评分加分
                if score_boost > 0:
                    s["entry_quality_score"] = min(8, s.get("entry_quality_score", 0) + score_boost)
            else:
                momentum_tag = f"动量不足{stock_ret_n * 100:+.1f}%"

        s["stock_momentum_ok"] = stock_momentum_ok
        s["momentum_tag"] = momentum_tag
        s["industry_mom"] = industry_mom
        # 保留兼容字段
        s["is_laggard_in_strong_sector"] = stock_momentum_ok
        s["lag_tag"] = momentum_tag

    return signals
