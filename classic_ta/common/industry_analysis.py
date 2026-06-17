"""
行业分析模块

从 v63_daily_push.py 提取的公共行业分析逻辑。
计算行业动量、冷热分布、轮动信号等。
"""
import logging
from collections import defaultdict

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
