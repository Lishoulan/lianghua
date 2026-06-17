"""
信号详情分析模块

从 v63_daily_push.py 提取的公共信号分析逻辑。
对信号日进行威科夫/VPA/蜡烛图/支撑阻力等多维度解读。
"""
import logging

logger = logging.getLogger(__name__)


def analyze_signal_detail(df, signal_idx, best_params):
    """对信号日进行详细分析

    Args:
        df: 已计算指标的DataFrame
        signal_idx: 信号日索引（整数位置）
        best_params: dict 策略参数

    Returns:
        dict: {
            'wyckoff': list[str],  # 威科夫解读
            'vpa': list[str],      # VPA量价解读
            'candle': list[str],   # 蜡烛图解读
            'support': float,      # 支撑位
            'resistance': float,   # 阻力位
        }
    """
    row = df.iloc[signal_idx]
    prev = df.iloc[signal_idx - 1] if signal_idx > 0 else row
    body = abs(row["Close"] - row["Open"])
    amplitude = row["High"] - row["Low"]
    lower_shadow = min(row["Close"], row["Open"]) - row["Low"]
    upper_shadow = row["High"] - max(row["Close"], row["Open"])

    analysis = {}

    # 威科夫解读
    wyckoff_parts = []
    if row.get("tag_sos_anchor", False):
        wyckoff_parts.append("SOS需求确认(主力入场)")
    if row.get("tag_no_supply", False):
        wyckoff_parts.append("No Supply供应枯竭")
    if row.get("tag_test", False):
        wyckoff_parts.append("Test测试柱(需求保护)")
    window = best_params["ambush_window"]
    if signal_idx >= window:
        recent_sos = df.iloc[signal_idx - window:signal_idx + 1]["tag_sos_anchor"].any()
        if recent_sos and not row.get("tag_sos_anchor", False):
            wyckoff_parts.append(f"近{window}日有SOS锚定(LPS回踩)")
    if row["J"] < best_params["ambush_j_oversold"]:
        wyckoff_parts.append(f"J={row['J']:.0f}情绪冰点(超卖)")
    analysis["wyckoff"] = wyckoff_parts if wyckoff_parts else ["标准潜伏信号"]

    # VPA量价解读
    vpa_parts = []
    vol_ratio = row["Volume"] / row["volume_ma"] if row["volume_ma"] > 0 else 0
    if vol_ratio < best_params["ambush_vol_shrink"]:
        vpa_parts.append(f"缩量({vol_ratio:.1%}均量=供应枯竭)")
    else:
        vpa_parts.append(f"量比{vol_ratio:.2f}")
    if body / (row["Close"] + 1e-8) < best_params["ambush_body_pct"]:
        vpa_parts.append("小实体(多空平衡/拒绝下跌)")
    if lower_shadow > body * 1.5:
        vpa_parts.append("下影线支撑(需求托底)")
    analysis["vpa"] = vpa_parts

    # 蜡烛图解读
    candle_parts = []
    if body < amplitude * 0.1:
        candle_parts.append("十字星(方向选择)")
    elif row["Close"] > row["Open"] and (row["Close"] - row["Low"]) / (amplitude + 1e-8) > 0.7:
        candle_parts.append("阳线收高(强势)")
    elif row["Close"] < row["Open"] and lower_shadow > body * 2:
        candle_parts.append("锤子线(潜在反转)")
    if not candle_parts:
        candle_parts.append("阳线" if row["Close"] >= row["Open"] else "阴线")
    analysis["candle"] = candle_parts

    # 支撑/阻力
    analysis["support"] = round(float(row["yellow_line"] - 0.5 * row["atr14"]), 2)
    analysis["resistance"] = round(float(row["yellow_line"] + 1.5 * row["atr14"]), 2)

    return analysis
