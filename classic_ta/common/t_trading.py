"""
Intraday T-trading helper.

This module provides a lightweight, dependency-free fallback for the
signal scanner. The existing message templates only require a compact
summary of T-trading intent, so we derive that summary from already
computed price / moving-average fields in the daily dataframe.
"""

from __future__ import annotations

import math

import pandas as pd


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _describe_slope(delta, atr):
    if atr <= 0:
        atr = 1.0
    normalized = delta / atr
    if normalized >= 0.35:
        return "上行"
    if normalized >= 0.1:
        return "偏强"
    if normalized <= -0.35:
        return "下行"
    if normalized <= -0.1:
        return "偏弱"
    return "走平"


def analyze_t_trading(df: pd.DataFrame, idx: int | None = None) -> dict:
    """Return a compact T-trading suggestion for message rendering.

    The output keys are intentionally aligned with the current
    `message_builder.py` expectations.
    """
    if df is None or len(df) < 3:
        return {
            "mode": "观望",
            "yellow_slope": "走平",
            "amplitude": 0.0,
            "buy_signal": "",
            "sell_signal": "",
            "risk_alert": "数据不足",
        }

    if idx is None or idx >= len(df):
        idx = len(df) - 1

    latest = df.iloc[idx]
    prev = df.iloc[max(0, idx - 1)]
    start = max(0, idx - 4)
    window = df.iloc[start : idx + 1]

    close = _safe_float(latest.get("Close"))
    high = _safe_float(window.get("High", pd.Series(dtype=float)).max(), close)
    low = _safe_float(window.get("Low", pd.Series(dtype=float)).min(), close)
    yellow = _safe_float(latest.get("yellow_line"))
    prev_yellow = _safe_float(prev.get("yellow_line"), yellow)
    white = _safe_float(latest.get("white_line"))
    atr = _safe_float(latest.get("atr14"), max(close * 0.02, 0.01))

    amplitude = ((high - low) / close * 100.0) if close > 0 else 0.0
    slope_delta = yellow - prev_yellow
    slope_label = _describe_slope(slope_delta, atr)
    distance_to_yellow = ((close - yellow) / yellow * 100.0) if yellow > 0 else 0.0
    white_above_yellow = white >= yellow

    result = {
        "mode": "观望",
        "yellow_slope": slope_label,
        "amplitude": round(amplitude, 1),
        "buy_signal": "",
        "sell_signal": "",
        "risk_alert": "",
    }

    if close <= 0 or yellow <= 0:
        result["risk_alert"] = "关键指标缺失"
        return result

    if amplitude < 2.0:
        result["risk_alert"] = "振幅不足"
        return result

    if slope_label in ("上行", "偏强") and white_above_yellow:
        result["mode"] = "正T"
        if distance_to_yellow <= 1.5:
            result["buy_signal"] = f"回踩黄线附近可低吸({yellow:.2f}附近)"
        else:
            result["buy_signal"] = f"等待靠近黄线再接({yellow:.2f}附近)"
        result["sell_signal"] = f"冲高靠近区间上沿可减仓({high:.2f}附近)"
        if distance_to_yellow >= 5.0:
            result["risk_alert"] = "价格偏离黄线过大，勿追高"
        return result

    if slope_label in ("下行", "偏弱") or not white_above_yellow:
        result["mode"] = "反T"
        result["buy_signal"] = f"仅在急跌靠近区间下沿试错({low:.2f}附近)"
        result["sell_signal"] = f"反弹接近黄线先减仓({yellow:.2f}附近)"
        result["risk_alert"] = "趋势偏弱，做T仓位宜轻"
        return result

    result["risk_alert"] = "方向不明，等待结构更清晰"
    return result
