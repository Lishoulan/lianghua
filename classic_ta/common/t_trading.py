"""
知行合一双轨趋势系统 — 做T策略分析模块
========================================
基于白线(EMA(EMA(CLOSE,10),10))和黄线((MA14+MA28+MA57+MA114)/4)的双轨系统，
分析个股做T（正T先买后卖 / 倒T先卖后买）的执行建议。

核心逻辑：
  1. 黄线斜率定基调：向上→正T，向下/走平→倒T
  2. 振幅过滤：<1.5%→观望（摩擦成本过高）
  3. 正T：回踩黄线买 / 白线上方卖
  4. 倒T：触黄线卖 / 白线下方买回
  5. 纠错：突破卖出价1.5%买回 / 跌破黄线1.5%止损
"""

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── 参数常量 ──
SLOPE_LOOKBACK = 5          # 黄线斜率回看天数
SLOPE_UP_THRESHOLD = 0.3    # 黄线斜率 >0.3% 判定向上
SLOPE_DOWN_THRESHOLD = -0.3 # 黄线斜率 <-0.3% 判定向下
MIN_AMPLITUDE_PCT = 1.5     # 最小振幅%，低于此观望
TOUCH_TOLERANCE_PCT = 0.5   # 触碰黄线容差%
J_OVERSOLD = 20             # KDJ J值超卖阈值（正T超跌买）
J_OVERBOUGHT = 80           # KDJ J值超买阈值（正T卖）
J_EXTREME_OVERBOUGHT = 100  # KDJ J值极度超买（倒T卖）
STOP_LOSS_PCT = 1.5         # 纠错止损%
SHORT_VOLUME_RATIO = 0.8    # 缩量判定（量比<0.8为缩量）


def analyze_t_trading(df: pd.DataFrame, signal_idx: int) -> Dict:
    """分析信号股的做T机会

    基于双轨趋势系统（白线+黄线），判定做T模式并给出买卖点建议。

    参数:
        df: 已计算指标的DataFrame（含 white_line, yellow_line, J, Close, High, Low, Volume, volume_ma）
        signal_idx: 信号日索引（通常为 len(df)-1）

    返回:
        dict: {
            "mode": "正T" | "倒T" | "观望",
            "yellow_slope": "向上" | "向下" | "走平",
            "buy_signal": str | None,
            "sell_signal": str | None,
            "risk_alert": str | None,
            "amplitude": float,
        }
    """
    if signal_idx < SLOPE_LOOKBACK or len(df) <= signal_idx:
        return _default_result()

    row = df.iloc[signal_idx]
    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])
    white = float(row.get("white_line", 0))
    yellow = float(row.get("yellow_line", 0))
    j_val = float(row.get("J", 50))
    volume = float(row.get("Volume", 0))
    volume_ma = float(row.get("volume_ma", 1))

    if close <= 0 or yellow <= 0 or white <= 0:
        return _default_result()

    # ── 振幅计算 ──
    amplitude = (high - low) / close * 100

    # ── 振幅过滤：日内振幅 < 1.5% → 观望 ──
    if amplitude < MIN_AMPLITUDE_PCT:
        return {
            "mode": "观望",
            "yellow_slope": _get_slope_label(df, signal_idx),
            "buy_signal": None,
            "sell_signal": None,
            "risk_alert": f"振幅仅{amplitude:.1f}%，不足{MIN_AMPLITUDE_PCT}%，摩擦成本过高",
            "amplitude": round(amplitude, 2),
        }

    # ── 黄线斜率判定 ──
    prev_yellow = float(df.iloc[signal_idx - SLOPE_LOOKBACK].get("yellow_line", 0))
    if prev_yellow <= 0:
        slope_label = "走平"
        slope_pct = 0.0
    else:
        slope_pct = (yellow - prev_yellow) / prev_yellow * 100
        if slope_pct > SLOPE_UP_THRESHOLD:
            slope_label = "向上"
        elif slope_pct < SLOPE_DOWN_THRESHOLD:
            slope_label = "向下"
        else:
            slope_label = "走平"

    # ── 价格与均线的空间关系 ──
    price_to_yellow_pct = (close - yellow) / yellow * 100  # 价格相对黄线的偏离%
    price_to_white_pct = (close - white) / white * 100    # 价格相对白线的偏离%
    is_above_white = close > white
    is_above_yellow = close > yellow
    is_near_yellow = abs(price_to_yellow_pct) <= TOUCH_TOLERANCE_PCT  # 触碰黄线±0.5%

    # ── 量比 ──
    vol_ratio = volume / volume_ma if volume_ma > 0 else 1.0
    is_shrink_volume = vol_ratio < SHORT_VOLUME_RATIO

    # ── 模式判定 + 信号生成 ──
    if slope_label == "向上":
        return _analyze_long_t(
            close, white, yellow, j_val, amplitude,
            price_to_yellow_pct, price_to_white_pct,
            is_above_white, is_above_yellow, is_near_yellow,
            slope_label, slope_pct,
        )
    else:
        return _analyze_short_t(
            close, white, yellow, j_val, amplitude,
            price_to_yellow_pct, price_to_white_pct,
            is_above_white, is_above_yellow, is_near_yellow,
            is_shrink_volume, vol_ratio,
            slope_label, slope_pct,
        )


def _analyze_long_t(
    close, white, yellow, j_val, amplitude,
    price_to_yellow_pct, price_to_white_pct,
    is_above_white, is_above_yellow, is_near_yellow,
    slope_label, slope_pct,
) -> Dict:
    """正T模式分析（黄线向上，先买后卖）"""
    buy_signals = []
    sell_signals = []
    risk_alerts = []

    # ── 买入信号 ──
    # 1. 回踩黄线买：价格触碰黄线且未跌破（±0.5%内）
    if is_near_yellow and close >= yellow * (1 - TOUCH_TOLERANCE_PCT / 100):
        buy_signals.append(f"回踩黄线{yellow:.2f}(偏离{price_to_yellow_pct:+.1f}%)")

    # 2. 超跌买：跌破双线且J<20
    if not is_above_white and close < yellow and j_val < J_OVERSOLD:
        buy_signals.append(f"超跌买(J={j_val:.0f}<{J_OVERSOLD},跌破双线)")

    # ── 卖出信号 ──
    # 股价在白线上方且J>80
    if is_above_white and j_val > J_OVERBOUGHT:
        sell_signals.append(f"白线{white:.2f}上方+J={j_val:.0f}>{J_OVERBOUGHT}")

    # 价格远离白线上方（正乖离过大）
    if price_to_white_pct > 3.0:
        sell_signals.append(f"正乖离过大(偏离白线{price_to_white_pct:+.1f}%)")

    # ── 风控提示 ──
    # 正T买入后，跌破黄线1.5%需止损
    stop_price = yellow * (1 - STOP_LOSS_PCT / 100)
    risk_alerts.append(f"跌破黄线{STOP_LOSS_PCT}%({stop_price:.2f})严格短线止损")

    return {
        "mode": "正T",
        "yellow_slope": slope_label,
        "buy_signal": " | ".join(buy_signals) if buy_signals else f"等待回踩黄线{yellow:.2f}或超跌J<{J_OVERSOLD}",
        "sell_signal": " | ".join(sell_signals) if sell_signals else f"等待反弹至白线{white:.2f}上方且J>{J_OVERBOUGHT}",
        "risk_alert": "; ".join(risk_alerts),
        "amplitude": round(amplitude, 2),
    }


def _analyze_short_t(
    close, white, yellow, j_val, amplitude,
    price_to_yellow_pct, price_to_white_pct,
    is_above_white, is_above_yellow, is_near_yellow,
    is_shrink_volume, vol_ratio,
    slope_label, slope_pct,
) -> Dict:
    """倒T模式分析（黄线向下/走平，先卖后买）"""
    buy_signals = []
    sell_signals = []
    risk_alerts = []

    # ── 卖出信号（减仓T）──
    # 1. 触黄线卖：从下方反弹触及黄线阻力
    if is_near_yellow and not is_above_yellow:
        sell_signals.append(f"触黄线阻力{yellow:.2f}(偏离{price_to_yellow_pct:+.1f}%)")

    # 2. 破白线卖：收盘价跌破白线
    if not is_above_white and close < white:
        sell_signals.append(f"跌破白线{white:.2f}")

    # 3. 超买卖：J>100且远离白线上方
    if j_val > J_EXTREME_OVERBOUGHT and is_above_white:
        sell_signals.append(f"超买卖(J={j_val:.0f}>{J_EXTREME_OVERBOUGHT})")

    # 价格远离白线上方（正乖离过大，无重大利好冲高）
    if price_to_white_pct > 3.0 and is_above_white:
        sell_signals.append(f"正乖离过大(偏离白线{price_to_white_pct:+.1f}%)")

    # ── 买回信号（补仓T）──
    # 1. 回落至白线下方企稳
    if not is_above_white:
        buy_signals.append(f"回落至白线{white:.2f}下方")

    # 2. 黄线附近缩量止跌
    if is_near_yellow and is_shrink_volume:
        buy_signals.append(f"黄线{yellow:.2f}附近缩量止跌(量比{vol_ratio:.1f})")

    # ── 风控提示 ──
    # 倒T卖出后，突破卖出价1.5%需买回（防卖飞）
    sell_ref = yellow if is_near_yellow else white
    buyback_price = sell_ref * (1 + STOP_LOSS_PCT / 100)
    risk_alerts.append(f"突破{sell_ref:.2f}的{STOP_LOSS_PCT}%({buyback_price:.2f})立即买回防卖飞")

    return {
        "mode": "倒T",
        "yellow_slope": slope_label,
        "buy_signal": " | ".join(buy_signals) if buy_signals else f"等待回落至白线{white:.2f}下方或黄线{yellow:.2f}缩量",
        "sell_signal": " | ".join(sell_signals) if sell_signals else f"等待触黄线{yellow:.2f}或跌破白线{white:.2f}",
        "risk_alert": "; ".join(risk_alerts),
        "amplitude": round(amplitude, 2),
    }


def _get_slope_label(df: pd.DataFrame, signal_idx: int) -> str:
    """获取黄线斜率方向标签"""
    if signal_idx < SLOPE_LOOKBACK:
        return "走平"
    yellow = float(df.iloc[signal_idx].get("yellow_line", 0))
    prev_yellow = float(df.iloc[signal_idx - SLOPE_LOOKBACK].get("yellow_line", 0))
    if prev_yellow <= 0 or yellow <= 0:
        return "走平"
    slope_pct = (yellow - prev_yellow) / prev_yellow * 100
    if slope_pct > SLOPE_UP_THRESHOLD:
        return "向上"
    elif slope_pct < SLOPE_DOWN_THRESHOLD:
        return "向下"
    return "走平"


def _default_result() -> Dict:
    """默认结果（数据不足时）"""
    return {
        "mode": "观望",
        "yellow_slope": "走平",
        "buy_signal": None,
        "sell_signal": None,
        "risk_alert": "数据不足，无法分析",
        "amplitude": 0.0,
    }
