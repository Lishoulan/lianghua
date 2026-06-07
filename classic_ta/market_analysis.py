import numpy as np
import pandas as pd


def calculate_trend_score(df):
    trend_score = 0

    white_above_yellow = df["white_line"] > df["yellow_line"]
    trend_score += np.where(white_above_yellow, 2, -2)

    yellow_rising = df["yellow_line"] > df["yellow_line"].shift(5)
    trend_score += np.where(yellow_rising, 1, -1)

    if "ma120" in df.columns:
        close_above_ma120 = df["Close"] > df["ma120"]
        trend_score += np.where(close_above_ma120, 2, -2)

    phase_scores = {"拉升": 3, "派发": 1, "震荡": 0, "下跌": -3, "吸筹": -1}
    trend_score += df["wyckoff_phase"].map(phase_scores).fillna(0)

    return trend_score


def calculate_sentiment_score(df):
    sentiment_score = 0

    j_val = df["J"]
    sentiment_score += np.where(j_val < 20, 2, 0)
    sentiment_score += np.where(j_val > 80, -2, 0)

    j_rising = df["J"] > df["J"].shift(1)
    sentiment_score += np.where(j_rising, 1, -1)

    k_above_d = df["K"] > df["D"]
    sentiment_score += np.where(k_above_d, 1, -1)

    return sentiment_score


def calculate_pattern_score(df):
    pattern_score = 0

    if "bullish_pattern_count" in df.columns:
        pattern_score += df["bullish_pattern_count"] * 2
    if "bearish_pattern_count" in df.columns:
        pattern_score -= df["bearish_pattern_count"] * 2

    if "bullish_vpa_count" in df.columns:
        pattern_score += df["bullish_vpa_count"] * 2
    if "bearish_vpa_count" in df.columns:
        pattern_score -= df["bearish_vpa_count"] * 2

    if "bullish_wyckoff_count" in df.columns:
        pattern_score += df["bullish_wyckoff_count"] * 2
    if "bearish_wyckoff_count" in df.columns:
        pattern_score -= df["bearish_wyckoff_count"] * 2

    return pattern_score


def calculate_volume_score(df):
    volume_score = 0

    vol_ratio = df["vol_ratio"]
    volume_score += np.where(vol_ratio > 1.5, 1, 0)
    volume_score += np.where(vol_ratio < 0.7, 1, 0)

    price_up = df["Close"] > df["Close"].shift(1)
    volume_up = df["Volume"] > df["Volume"].shift(1)
    volume_score += np.where(price_up & volume_up, 2, 0)
    volume_score += np.where((~price_up) & (~volume_up), 1, 0)

    return volume_score


def calculate_liquidity_score(df):
    liquidity_score = 0

    liquidity_score += np.where(df["Volume"] > df["volume_ma"], 1, 0)

    amplitude = (df["High"] - df["Low"]) / (df["Close"].shift(1) + 1e-8)
    liquidity_score += np.where(amplitude > 0.02, 1, 0)

    return liquidity_score


def calculate_market_score(df):
    df = df.copy()

    df["trend_score"] = calculate_trend_score(df)
    df["sentiment_score"] = calculate_sentiment_score(df)
    df["pattern_score"] = calculate_pattern_score(df)
    df["volume_score"] = calculate_volume_score(df)
    df["liquidity_score"] = calculate_liquidity_score(df)

    total_score = (
        df["trend_score"] * 0.3
        + df["sentiment_score"] * 0.2
        + df["pattern_score"] * 0.3
        + df["volume_score"] * 0.1
        + df["liquidity_score"] * 0.1
    )
    df["total_score"] = total_score

    def judge_market(score):
        if score < -5:
            return "极弱"
        elif score < 0:
            return "偏弱"
        elif score < 5:
            return "中性"
        else:
            return "偏强" if score < 10 else "极强"
    
    df["market_judgment"] = df["total_score"].apply(judge_market)

    return df


def get_bullish_patterns(latest):
    patterns = []
    if latest.get("is_hammer", False):
        patterns.append("锤子线")
    if latest.get("is_bullish_engulfing", False):
        patterns.append("看涨吞没")
    if latest.get("is_morning_star", False):
        patterns.append("启明星")
    if latest.get("is_piercing", False):
        patterns.append("刺透线")
    if latest.get("is_three_white_soldiers", False):
        patterns.append("白三兵")
    return patterns


def get_bearish_patterns(latest):
    patterns = []
    if latest.get("is_hanging_man", False):
        patterns.append("上吊线")
    if latest.get("is_bearish_engulfing", False):
        patterns.append("看跌吞没")
    if latest.get("is_evening_star", False):
        patterns.append("黄昏星")
    if latest.get("is_shooting_star", False):
        patterns.append("流星线")
    if latest.get("is_dark_cloud_cover", False):
        patterns.append("乌云盖顶")
    if latest.get("is_three_black_crows", False):
        patterns.append("黑三鸦")
    return patterns


def get_bullish_vpa(latest):
    vpa_list = []
    if latest.get("vol_surge_bottom_rejection", False):
        vpa_list.append("底部放量拒跌")
    if latest.get("shrink_volume_pullback", False):
        vpa_list.append("缩量回调")
    if latest.get("volume_divergence_bullish", False):
        vpa_list.append("量价底背离")
    if latest.get("no_supply_bar", False):
        vpa_list.append("无供应柱")
    if latest.get("test_bar", False):
        vpa_list.append("测试柱")
    return vpa_list


def get_bearish_vpa(latest):
    vpa_list = []
    if latest.get("vol_surge_stagnant", False):
        vpa_list.append("放量滞涨")
    if latest.get("high_volume_chase", False):
        vpa_list.append("放量追高")
    if latest.get("volume_divergence_bearish", False):
        vpa_list.append("量价顶背离")
    if latest.get("upthrust", False):
        vpa_list.append("上冲回落")
    return vpa_list


def get_wyckoff_signals(latest):
    bullish = []
    bearish = []
    if latest.get("is_spring", False):
        bullish.append("弹簧效应")
    if latest.get("is_spring_short", False):
        bullish.append("弹簧效应(短期)")
    if latest.get("is_secondary_test", False):
        bullish.append("二次测试")
    if latest.get("is_stop_volume", False):
        bullish.append("止损量")
    if latest.get("is_sign_of_strength", False):
        bullish.append("强势信号")
    if latest.get("is_upthrust", False):
        bearish.append("上冲下洗")
    if latest.get("is_upthrust_short", False):
        bearish.append("上冲下洗(短期)")
    if latest.get("is_ut_ad", False):
        bearish.append("UT诱多")
    return bullish, bearish


def generate_market_summary(df):
    if df is None or len(df) < 1:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    summary = {
        "price": float(latest["Close"]),
        "change": float(latest["Close"] - prev["Close"]),
        "change_pct": float((latest["Close"] - prev["Close"]) / prev["Close"] * 100),
        "wyckoff_phase": latest["wyckoff_phase"],
        "white_line": float(latest["white_line"]),
        "yellow_line": float(latest["yellow_line"]),
        "line_position": "白>黄" if latest["white_line"] > latest["yellow_line"] else "白<黄",
        "J": float(latest["J"]),
        "K": float(latest["K"]),
        "D": float(latest["D"]),
        "vol_ratio": float(latest.get("vol_ratio", 0)),
        "support": float(latest["support_level"]),
        "resistance": float(latest["resistance_level"]),
        "bullish_patterns": get_bullish_patterns(latest),
        "bearish_patterns": get_bearish_patterns(latest),
        "bullish_vpa": get_bullish_vpa(latest),
        "bearish_vpa": get_bearish_vpa(latest),
    }

    wyckoff_bull, wyckoff_bear = get_wyckoff_signals(latest)
    summary["bullish_wyckoff"] = wyckoff_bull
    summary["bearish_wyckoff"] = wyckoff_bear

    if "total_score" in df.columns:
        summary["total_score"] = float(latest["total_score"])
        summary["market_judgment"] = str(latest["market_judgment"])

    return summary
