import numpy as np
import pandas as pd


def detect_reversal_signal(df):
    df = df.copy()

    body = (df["Close"] - df["Open"]).abs()
    full_range = df["High"] - df["Low"] + 1e-8
    upper_shadow = df["High"] - df[["Close", "Open"]].max(axis=1)
    lower_shadow = df[["Close", "Open"]].min(axis=1) - df["Low"]

    wyckoff_ok = df["wyckoff_phase"].isin(["吸筹", "震荡"])

    yellow_safe = df["yellow_line"].replace(0, np.nan)
    line_dist_ratio = (df["white_line"] - df["yellow_line"]).abs() / yellow_safe
    lines_close = (line_dist_ratio < 0.03).astype(int)
    entangle_count = lines_close.rolling(window=40, min_periods=1).sum()
    entangle_ok = entangle_count >= 25

    position_ok = wyckoff_ok & entangle_ok

    vol_surge_stagnant = (df["Volume"] > df["volume_ma"] * 2) & (
        (body / full_range < 0.3) | (lower_shadow > body * 1.5)
    )
    vol_surge_stagnant_recent = (
        vol_surge_stagnant.rolling(window=10, min_periods=1).max().astype(bool)
    )

    vol_surge_rise = (
        (df["Volume"] > df["Volume"].shift(1) * 1.5)
        & (df["daily_return"] > 1)
        & (upper_shadow < body * 0.3)
    )

    vpa_ok = vol_surge_stagnant_recent | vol_surge_rise

    is_bullish = df["Close"] > df["Open"]
    rise_ok = (df["daily_return"] >= 2) & (df["daily_return"] <= 9.5)
    body_full = (body / full_range) > 0.6
    no_long_upper = upper_shadow < body * 0.3

    prev_close_1 = df["Close"].shift(1)
    prev_open_1 = df["Open"].shift(1)
    prev_bearish_1 = prev_close_1 < prev_open_1
    engulf_1 = (df["Close"] > prev_open_1) & (df["Open"] < prev_close_1) & prev_bearish_1

    prev_close_2 = df["Close"].shift(2)
    prev_open_2 = df["Open"].shift(2)
    prev_bearish_2 = prev_close_2 < prev_open_2
    engulf_2 = (df["Close"] > prev_open_2) & (df["Open"] < prev_close_2) & prev_bearish_2

    engulf_ok = engulf_1 | engulf_2

    candle_ok = is_bullish & rise_ok & body_full & no_long_upper & engulf_ok

    j_recently_oversold = (df["J"].rolling(window=5, min_periods=1).min() < 20)
    j_turning_up = df["J"] > df["J"].shift(1)
    j_sentiment_ok = j_recently_oversold & j_turning_up

    df["reversal_signal"] = (position_ok & vpa_ok & candle_ok & j_sentiment_ok).fillna(False)

    return df


def detect_uptrend_signal(df):
    df = df.copy()

    body = (df["Close"] - df["Open"]).abs()
    full_range = df["High"] - df["Low"] + 1e-8
    upper_shadow = df["High"] - df[["Close", "Open"]].max(axis=1)

    breakout = (df["Close"] > df["resistance_level"]) & (
        df["Close"].shift(1) <= df["resistance_level"]
    )
    wyckoff_ok = breakout | (df["wyckoff_phase"] == "拉升")

    vpa_ok = (
        (df["daily_return"] > 0)
        & (df["Volume"] > df["Volume"].shift(1) * 1.5)
        & (df["Volume"] < df["Volume"].shift(1) * 2)
        & (df["Volume"] < df["volume_ma"] * 5)
    )

    is_bullish = df["Close"] > df["Open"]
    body_full = (body / full_range) > 0.5
    no_long_upper = upper_shadow < body * 0.2

    candle_ok = is_bullish & body_full & no_long_upper

    j_has_momentum = df["J"] > 30
    j_rising = df["J"] > df["J"].shift(1)
    j_momentum_ok = j_has_momentum & j_rising

    df["uptrend_signal"] = (wyckoff_ok & vpa_ok & candle_ok & j_momentum_ok).fillna(False)

    return df


def run_buy_signal_detection(df):
    df = detect_reversal_signal(df)
    df = detect_uptrend_signal(df)
    return df
