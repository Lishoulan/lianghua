import numpy as np
import pandas as pd


def detect_hammer(df):
    body = (df["Close"] - df["Open"]).abs()
    upper_shadow = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower_shadow = df[["Open", "Close"]].min(axis=1) - df["Low"]
    full_range = df["High"] - df["Low"]

    cond_lower = lower_shadow >= body * 2
    cond_upper = upper_shadow <= body * 0.3
    cond_body = body / (full_range + 1e-8) < 0.4

    df["is_hammer"] = cond_lower & cond_upper & cond_body
    df["is_hammer"] = df["is_hammer"].fillna(False)
    return df


def detect_engulfing(df):
    today_body = df["Close"] - df["Open"]
    prev_body = df["Close"].shift(1) - df["Open"].shift(1)

    prev_close = df["Close"].shift(1)
    prev_open = df["Open"].shift(1)

    bullish = (
        (today_body > 0)
        & (prev_body < 0)
        & (df["Close"] > prev_open)
        & (df["Open"] < prev_close)
    )

    bearish = (
        (today_body < 0)
        & (prev_body > 0)
        & (df["Open"] > prev_close)
        & (df["Close"] < prev_open)
    )

    df["is_bullish_engulfing"] = bullish.fillna(False)
    df["is_bearish_engulfing"] = bearish.fillna(False)
    return df


def detect_morning_star(df):
    c2 = df["Close"].shift(2)
    o2 = df["Open"].shift(2)
    c1 = df["Close"].shift(1)
    o1 = df["Open"].shift(1)
    h1 = df["High"].shift(1)
    l1 = df["Low"].shift(1)

    body2 = (c2 - o2).abs()
    body1 = (c1 - o1).abs()
    range1 = h1 - l1 + 1e-8

    day1_bearish = c2 < o2
    day1_big_body = body2 > (df["High"].shift(2) - df["Low"].shift(2)) * 0.3
    day2_small = body1 / range1 < 0.2
    day2_close_near = (c1 - c2).abs() < body2 * 0.5
    day3_bullish = df["Close"] > df["Open"]
    day3_recover = df["Close"] > (o2 + c2) / 2

    df["is_morning_star"] = (
        day1_bearish & day1_big_body & day2_small & day2_close_near & day3_bullish & day3_recover
    )
    df["is_morning_star"] = df["is_morning_star"].fillna(False)
    return df


def detect_shooting_star(df):
    body = (df["Close"] - df["Open"]).abs()
    upper_shadow = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower_shadow = df[["Open", "Close"]].min(axis=1) - df["Low"]

    cond_upper = upper_shadow >= body * 2
    cond_lower = lower_shadow <= body * 0.3

    cond_trend = df["Close"] > df["white_line"]

    df["is_shooting_star"] = (cond_upper & cond_lower & cond_trend).fillna(False)
    return df


def run_candlestick_detection(df):
    df = detect_hammer(df)
    df = detect_engulfing(df)
    df = detect_morning_star(df)
    df = detect_shooting_star(df)
    return df
