import numpy as np
import pandas as pd


def calc_volume_features(df):
    df = df.copy()
    df["volume_ma"] = df["Volume"].rolling(window=20, min_periods=1).mean()
    df["vol_ratio"] = df["Volume"] / df["volume_ma"]
    low_20 = df["Low"].rolling(window=20, min_periods=1).min()
    high_20 = df["High"].rolling(window=20, min_periods=1).max()
    df["price_position_20"] = (df["Close"] - low_20) / (high_20 - low_20 + 1e-8)
    low_60 = df["Low"].rolling(window=60, min_periods=1).min()
    high_60 = df["High"].rolling(window=60, min_periods=1).max()
    df["price_position_60"] = (df["Close"] - low_60) / (high_60 - low_60 + 1e-8)
    df["daily_return"] = df["Close"].pct_change() * 100
    return df


def detect_vol_surge_stagnant(df):
    df = df.copy()
    df["vol_surge_stagnant"] = False
    mask = (
        (df["Volume"] > df["volume_ma"] * 2)
        & (df["daily_return"].abs() < 1.5)
        & (df["price_position_20"] > 0.7)
    )
    df.loc[mask, "vol_surge_stagnant"] = True
    return df


def detect_vol_surge_bottom_rejection(df):
    df = df.copy()
    df["vol_surge_bottom_rejection"] = False
    lower_shadow = df[["Open", "Close"]].min(axis=1) - df["Low"]
    body = (df["Close"] - df["Open"]).abs()
    mask = (
        (df["Volume"] > df["volume_ma"] * 2)
        & (lower_shadow > body)
        & (df["price_position_60"] < 0.3)
    )
    df.loc[mask, "vol_surge_bottom_rejection"] = True
    return df


def detect_shrink_volume_pullback(df):
    df = df.copy()
    df["shrink_volume_pullback"] = False
    mask = (df["Close"] < df["Close"].shift(1)) & (df["Volume"] < df["volume_ma"] * 0.7)
    df.loc[mask, "shrink_volume_pullback"] = True
    return df


def run_vpa_analysis(df):
    df = calc_volume_features(df)
    df = detect_vol_surge_stagnant(df)
    df = detect_vol_surge_bottom_rejection(df)
    df = detect_shrink_volume_pullback(df)
    return df
