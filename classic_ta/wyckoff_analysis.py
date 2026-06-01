import numpy as np
import pandas as pd


def identify_wyckoff_phase(df):
    yellow_line = df["yellow_line"]
    white_line = df["white_line"]
    close = df["Close"]
    volume = df["Volume"]

    window = 5
    indices = np.arange(window)
    slope = yellow_line.rolling(window).apply(
        lambda x: np.polyfit(indices, x, 1)[0], raw=True
    )
    slope_norm = slope / yellow_line * 100

    vol_ma20 = volume.rolling(20).mean()

    cond_markup = (close > white_line) & (white_line > yellow_line) & (slope_norm > 0.05)
    cond_distribution = (
        (close > yellow_line)
        & (white_line > yellow_line)
        & (slope_norm.abs() <= 0.05)
        & (volume < vol_ma20)
    )
    cond_markdown = (close < yellow_line) & (white_line < yellow_line) & (slope_norm < -0.05)
    cond_accumulation = (
        (close < yellow_line)
        & (white_line < yellow_line)
        & (slope_norm.abs() <= 0.05)
        & (volume > vol_ma20 * 0.8)
    )

    conditions = [
        cond_markup,
        cond_distribution,
        cond_markdown,
        cond_accumulation,
    ]
    choices = ["拉升", "派发", "下跌", "吸筹"]
    default = np.where(close >= yellow_line, "震荡", "震荡")

    df["wyckoff_phase"] = np.select(conditions, choices, default=default)
    return df


def calc_support_resistance(df):
    df["support_level"] = df["Low"].rolling(60).min()
    df["resistance_level"] = df["High"].rolling(60).max()
    return df


def detect_spring(df):
    prev_support = df["support_level"].shift(1)
    df["is_spring"] = (df["Low"] < prev_support) & (
        df["Close"] > prev_support
    )
    df["is_spring"] = df["is_spring"].fillna(False)
    return df


def run_wyckoff_analysis(df):
    df = identify_wyckoff_phase(df)
    df = calc_support_resistance(df)
    df = detect_spring(df)
    return df
