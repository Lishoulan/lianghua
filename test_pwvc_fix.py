import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pip_libs"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import tushare as ts

load_dotenv()
ts.set_token(os.getenv('TUSHARE_TOKEN'))
pro = ts.pro_api()

print("=== 测试德展健康的新PWVC计算（最终版）===")

df = pro.daily(ts_code='000813.SZ', start_date='20260401', end_date='20260522')
df = df.sort_values('trade_date').reset_index(drop=True)
df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'vol': 'Volume'}, inplace=True)

df['white_line'] = df['Close'].ewm(span=10, adjust=False).mean()
df['white_line'] = df['white_line'].ewm(span=10, adjust=False).mean()
df['ma14'] = df['Close'].rolling(window=14).mean()
df['ma28'] = df['Close'].rolling(window=28).mean()
df['ma57'] = df['Close'].rolling(window=57).mean()
df['ma114'] = df['Close'].rolling(window=114).mean()
df['yellow_line'] = (df['ma14'] + df['ma28'] + df['ma57'] + df['ma114']) / 4
df['high_20'] = df['High'].rolling(window=20, min_periods=1).max()
df['low_20'] = df['Low'].rolling(window=20, min_periods=1).min()
df['price_position_20'] = (df['Close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)

print(df[['trade_date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10).to_string(index=False))

df['Vol_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
df['vol_ratio_20'] = df['Volume'] / df['Vol_MA20']

print()
print("vol_ratio_20 (最近10天):")
for i in range(-10, 0):
    print(f"  {df['trade_date'].iloc[i]}: {df['vol_ratio_20'].iloc[i]:.2f}")

print()
df['is_red_candle'] = (df['Close'] < df['Open']).astype(int)
high_5 = df['High'].rolling(window=5, min_periods=1).max()
low_5 = df['Low'].rolling(window=5, min_periods=1).min()

df['high_position_5'] = np.where(high_5 > low_5, 
                                 (df['High'] - low_5) / (high_5 - low_5), 
                                 0.5)
df['open_position_5'] = np.where(high_5 > low_5, 
                                 (df['Open'] - low_5) / (high_5 - low_5), 
                                 0.5)
df['top_position_5'] = np.maximum(df['high_position_5'], df['open_position_5'])

df['pwvc_day'] = df['vol_ratio_20'] * (df['top_position_5'] - 0.5) * df['is_red_candle']
df['pwvc'] = df['pwvc_day'].rolling(window=3, min_periods=1).max()

print("最近10天的详细指标：")
cols = ['trade_date', 'Open', 'High', 'Close', 'is_red_candle', 'vol_ratio_20', 'top_position_5', 'pwvc_day', 'pwvc']
print(df[cols].tail(10).to_string(index=False))

print()
print(f"最后一天({df['trade_date'].iloc[-1]})：")
print(f"  is_red_candle: {df['is_red_candle'].iloc[-1]}")
print(f"  vol_ratio_20: {df['vol_ratio_20'].iloc[-1]:.2f}")
print(f"  top_position_5: {df['top_position_5'].iloc[-1]:.2f}")
print(f"  pwvc_day: {df['pwvc_day'].iloc[-1]:.2f}")
print(f"  pwvc (3天max): {df['pwvc'].iloc[-1]:.2f}")
print()
print(f"是否否决(pwvc>1.5): {df['pwvc'].iloc[-1] > 1.5}")
print(f"是否否决(pwvc>0.8): {df['pwvc'].iloc[-1] > 0.8}")

