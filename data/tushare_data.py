
"""
Tushare 数据获取工具
用于为 TradingAgents 提供稳定的 A 股数据源
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

import tushare as ts
import pandas as pd

load_dotenv(Path(__file__).parent / ".env")

# 初始化 Tushare (避免写入本地文件)
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
pro = None
if TUSHARE_TOKEN:
    try:
        # 直接初始化pro接口，不调用ts.set_token()避免写权限问题
        pro = ts.pro_api(TUSHARE_TOKEN)
        print("✅ Tushare 初始化成功")
    except Exception as e:
        print(f"⚠️ Tushare 初始化失败: {e}")
        pro = None
else:
    pro = None
    print("⚠️ Tushare Token 未配置，请检查 .env 文件")


def convert_ticker_format(ticker: str) -> str:
    if ticker.endswith('.SS'):
        return ticker.replace('.SS', '.SH')
    return ticker


def get_stock_daily(ticker: str, start_date: str = '20240101', end_date: str = '20241231') -> Optional[pd.DataFrame]:
    if not pro:
        print("❌ Tushare 未初始化，请先配置 Token")
        return None

    ts_code = convert_ticker_format(ticker)

    try:
        print(f"📊 正在获取 {ticker} ({ts_code}) 数据...")
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        if df is not None and not df.empty:
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume',
                'trade_date': 'Date'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = df[old_col]
            
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            df.set_index('Date', inplace=True)
            
            print(f"✅ 成功获取 {len(df)} 条数据")
            print(f"日期范围: {df.index[0]} 到 {df.index[-1]}")
            return df
        else:
            print(f"⚠️ 未获取到 {ticker} 的数据")
            return None

    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_tushare_connection():
    print("=" * 60)
    print("Testing Tushare Connection...")
    print("=" * 60)

    if not TUSHARE_TOKEN:
        print("❌ TUSHARE_TOKEN 未设置")
        return False

    if not pro:
        print("❌ Tushare pro 未初始化")
        return False

    try:
        df = pro.daily(ts_code='600519.SH', start_date='20240101', end_date='20240131')
        if df is not None and not df.empty:
            print(f"✅ Tushare 连接成功！获取到 {len(df)} 条数据")
            print(df.head(3))
            return True
        else:
            print("⚠️ 连接成功，但未获取到数据")
            return True
    except Exception as e:
        print(f"❌ Tushare 连接失败: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            test_tushare_connection()
        elif sys.argv[1] == 'fetch' and len(sys.argv) >= 3:
            ticker = sys.argv[2]
            start = sys.argv[3] if len(sys.argv) > 3 else '20240101'
            end = sys.argv[4] if len(sys.argv) > 4 else '20241231'
            df = get_stock_daily(ticker, start, end)
            if df is not None:
                print(df.head())
    else:
        test_tushare_connection()

