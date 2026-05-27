"""
Mootdx 数据获取工具
使用通达信数据源获取A股和ETF数据
"""
import sys
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# 设置mootdx配置目录到项目目录下，避免权限问题
os.environ['MOOTDX_HOME'] = str(Path(__file__).parent / '.mootdx')

try:
    from mootdx.quotes import Quotes
except ImportError:
    print("⚠️ mootdx 未安装，请运行: pip install mootdx")
    sys.exit(1)


def get_stock_daily_mootdx(
    ts_code: str,
    start_date: str = "20210101",
    end_date: str = "20260516"
) -> Optional[pd.DataFrame]:
    """
    使用mootdx获取股票/ETF日线数据

    Args:
        ts_code: 股票/ETF代码 (如 600036.SH, 510300.SH)
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)

    Returns:
        DataFrame: 包含 OHLCV 数据
    """
    try:
        # 解析股票代码
        symbol = ts_code.split(".")[0]
        
        print(f"📊 正在使用mootdx获取 {ts_code} 数据...")
        
        # 创建客户端
        client = Quotes.factory(market='std', multithread=True, heartbeat=True)
        
        # 获取日线数据 (frequency=9 表示日线，offset表示获取的条数)
        # 先获取较多的数据，然后筛选日期
        df = client.bars(symbol=symbol, frequency=9, offset=2000)
        
        if df is None or df.empty:
            print(f"⚠️ 未获取到 {ts_code} 的数据")
            return None
        
        # 重命名列以匹配回测系统需要的格式
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'datetime': 'Date'
        }
        
        # 重命名存在的列
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # 转换日期格式并设为索引
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 按日期排序
        df = df.sort_index()
        
        # 筛选日期范围
        if start_date:
            start_dt = pd.to_datetime(start_date, format='%Y%m%d')
            df = df[df.index >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date, format='%Y%m%d')
            df = df[df.index <= end_dt]
        
        if df.empty:
            print(f"⚠️ 筛选后 {ts_code} 数据为空")
            return None
        
        print(f"✅ 成功获取 {len(df)} 条数据")
        print(f"日期范围: {df.index[0].strftime('%Y-%m-%d')} 到 {df.index[-1].strftime('%Y-%m-%d')}")
        return df
        
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试 mootdx 数据获取")
    print("=" * 60)
    
    # 测试获取一只股票
    test_codes = [
        "600036.SH",  # 招商银行
        "510300.SH",  # 沪深300ETF
        "600570.SH",  # 恒生电子
    ]
    
    for code in test_codes:
        print(f"\n测试 {code}...")
        df = get_stock_daily_mootdx(code, "20210101", "20241231")
        if df is not None:
            print(df.head())
