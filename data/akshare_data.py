"""
AKShare 数据获取工具
使用东方财富等数据源获取A股和ETF数据
"""
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import akshare as ak
except ImportError:
    print("⚠️ akshare 未安装，请运行: pip install akshare")
    sys.exit(1)


def get_stock_daily_akshare(
    ts_code: str,
    start_date: str = "20210101",
    end_date: str = "20260516"
) -> Optional[pd.DataFrame]:
    """
    使用AKShare获取股票/ETF日线数据

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
        market = ts_code.split(".")[1] if "." in ts_code else "SH"
        
        print(f"📊 正在使用AKShare获取 {ts_code} 数据...")
        
        # 根据市场选择接口
        if market == "SH":
            # 沪市
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                   start_date=start_date, end_date=end_date, adjust="")
        elif market == "SZ":
            # 深市
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                   start_date=start_date, end_date=end_date, adjust="")
        else:
            # 默认尝试
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                   start_date=start_date, end_date=end_date, adjust="")
        
        if df is None or df.empty:
            print(f"⚠️ 未获取到 {ts_code} 的数据")
            return None
        
        # 重命名列以匹配回测系统需要的格式
        column_mapping = {
            '开盘': 'Open',
            '最高': 'High',
            '最低': 'Low',
            '收盘': 'Close',
            '成交量': 'Volume',
            '日期': 'Date'
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
    print("测试 AKShare 数据获取")
    print("=" * 60)
    
    # 测试获取一只股票
    test_codes = [
        "600036.SH",  # 招商银行
        "510300.SH",  # 沪深300ETF
        "600570.SH",  # 恒生电子
    ]
    
    for code in test_codes:
        print(f"\n测试 {code}...")
        df = get_stock_daily_akshare(code, "20210101", "20241231")
        if df is not None:
            print(df.head())
