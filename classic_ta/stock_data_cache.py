"""股票日线数据增量缓存模块

缓存每只股票的日线OHLCV数据到本地CSV文件，
后续运行只增量获取新交易日数据，大幅减少API调用次数。

缓存目录: results/stock_cache/
缓存格式: 每只股票一个CSV文件 (如 600519.SH.csv)
         索引为Date, 列为 Open/High/Low/Close/Volume

预估效果: 首次运行 ~15min → 后续运行 ~1-2min
"""
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 缓存目录
CACHE_DIR = Path(__file__).parent.parent / "results" / "stock_cache"

# 数据起始日期（与原始推送脚本一致）
DEFAULT_START_DATE = "20240101"


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ts_code: str) -> Path:
    """缓存文件路径: results/stock_cache/{ts_code}.csv"""
    return CACHE_DIR / f"{ts_code}.csv"


def load_stock_cache(ts_code: str):
    """加载单只股票的缓存数据

    返回:
        pd.DataFrame (Date索引, Open/High/Low/Close/Volume列) 或 None
    """
    path = _cache_path(ts_code)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        if len(df) < 10:
            return None
        return df
    except Exception:
        return None


def save_stock_cache(ts_code: str, df: pd.DataFrame):
    """保存单只股票数据到缓存"""
    _ensure_cache_dir()
    df.to_csv(_cache_path(ts_code))


def _fetch_raw_stock_data(ts_code, start_date=None, end_date=None):
    """从 akshare/tushare 获取原始日线数据（前复权）

    返回:
        pd.DataFrame (Date索引, Open/High/Low/Close/Volume列) 或 None
    """
    if start_date is None:
        start_date = DEFAULT_START_DATE
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y%m%d")

    # 优先 akshare（无限流）
    try:
        import akshare as ak
        symbol = ts_code.split(".")[0]
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start_date, end_date=end_date, adjust="qfq",
        )
        if df is not None and len(df) > 0:
            col_map = {"开盘": "Open", "最高": "High", "最低": "Low",
                       "收盘": "Close", "成交量": "Volume"}
            for old, new in col_map.items():
                if old in df.columns:
                    df[new] = df[old].astype(float)
            if "日期" in df.columns:
                df["Date"] = pd.to_datetime(df["日期"])
            df.set_index("Date", inplace=True)
            df = df.sort_index()
            df = df[df["Volume"] > 0]
            if not df.empty:
                return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass

    # 降级到 tushare（手动前复权）
    try:
        import tushare as ts
        from dotenv import load_dotenv
        import os
        load_dotenv(Path(__file__).parent.parent / ".env", override=True)
        token = os.getenv("TUSHARE_TOKEN")
        if token is None:
            return None
        pro = ts.pro_api(token)

        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or len(df) == 0:
            return None

        try:
            adj = pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if adj is not None and len(adj) > 0:
                adj = adj.sort_values("trade_date").reset_index(drop=True)
                latest_adj = adj["adj_factor"].iloc[-1]
                adj_ratio = adj["adj_factor"].astype(float) / float(latest_adj)
                df = df.sort_values("trade_date").reset_index(drop=True)
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col].astype(float) * adj_ratio
            else:
                df = df.sort_values("trade_date").reset_index(drop=True)
        except Exception:
            df = df.sort_values("trade_date").reset_index(drop=True)

        df["Date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df.set_index("Date", inplace=True)
        df["Open"] = df["open"].astype(float)
        df["High"] = df["high"].astype(float)
        df["Low"] = df["low"].astype(float)
        df["Close"] = df["close"].astype(float)
        df["Volume"] = df["vol"].astype(float)
        df = df[df["Volume"] > 0]
        if df.empty:
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        time.sleep(0.3)
        return None


def get_stock_data_cached(ts_code, min_rows=130):
    """带增量缓存的股票日线数据获取

    流程:
      1. 加载本地缓存 CSV
      2. 如果缓存的最后日期已是今天 → 直接返回（零 API 调用）
      3. 否则只获取 缓存最后日期+1 ~ 今天 的增量数据
      4. 合并并保存

    参数:
        ts_code: 股票代码 (如 600519.SH)
        min_rows: 最少需要的数据行数，默认130

    返回:
        pd.DataFrame (Date索引, Open/High/Low/Close/Volume列) 或 None
    """
    end_date = pd.Timestamp.now().strftime("%Y%m%d")
    today_ts = pd.Timestamp.now().normalize()

    # 1. 加载缓存
    cached = load_stock_cache(ts_code)

    if cached is not None and len(cached) > 0:
        last_date = cached.index[-1].normalize()

        # 缓存已是最新（最后日期 >= 今天）→ 直接返回
        if last_date >= today_ts and len(cached) >= min_rows:
            return cached

        # 2. 增量获取: 从缓存最后日期的下一天开始
        fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        new_data = _fetch_raw_stock_data(ts_code, start_date=fetch_start, end_date=end_date)

        if new_data is not None and len(new_data) > 0:
            # 合并缓存 + 新数据（去重保留最新）
            combined = pd.concat([cached, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            save_stock_cache(ts_code, combined)
            if len(combined) >= min_rows:
                return combined
        else:
            # 无新数据但缓存足够
            if len(cached) >= min_rows:
                return cached

        return None

    # 3. 无缓存 → 完整获取
    df = _fetch_raw_stock_data(ts_code, start_date=DEFAULT_START_DATE, end_date=end_date)
    if df is not None and len(df) >= min_rows:
        save_stock_cache(ts_code, df)
        return df
    return None


def get_cache_stats():
    """获取缓存统计信息"""
    if not CACHE_DIR.exists():
        return {"count": 0, "dir": str(CACHE_DIR)}

    csv_files = list(CACHE_DIR.glob("*.csv"))
    total_size = sum(f.stat().st_size for f in csv_files)
    return {
        "count": len(csv_files),
        "dir": str(CACHE_DIR),
        "size_mb": round(total_size / 1024 / 1024, 1),
    }
