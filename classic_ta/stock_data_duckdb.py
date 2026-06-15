"""股票日线数据DuckDB缓存模块

将4855个CSV碎文件合并为单个DuckDB文件，大幅提升缓存恢复速度和存储效率。
- 列式存储+向量化计算，查询速度远超CSV
- 列式压缩后体积仅为CSV的1/5到1/10
- 单文件管理，GitHub Actions Cache恢复极快
- 自动检测DuckDB可用性，不可用时回退到CSV模式

缓存文件: results/stock_cache.duckdb
表结构: daily_data (ts_code, date, open, high, low, close, volume)
"""

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 缓存目录和文件
CACHE_DIR = Path(__file__).parent.parent / "results" / "stock_cache"
DUCKDB_PATH = Path(__file__).parent.parent / "results" / "stock_cache.duckdb"

# 数据起始日期
DEFAULT_START_DATE = "20240101"

# DuckDB可用性检测
_DUCKDB_AVAILABLE = None


def _is_duckdb_available():
    """检测DuckDB是否可用，结果缓存"""
    global _DUCKDB_AVAILABLE
    if _DUCKDB_AVAILABLE is None:
        try:
            import duckdb
            _DUCKDB_AVAILABLE = True
        except ImportError:
            _DUCKDB_AVAILABLE = False
            logger.warning("DuckDB未安装，回退到CSV缓存模式。建议: pip install duckdb")
    return _DUCKDB_AVAILABLE


def _get_duckdb_conn():
    """获取DuckDB连接（读写模式）"""
    import duckdb
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=False)
    # 确保表存在
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_data (
            ts_code VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
    """)
    # 创建索引（如果不存在）
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_code ON daily_data(ts_code)")
    except Exception:
        pass  # 索引可能已存在
    return conn


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗：写入DuckDB前严格清洗

    - 排除Volume=0的停牌日
    - 排除Close/High/Low/Open为NaN或0的行
    - 对前复权突变检测（单日涨跌幅>50%可能是复权异常）
    - fillna和dropna
    """
    if df is None or df.empty:
        return df

    # 确保列名标准化
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("open", "high", "low", "close", "volume"):
            col_map[col] = lower.capitalize()
    if col_map:
        df = df.rename(columns=col_map)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"数据清洗: 缺少必要列 {col}")
            return pd.DataFrame()

    # 排除Volume=0的停牌日
    df = df[df["Volume"] > 0]

    # 排除OHLC为NaN或0的行
    for col in ["Open", "High", "Low", "Close"]:
        df = df[pd.notna(df[col])]
        df = df[df[col] > 0]

    if df.empty:
        return df

    # 前复权突变检测：单日涨跌幅>50%可能是复权异常
    if len(df) > 1:
        pct_change = df["Close"].pct_change()
        anomaly_mask = pct_change.abs() > 0.5
        if anomaly_mask.any():
            anomaly_count = anomaly_mask.sum()
            if anomaly_count <= 3:
                # 少量异常行直接删除
                df = df[~anomaly_mask]
                logger.info(f"数据清洗: 移除{anomaly_count}行前复权异常数据")
            else:
                # 大量异常可能是正常的，仅记录警告
                logger.warning(f"数据清洗: 检测到{anomaly_count}行涨跌幅>50%，可能为复权异常")

    # fillna和dropna
    df = df.fillna(method="ffill")
    df = df.dropna(subset=required_cols)

    return df


def load_stock_cache(ts_code: str):
    """加载单只股票的缓存数据

    返回:
        pd.DataFrame (Date索引, Open/High/Low/Close/Volume列) 或 None
    """
    if _is_duckdb_available():
        return _load_from_duckdb(ts_code)
    else:
        return _load_from_csv(ts_code)


def _load_from_duckdb(ts_code: str):
    """从DuckDB加载单只股票数据"""
    try:
        conn = _get_duckdb_conn()
        result = conn.execute(
            "SELECT date, open, high, low, close, volume FROM daily_data "
            "WHERE ts_code = ? ORDER BY date",
            [ts_code]
        ).fetchdf()
        conn.close()

        if result is None or len(result) < 10:
            return None

        result["Date"] = pd.to_datetime(result["date"])
        result = result.set_index("Date")
        result = result[["open", "high", "low", "close", "volume"]]
        result.columns = ["Open", "High", "Low", "Close", "Volume"]
        return result
    except Exception as e:
        logger.warning(f"DuckDB加载失败 {ts_code}: {e}")
        return _load_from_csv(ts_code)


def _load_from_csv(ts_code: str):
    """从CSV加载单只股票数据（回退模式）"""
    path = CACHE_DIR / f"{ts_code}.csv"
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
    # 数据清洗
    df = _clean_dataframe(df)
    if df is None or df.empty:
        return

    if _is_duckdb_available():
        _save_to_duckdb(ts_code, df)
    else:
        _save_to_csv(ts_code, df)


def _save_to_duckdb(ts_code: str, df: pd.DataFrame):
    """保存到DuckDB（先删除旧数据再插入新数据）"""
    try:
        conn = _get_duckdb_conn()
        # 删除该股票的旧数据
        conn.execute("DELETE FROM daily_data WHERE ts_code = ?", [ts_code])

        # 准备插入数据
        insert_df = df.copy()
        insert_df["ts_code"] = ts_code
        insert_df["date"] = insert_df.index.date if hasattr(insert_df.index, "date") else insert_df.index

        # 确保列顺序正确
        insert_df = insert_df[["ts_code", "date", "Open", "High", "Low", "Close", "Volume"]]
        insert_df.columns = ["ts_code", "date", "open", "high", "low", "close", "volume"]

        conn.execute("INSERT INTO daily_data SELECT * FROM insert_df")
        conn.close()
    except Exception as e:
        logger.warning(f"DuckDB保存失败 {ts_code}: {e}，回退到CSV")
        _save_to_csv(ts_code, df)


def _save_to_csv(ts_code: str, df: pd.DataFrame):
    """保存到CSV（回退模式）"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_DIR / f"{ts_code}.csv")


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
      1. 加载本地缓存（DuckDB优先，CSV回退）
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
    stats = {"mode": "unknown", "count": 0}

    if _is_duckdb_available() and DUCKDB_PATH.exists():
        try:
            conn = _get_duckdb_conn()
            count = conn.execute("SELECT COUNT(DISTINCT ts_code) FROM daily_data").fetchone()[0]
            size_mb = round(DUCKDB_PATH.stat().st_size / 1024 / 1024, 1)
            conn.close()
            return {
                "mode": "duckdb",
                "count": count,
                "dir": str(DUCKDB_PATH),
                "size_mb": size_mb,
            }
        except Exception as e:
            logger.warning(f"DuckDB统计失败: {e}")

    # 回退到CSV统计
    if CACHE_DIR.exists():
        csv_files = list(CACHE_DIR.glob("*.csv"))
        total_size = sum(f.stat().st_size for f in csv_files)
        return {
            "mode": "csv",
            "count": len(csv_files),
            "dir": str(CACHE_DIR),
            "size_mb": round(total_size / 1024 / 1024, 1),
        }

    return stats


def migrate_csv_to_duckdb():
    """一键将现有4855个CSV文件迁移到DuckDB

    返回:
        dict: {"migrated": 成功数, "failed": 失败数, "total": 总数}
    """
    if not _is_duckdb_available():
        logger.error("DuckDB未安装，无法迁移。请: pip install duckdb")
        return {"migrated": 0, "failed": 0, "total": 0}

    if not CACHE_DIR.exists():
        logger.info("CSV缓存目录不存在，无需迁移")
        return {"migrated": 0, "failed": 0, "total": 0}

    csv_files = list(CACHE_DIR.glob("*.csv"))
    if not csv_files:
        logger.info("没有CSV文件需要迁移")
        return {"migrated": 0, "failed": 0, "total": 0}

    logger.info(f"开始迁移 {len(csv_files)} 个CSV文件到DuckDB...")

    conn = _get_duckdb_conn()
    migrated = 0
    failed = 0

    for i, csv_path in enumerate(csv_files):
        ts_code = csv_path.stem  # 文件名即ts_code
        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            if df is None or len(df) < 10:
                failed += 1
                continue

            # 数据清洗
            df = _clean_dataframe(df)
            if df is None or df.empty:
                failed += 1
                continue

            # 删除旧数据
            conn.execute("DELETE FROM daily_data WHERE ts_code = ?", [ts_code])

            # 准备插入
            insert_df = df.copy()
            insert_df["ts_code"] = ts_code
            insert_df["date"] = insert_df.index.date if hasattr(insert_df.index, "date") else insert_df.index
            insert_df = insert_df[["ts_code", "date", "Open", "High", "Low", "Close", "Volume"]]
            insert_df.columns = ["ts_code", "date", "open", "high", "low", "close", "volume"]
            conn.execute("INSERT INTO daily_data SELECT * FROM insert_df")
            migrated += 1

            if (i + 1) % 500 == 0:
                logger.info(f"迁移进度: {i + 1}/{len(csv_files)} (成功:{migrated} 失败:{failed})")

        except Exception as e:
            logger.warning(f"迁移失败 {ts_code}: {e}")
            failed += 1

    conn.close()

    result = {"migrated": migrated, "failed": failed, "total": len(csv_files)}
    logger.info(f"迁移完成! 成功:{migrated} 失败:{failed} 总计:{len(csv_files)}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    stats = get_cache_stats()
    print(f"缓存模式: {stats.get('mode', 'unknown')}")
    print(f"缓存数量: {stats.get('count', 0)}只")
    print(f"缓存大小: {stats.get('size_mb', 0)}MB")
    print(f"缓存路径: {stats.get('dir', 'N/A')}")
