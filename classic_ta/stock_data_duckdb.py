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
import queue
import time
import threading
from pathlib import Path

# Tushare 频率超限重试机制
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False


class TushareRateLimitError(Exception):
    """Tushare 接口频率超限异常"""
    pass

import pandas as pd

logger = logging.getLogger(__name__)

# DuckDB写入锁（防止多线程并发写入冲突）
_duckdb_write_lock = threading.Lock()

# 缓存目录和文件
CACHE_DIR = Path(__file__).parent.parent / "results" / "stock_cache"
DUCKDB_PATH = Path(__file__).parent.parent / "results" / "stock_cache.duckdb"

# 数据起始日期（5年完整历史，确保回测有足够数据）
DEFAULT_START_DATE = "20210101"

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


def _get_duckdb_read_conn():
    """获取DuckDB读取连接（使用 read_only=False 统一配置，避免与写连接配置冲突）

    注意：DuckDB 不允许同一数据库文件同时存在 read_only=True 和 read_only=False 的连接。
    因此所有连接统一使用 read_only=False，写操作由 _duckdb_write_lock 保护并发安全。
    """
    import duckdb
    if not DUCKDB_PATH.exists():
        return None
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=False)
    return conn


# ── 线程本地连接池（多线程扫描性能优化）──
# 避免每次查询都创建/关闭DuckDB连接，每线程复用同一个连接
_thread_local = threading.local()


def _get_thread_local_read_conn():
    """获取线程本地的DuckDB只读连接（复用，避免5000次连接创建/销毁）

    在ThreadPoolExecutor扫描场景下，每个worker线程持有一个长生命周期的只读连接，
    扫描期间复用，避免连接创建/销毁开销（从30分钟降至数秒）。
    """
    if not hasattr(_thread_local, 'read_conn') or _thread_local.read_conn is None:
        _thread_local.read_conn = _get_duckdb_read_conn()
    return _thread_local.read_conn


def close_thread_local_conns():
    """关闭当前线程的DuckDB只读连接（扫描结束后调用）"""
    if hasattr(_thread_local, 'read_conn') and _thread_local.read_conn is not None:
        try:
            _thread_local.read_conn.close()
        except Exception:
            pass
        _thread_local.read_conn = None


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
    df = df.ffill()
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
    """从DuckDB加载单只股票数据（只读模式，线程本地连接复用）

    使用线程本地连接池避免每次查询创建/销毁连接的开销。
    在多线程扫描场景下，性能从30分钟降至数秒。
    """
    try:
        conn = _get_thread_local_read_conn()
        if conn is None:
            return _load_from_csv(ts_code)
        result = conn.execute(
            "SELECT date, open, high, low, close, volume FROM daily_data "
            "WHERE ts_code = ? ORDER BY date",
            [ts_code]
        ).fetchdf()

        if result is None or len(result) < 10:
            return None

        result["Date"] = pd.to_datetime(result["date"])
        result = result.set_index("Date")
        result = result[["open", "high", "low", "close", "volume"]]
        result.columns = ["Open", "High", "Low", "Close", "Volume"]
        return result
    except Exception as e:
        logger.warning(f"DuckDB加载失败 {ts_code}: {e}")
        # 连接可能已损坏，重置线程本地连接
        close_thread_local_conns()
        return _load_from_csv(ts_code)


def _check_ex_rights_consistency(cached_last_close: float, new_data: pd.DataFrame, last_date) -> bool:
    """除权除息一致性校验

    通过overlap一天的数据，对比缓存最后一天的Close与增量数据同一天的Close，
    如果偏差超过1%则判定为除权除息导致的数据不一致。

    参数:
        cached_last_close: 缓存最后一天的Close价格
        new_data: 增量获取的数据（包含overlap天）
        last_date: 缓存最后日期

    返回:
        True=一致, False=不一致（可能发生了除权除息）
    """
    if new_data is None or len(new_data) == 0:
        return True

    # 在new_data中找到与cached最后日期相同的那天
    last_date_normalized = pd.Timestamp(last_date).normalize()
    overlap_rows = new_data[new_data.index.normalize() == last_date_normalized]

    if overlap_rows is None or len(overlap_rows) == 0:
        # 没有overlap数据（可能停牌），无法校验，放行
        return True

    overlap_close = float(overlap_rows.iloc[-1]["Close"])

    # 避免除零
    if cached_last_close == 0:
        return True

    diff_ratio = abs(cached_last_close - overlap_close) / cached_last_close

    if diff_ratio > 0.01:
        logger.info(
            f"除权校验不一致: cached_close={cached_last_close:.2f}, "
            f"overlap_close={overlap_close:.2f}, diff={diff_ratio:.4f}"
        )
        return False

    return True


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
    """保存到DuckDB（先删除旧数据再插入新数据，加锁防并发冲突）"""
    try:
        with _duckdb_write_lock:
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


def delete_stock_cache(ts_code: str):
    """删除单只股票的缓存数据（除权重建时使用）"""
    if _is_duckdb_available():
        with _duckdb_write_lock:
            conn = _get_duckdb_conn()
            conn.execute("DELETE FROM daily_data WHERE ts_code = ?", [ts_code])
            conn.close()
    # 同时清理CSV
    csv_path = CACHE_DIR / f"{ts_code}.csv"
    if csv_path.exists():
        csv_path.unlink()


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

    # 降级到 baostock（免费、无 token、原生前复权，自有 TCP 服务非爬虫）
    # 定位：akshare 爬虫被封/限流时的稳定补充层，海外可达性优于爬虫类
    try:
        from scripts.baostock_data_source import fetch_qfq_history as _bs_fetch
        df = _bs_fetch(ts_code, start_date=start_date, end_date=end_date, adjustflag="2")
        if df is not None and not df.empty:
            return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass

    # 降级到 tushare（手动前复权）—— 频率限制：强制每秒 ≤3 次
    try:
        import tushare as ts
        from dotenv import load_dotenv
        import os
        load_dotenv(Path(__file__).parent.parent / ".env", override=True)
        token = os.getenv("TUSHARE_TOKEN")
        if token is None:
            return None
        pro = ts.pro_api(token)

        # 强制降速：每次 Tushare 调用前 sleep 0.35s（≤3 次/秒）
        time.sleep(0.35)
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or len(df) == 0:
            return None

        # adj_factor 同样消耗 Tushare 配额，继续降速
        time.sleep(0.35)
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
    except Exception as e:
        # 捕获 Tushare 频率超限异常，退避等待后返回 None 触发上层重试
        err_msg = str(e)
        if "抱歉" in err_msg and ("每分钟" in err_msg or "频率" in err_msg or "访问" in err_msg):
            logger.warning(f"Tushare 频率超限 {ts_code}: {err_msg[:80]}，退避 60s")
            time.sleep(60)  # Tushare 按分钟限流，等满 1 分钟
        else:
            time.sleep(0.3)
        return None


def get_stock_data_cached(ts_code, min_rows=130):
    """带增量缓存的股票日线数据获取

    流程:
      1. 加载本地缓存（DuckDB优先，CSV回退）
      2. 如果缓存的最后日期已是今天 → 直接返回（零 API 调用）
      3. 否则获取 缓存最后日期 ~ 今天 的数据（重叠一天用于除权校验）
      4. 除权校验：对比overlap日Close，偏差>1%则全量重建
      5. 正常增量合并并保存

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
        first_date = cached.index[0].normalize()

        # 缓存已是最新（最后日期 >= 今天）且历史完整 → 直接返回
        if last_date >= today_ts and len(cached) >= min_rows:
            # 检查历史是否完整（起始日期不晚于DEFAULT_START_DATE）
            default_start = pd.Timestamp(DEFAULT_START_DATE)
            if first_date.normalize() <= default_start:
                return cached
            # 历史不完整，需要补全

        # 2. 增量获取: 从缓存最后日期开始（重叠一天，用于除权校验）
        overlap_start = last_date.strftime("%Y%m%d")  # 从缓存最后日期开始（重叠一天）
        new_data = _fetch_raw_stock_data(ts_code, start_date=overlap_start, end_date=end_date)

        if new_data is not None and len(new_data) > 0:
            # 除权校验：对比overlap日的Close是否一致
            cached_last_close = float(cached.iloc[-1]["Close"])
            if not _check_ex_rights_consistency(cached_last_close, new_data, last_date):
                logger.info(f"除权检测: {ts_code} 数据不一致，触发全量重建")
                delete_stock_cache(ts_code)
                full_df = _fetch_raw_stock_data(ts_code, DEFAULT_START_DATE, end_date)
                if full_df is not None and len(full_df) >= min_rows:
                    save_stock_cache(ts_code, full_df)
                    return full_df
                return None

            # 正常增量合并（排除overlap日的重复数据）
            new_only = new_data[new_data.index > last_date]
            if len(new_only) > 0:
                combined = pd.concat([cached, new_only])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                save_stock_cache(ts_code, combined)
                if len(combined) >= min_rows:
                    cached = combined
        else:
            # 无新数据但缓存足够
            if len(cached) >= min_rows:
                pass  # 继续检查历史完整性

        # 3. 检查历史完整性：如果缓存起始日期晚于DEFAULT_START_DATE，补全历史
        default_start = pd.Timestamp(DEFAULT_START_DATE)
        if cached.index[0].normalize() > default_start + pd.Timedelta(days=5):
            # 缓存缺少早期历史数据，补全
            hist_start = DEFAULT_START_DATE
            hist_end = (cached.index[0].normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")
            hist_data = _fetch_raw_stock_data(ts_code, start_date=hist_start, end_date=hist_end)
            if hist_data is not None and len(hist_data) > 0:
                combined = pd.concat([hist_data, cached])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                save_stock_cache(ts_code, combined)
                cached = combined

        if len(cached) >= min_rows:
            return cached

        return None

    # 3. 无缓存 → 完整获取
    df = _fetch_raw_stock_data(ts_code, start_date=DEFAULT_START_DATE, end_date=end_date)
    if df is not None and len(df) >= min_rows:
        save_stock_cache(ts_code, df)
        return df
    return None


def get_stock_data_readonly(ts_code, min_rows=130):
    """纯只读获取股票数据（供多线程扫描使用）

    不触发增量更新，不写入DuckDB。
    如果缓存不存在或数据不足，返回 None。
    """
    cached = load_stock_cache(ts_code)
    if cached is not None and len(cached) >= min_rows:
        return cached
    return None


_pending_write_queue = queue.Queue()


def batch_update_stocks(ts_codes: list):
    """批量更新多只股票的缓存数据（单线程调用）

    适用于扫描结束后一次性补全缓存缺失的数据。
    """
    if not ts_codes:
        return {"updated": 0, "failed": 0}

    logger.info(f"批量更新缓存: {len(ts_codes)}只股票")
    updated = 0
    failed = 0

    for ts_code in ts_codes:
        try:
            df = get_stock_data_cached(ts_code, min_rows=1)
            if df is not None:
                updated += 1
            else:
                failed += 1
            # 强制降速：每次请求后 sleep 0.35s，避免 Tushare 频率超限
            time.sleep(0.35)
        except Exception as e:
            logger.warning(f"批量更新失败 {ts_code}: {e}")
            failed += 1
            time.sleep(0.35)

    logger.info(f"批量更新完成: 成功={updated} 失败={failed}")
    return {"updated": updated, "failed": failed}


def get_cache_stats():
    """获取缓存统计信息"""
    stats = {"mode": "unknown", "count": 0}

    if _is_duckdb_available() and DUCKDB_PATH.exists():
        try:
            conn = _get_duckdb_read_conn()
            if conn is not None:
                count = conn.execute("SELECT COUNT(DISTINCT ts_code) FROM daily_data").fetchone()[0]
                conn.close()
                size_mb = round(DUCKDB_PATH.stat().st_size / 1024 / 1024, 1)
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
