"""BaoStock 批量增量更新 DuckDB 缓存

用 BaoStock 前复权数据批量更新本地 DuckDB 缓存中滞后的股票。

设计要点：
1. 扫描 DuckDB 找出所有"最新日期 < 目标日期"的滞后股票
2. 用 BaoStock 逐股拉取前复权增量数据（adjustflag="2"）
3. 线程池并行加速（baostock 支持 session 复用，每线程独立 login/logout）
4. 前复权数据直接覆盖写入，无需 tushare 的 pre_close 复权校验（baostock 服务端已处理）
5. 增量合并：仅写入缓存最新日期之后的新数据，避免重复

用法:
    python scripts/baostock_batch_update.py                    # 更新所有滞后股票到最新
    python scripts/baostock_batch_update.py --workers 8        # 指定并发数
    python scripts/baostock_batch_update.py --target 20260626  # 指定目标日期
    python scripts/baostock_batch_update.py --full             # 全量重建（从20210101）
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.baostock_data_source import fetch_qfq_history, query_all_stock_codes

_BJT = ZoneInfo("Asia/Shanghai")
DUCKDB_PATH = REPO_ROOT / "results" / "stock_cache.duckdb"
DEFAULT_START_DATE = "20210101"


def _get_stale_stocks(target_date: str, full: bool = False) -> list[tuple[str, str]]:
    """找出 DuckDB 中所有滞后的股票

    Args:
        target_date: 目标最新日期 YYYYMMDD
        full: True 则返回所有股票（全量重建）

    Returns:
        list[(ts_code, cache_latest_date)] 需要更新的股票及其当前缓存最新日期
    """
    import duckdb

    if not DUCKDB_PATH.exists():
        print(f"[baostock_update] DuckDB 不存在: {DUCKDB_PATH}")
        return []

    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        if full:
            df = conn.execute(
                "SELECT ts_code, '' as latest_date FROM daily_data GROUP BY ts_code"
            ).fetch_df()
        else:
            df = conn.execute(
                "SELECT ts_code, MAX(date) as latest_date FROM daily_data GROUP BY ts_code"
            ).fetch_df()
    finally:
        conn.close()

    if df.empty:
        return []

    target_ts = pd.Timestamp(target_date)
    df["latest_date"] = pd.to_datetime(df["latest_date"], errors="coerce")

    if full:
        stale = df
    else:
        stale = df[df["latest_date"] < target_ts]

    result = []
    for _, row in stale.iterrows():
        latest = row["latest_date"]
        latest_str = latest.strftime("%Y%m%d") if pd.notna(latest) else DEFAULT_START_DATE
        result.append((row["ts_code"], latest_str))

    return result


def _update_one_stock(ts_code: str, cache_latest: str, target_date: str) -> dict:
    """用 baostock 更新单只股票的增量数据并写入 DuckDB

    Returns:
        dict: {ts_code, status, rows_added, error}
    """
    result = {"ts_code": ts_code, "status": "skip", "rows_added": 0, "error": ""}

    try:
        # 增量拉取：从缓存最新日期开始（重叠一天用于校验）
        start_date = cache_latest if cache_latest else DEFAULT_START_DATE
        df = fetch_qfq_history(ts_code, start_date=start_date, end_date=target_date, adjustflag="2")
        if df is None or df.empty:
            result["status"] = "no_data"
            return result

        import duckdb

        conn = duckdb.connect(str(DUCKDB_PATH), read_only=False)
        try:
            # 查询 DuckDB 中该股票已有数据的最大日期
            existing = conn.execute(
                "SELECT MAX(date) FROM daily_data WHERE ts_code = ?", [ts_code]
            ).fetchone()
            existing_max = existing[0] if existing and existing[0] else None

            # 过滤出比已有数据更新的行
            if existing_max is not None:
                existing_ts = pd.Timestamp(existing_max)
                df = df[df.index > existing_ts]

            if df.empty:
                result["status"] = "already_latest"
                return result

            # 准备插入数据
            rows = []
            for date, row in df.iterrows():
                rows.append({
                    "ts_code": ts_code,
                    "date": pd.Timestamp(date).date(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                })

            if not rows:
                result["status"] = "empty_after_filter"
                return result

            df_insert = pd.DataFrame(rows)
            conn.execute("BEGIN TRANSACTION")
            conn.execute(
                "INSERT INTO daily_data (ts_code, date, open, high, low, close, volume) "
                "SELECT ts_code, date, open, high, low, close, volume FROM df_insert"
            )
            conn.execute("COMMIT")

            result["status"] = "ok"
            result["rows_added"] = len(rows)
            return result

        except Exception as e:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            result["status"] = "write_error"
            result["error"] = str(e)[:120]
            return result
        finally:
            conn.close()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:120]
        return result


def batch_update(
    target_date: str | None = None,
    workers: int = 1,
    full: bool = False,
    max_retries: int = 2,
) -> dict:
    """批量用 baostock 更新 DuckDB 缓存到最新

    注意：baostock 的全局 session 非线程安全，多线程并发会导致连接冲突
    (WinError 10053/10038/用户未登录)。因此强制串行执行，通过重试机制
    应对偶发网络错误。

    Args:
        target_date: 目标日期 YYYYMMDD，默认今天
        workers: 保留参数兼容性，实际强制为1（baostock 不支持并发）
        full: True 则全量重建
        max_retries: 单只股票最大重试次数（应对网络抖动）

    Returns:
        dict: {total, updated, skipped, failed, rows_added}
    """
    if target_date is None:
        target_date = datetime.now(_BJT).strftime("%Y%m%d")

    if workers > 1:
        print(f"[baostock_update] 警告: baostock 全局 session 非线程安全，强制串行执行 (workers=1)", flush=True)

    print(f"[baostock_update] 目标日期: {target_date}, 全量重建: {full}, 重试: {max_retries}", flush=True)

    # 1. 找出滞后股票
    stale = _get_stale_stocks(target_date, full=full)
    total = len(stale)
    print(f"[baostock_update] 需要更新 {total} 只股票", flush=True)

    if total == 0:
        print("[baostock_update] 所有股票已是最新，无需更新", flush=True)
        return {"total": 0, "updated": 0, "skipped": 0, "failed": 0, "rows_added": 0}

    # 2. 串行更新（baostock 不支持并发）
    updated = 0
    skipped = 0
    failed = 0
    total_rows = 0
    failed_list = []
    start_time = time.time()

    for i, (ts_code, cache_latest) in enumerate(stale, 1):
        result = None
        for attempt in range(1, max_retries + 2):
            result = _update_one_stock(ts_code, cache_latest, target_date)
            if result["status"] in ("ok", "already_latest", "no_data", "empty_after_filter", "skip"):
                break
            # 网络错误重试，间隔递增
            if attempt <= max_retries:
                time.sleep(0.5 * attempt)

        status = result["status"]
        rows = result["rows_added"]

        if status == "ok":
            updated += 1
            total_rows += rows
        elif status in ("already_latest", "no_data", "empty_after_filter", "skip"):
            skipped += 1
        else:
            failed += 1
            failed_list.append((ts_code, result.get("error", status)))

        if i % 50 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"[baostock_update] 进度 {i}/{total} "
                f"({i*100//total}%) | 更新={updated} 跳过={skipped} 失败={failed} "
                f"新增行={total_rows} | {rate:.1f}只/s ETA={eta:.0f}s",
                flush=True,
            )

    elapsed = time.time() - start_time
    print(f"\n[baostock_update] 完成 | 耗时 {elapsed:.1f}s", flush=True)
    print(f"  总计={total} 更新={updated} 跳过={skipped} 失败={failed} 新增行={total_rows}", flush=True)

    if failed_list:
        print(f"  失败明细（前20条）:", flush=True)
        for ts_code, err in failed_list[:20]:
            print(f"    {ts_code}: {err}", flush=True)

    return {
        "total": total,
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "rows_added": total_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BaoStock 批量增量更新 DuckDB 缓存")
    parser.add_argument("--target", help="目标日期 YYYYMMDD，默认今天")
    parser.add_argument("--workers", type=int, default=6, help="并发线程数，默认6")
    parser.add_argument("--full", action="store_true", help="全量重建（从20210101）")
    args = parser.parse_args()

    result = batch_update(target_date=args.target, workers=args.workers, full=args.full)
    print(f"\n结果: {result}")


if __name__ == "__main__":
    main()
