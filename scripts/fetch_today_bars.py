"""批量预热当日全市场日线数据

核心思路：利用 pro.daily(trade_date=today) 一次调用拉取全市场当日数据，
绕过逐股拉取的 tushare 配额限制（500次/天不够 4862 只）。

调用链路：
  run_scan_pipeline.py -> fetch_today_bars() -> prewarm_data() -> ensure_freshness()

数据格式对齐验证（2026-06-25）：
  DuckDB.close == pro.daily.close  ✅
  DuckDB.volume == pro.daily.vol   ✅ (单位一致，无需 ×100)
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
_BJT = ZoneInfo("Asia/Shanghai")


def _today_trade_date() -> str:
    """获取当前北京日期，格式 YYYYMMDD"""
    return datetime.now(_BJT).strftime("%Y%m%d")


def fetch_today_bars(trade_date: str | None = None) -> dict:
    """拉取当日（及回填缺失日期）全市场日线数据并增量合并到 DuckDB

    核心思路：利用 pro.daily(trade_date=YYYYMMDD) 一次调用拉取全市场当日数据。
    当 DuckDB 中部分股票数据滞后多日时，会自动回填从最早滞后日到当日的所有交易日数据。

    Args:
        trade_date: 指定交易日（YYYYMMDD），None 则用当天

    Returns:
        dict: {fetched, merged, skipped, latest_date, days_backfilled}
    """
    from classic_ta.stock_data_duckdb import (
        _get_duckdb_conn, _get_duckdb_read_conn, close_thread_local_conns,
        DUCKDB_PATH,
    )

    if trade_date is None:
        trade_date = _today_trade_date()

    print(f"[fetch_today_bars] 拉取 {trade_date} 全市场日线...", flush=True)

    # 1. 初始化 tushare
    try:
        import tushare as ts
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env", override=True)
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            raise RuntimeError("TUSHARE_TOKEN not set")
        pro = ts.pro_api(token)
    except Exception as e:
        print(f"[fetch_today_bars] Tushare 初始化失败: {e}", flush=True)
        return {"fetched": 0, "merged": 0, "skipped": 0, "latest_date": "", "days_backfilled": 0, "error": str(e)[:120]}

    # 2. 检查 DuckDB 缓存中每只股票的最新日期，确定需要回填的日期范围
    if not DUCKDB_PATH.exists():
        print(f"[fetch_today_bars] DuckDB 缓存不存在，仅拉取当日数据", flush=True)
        dates_to_fetch = [trade_date]
    else:
        read_conn = _get_duckdb_read_conn()
        if read_conn is None:
            dates_to_fetch = [trade_date]
        else:
            try:
                df_latest = read_conn.execute(
                    "SELECT ts_code, MAX(date) as latest_date FROM daily_data GROUP BY ts_code"
                ).fetch_df()
                read_conn.close()
                df_latest["latest_date"] = pd.to_datetime(df_latest["latest_date"])
                earliest_latest = df_latest["latest_date"].min()
                print(f"[fetch_today_bars] DuckDB 最早最新日期: {earliest_latest.strftime('%Y%m%d')}", flush=True)

                # 构造需要回填的日期列表（从 earliest_latest+1 到 trade_date）
                target_ts = pd.Timestamp(trade_date)
                dates_to_fetch = []
                cursor = earliest_latest + pd.Timedelta(days=1)
                while cursor <= target_ts:
                    if cursor.weekday() < 5:  # 周一~周五
                        dates_to_fetch.append(cursor.strftime("%Y%m%d"))
                    cursor += pd.Timedelta(days=1)

                if not dates_to_fetch:
                    dates_to_fetch = [trade_date]
                print(f"[fetch_today_bars] 需要拉取 {len(dates_to_fetch)} 个交易日: {dates_to_fetch[0]} ~ {dates_to_fetch[-1]}", flush=True)
            except Exception as e:
                read_conn.close()
                print(f"[fetch_today_bars] 读取 DuckDB 失败，仅拉当日: {e}", flush=True)
                dates_to_fetch = [trade_date]

    # 3. 逐日拉取全市场数据并合并
    total_fetched = 0
    total_merged = 0
    total_skipped = 0

    for dt in dates_to_fetch:
        try:
            time.sleep(0.35)  # 降速保护
            df_day = pro.daily(trade_date=dt)
        except Exception as e:
            err_msg = str(e)
            if "抱歉" in err_msg and ("频率" in err_msg or "每分钟" in err_msg):
                print(f"[fetch_today_bars] {dt} 频率超限，退避 60s", flush=True)
                time.sleep(60)
                continue
            print(f"[fetch_today_bars] {dt} 拉取失败: {err_msg[:80]}", flush=True)
            continue

        if df_day is None or len(df_day) == 0:
            print(f"[fetch_today_bars] {dt} 无数据（非交易日），跳过", flush=True)
            continue

        total_fetched += len(df_day)
        print(f"[fetch_today_bars] {dt}: 获取 {len(df_day)} 只", flush=True)

        # 合并到 DuckDB
        if not DUCKDB_PATH.exists():
            continue

        merged, skipped = _merge_day_to_duckdb(df_day, dt)
        total_merged += merged
        total_skipped += skipped

    close_thread_local_conns()

    result = {
        "fetched": total_fetched,
        "merged": total_merged,
        "skipped": total_skipped,
        "latest_date": trade_date,
        "days_backfilled": len(dates_to_fetch),
    }
    print(f"[fetch_today_bars] ✅ 完成: {result}", flush=True)
    return result


def _merge_day_to_duckdb(df_day: pd.DataFrame, trade_date: str) -> tuple[int, int]:
    """将单日全市场数据合并到 DuckDB，返回 (merged, skipped)"""
    from classic_ta.stock_data_duckdb import (
        _get_duckdb_conn, _get_duckdb_read_conn,
    )

    read_conn = _get_duckdb_read_conn()
    if read_conn is None:
        return 0, len(df_day)

    # 读取每只股票的最新日期
    try:
        df_latest = read_conn.execute(
            "SELECT ts_code, MAX(date) as latest_date FROM daily_data GROUP BY ts_code"
        ).fetch_df()
    except Exception as e:
        read_conn.close()
        print(f"[fetch_today_bars] 读取 DuckDB 最新日期失败: {e}", flush=True)
        return 0, len(df_day)

    trade_date_ts = pd.Timestamp(trade_date)
    df_latest["latest_date"] = pd.to_datetime(df_latest["latest_date"])
    need_update = set(df_latest[df_latest["latest_date"] < trade_date_ts]["ts_code"].tolist())

    # 前复权校验：对比 DuckDB close[-1] vs pro.daily pre_close
    rows_to_insert = []
    skipped_dividend = 0

    for ts_code in need_update:
        row_ts = df_day[df_day["ts_code"] == ts_code]
        if row_ts.empty:
            continue

        try:
            db_row = read_conn.execute(
                "SELECT close FROM daily_data WHERE ts_code = ? ORDER BY date DESC LIMIT 1",
                [ts_code]
            ).fetchone()
        except Exception:
            continue

        if db_row is None:
            continue

        db_close = float(db_row[0])
        ts_pre_close = float(row_ts["pre_close"].iloc[0])

        # 复权校验：如果 DuckDB 最后 close != tushare pre_close，说明有除权除息或数据缺口
        if abs(db_close - ts_pre_close) > 0.01:
            skipped_dividend += 1
            continue

        row = row_ts.iloc[0]
        rows_to_insert.append({
            "ts_code": ts_code,
            "date": pd.Timestamp(trade_date),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["vol"]),
        })

    read_conn.close()

    # 批量 INSERT
    merged = 0
    if rows_to_insert:
        write_conn = _get_duckdb_conn()
        try:
            df_insert = pd.DataFrame(rows_to_insert)
            write_conn.execute("BEGIN TRANSACTION")
            write_conn.execute(
                "INSERT INTO daily_data (ts_code, date, open, high, low, close, volume) "
                "SELECT ts_code, date, open, high, low, close, volume FROM df_insert"
            )
            write_conn.execute("COMMIT")
            merged = len(rows_to_insert)
        except Exception as e:
            write_conn.execute("ROLLBACK")
            print(f"[fetch_today_bars] {trade_date} DuckDB 写入失败: {e}", flush=True)
        finally:
            write_conn.close()

    return merged, skipped_dividend


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="批量预热当日全市场日线数据")
    parser.add_argument("--date", help="指定交易日 YYYYMMDD，默认当天")
    args = parser.parse_args()
    result = fetch_today_bars(args.date)
    print(f"\n结果: {result}")
