from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import tushare as ts

from scripts.free_data_sources import (
    get_efinance_quote_history,
    get_efinance_realtime_quotes,
    get_latest_daily_bar_date_via_efinance,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = Path(os.getenv("DOCKER_ENV_FILE", REPO_ROOT / ".env.docker.local"))
DUCKDB_PATH = REPO_ROOT / "results" / "stock_cache.duckdb"
_BJT = ZoneInfo("Asia/Shanghai")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _require_token() -> str:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is empty in .env.docker.local")
    return token


def _open_trade_days() -> list[str]:
    token = _require_token()
    pro = ts.pro_api(token)
    now = datetime.now(_BJT).date()
    start = (now - timedelta(days=14)).strftime("%Y%m%d")
    end = now.strftime("%Y%m%d")

    # 优先用 trade_cal（1次/分钟配额）
    try:
        cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
        cal = cal[cal["is_open"] == 1]
        days = sorted(cal["cal_date"].tolist())
        if days:
            return days
    except Exception:
        pass

    # Fallback 1: 用 pro.daily(trade_date=today) 探测今日是否开盘
    today_str = now.strftime("%Y%m%d")
    try:
        time.sleep(0.35)
        df_today = pro.daily(trade_date=today_str)
        if df_today is not None and len(df_today) > 0:
            # 今日有数据，用最近工作日构造交易日历
            cursor = now - timedelta(days=14)
            days: list[str] = []
            while cursor <= now:
                if cursor.weekday() < 5:
                    days.append(cursor.strftime("%Y%m%d"))
                cursor += timedelta(days=1)
            return sorted(set(days))
    except Exception:
        pass

    # Fallback 2: 用缓存最新日期
    try:
        latest = _latest_cached_date()
    except Exception:
        latest = get_latest_daily_bar_date_via_efinance("000001")
    cursor = now - timedelta(days=14)
    days: list[str] = []
    while cursor <= now:
        if cursor.weekday() < 5:
            days.append(cursor.strftime("%Y%m%d"))
        cursor += timedelta(days=1)
    days.append(latest)
    return sorted(set(days))


def _expected_cache_date(mode: str) -> str:
    days = _open_trade_days()
    today = datetime.now(_BJT).strftime("%Y%m%d")
    latest_open = days[-1]

    if mode == "intraday":
        if latest_open == today and len(days) >= 2:
            return days[-2]
        return latest_open

    if datetime.now(_BJT).weekday() < 5:
        return max(today, latest_open)

    return latest_open


def _latest_cached_date() -> str:
    if not DUCKDB_PATH.exists():
        raise RuntimeError(f"DuckDB cache is missing: {DUCKDB_PATH}")

    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as conn:
        latest = conn.execute("select max(date) from daily_data").fetchone()[0]

    if latest is None:
        raise RuntimeError("DuckDB cache exists but daily_data is empty")

    return latest.strftime("%Y%m%d")


def _check_intraday_provider() -> None:
    errors: list[str] = []

    try:
        quotes = get_efinance_realtime_quotes()
        if quotes is not None and len(quotes) >= 1000:
            return
        errors.append("efinance returned too few rows")
    except Exception as exc:
        errors.append(f"efinance: {exc}")

    try:
        import akshare as ak

        quotes = ak.stock_zh_a_spot_em()
        if quotes is not None and len(quotes) >= 1000:
            return
        errors.append("akshare returned too few rows")
    except Exception as exc:
        errors.append(f"akshare: {exc}")

    raise RuntimeError("No free intraday quote provider is healthy: " + " | ".join(errors))


def _check_after_hours_provider(expected_date: str) -> None:
    errors: list[str] = []

    try:
        token = _require_token()
        pro = ts.pro_api(token)
        df = pro.daily_basic(
            trade_date=expected_date,
            fields="ts_code,trade_date,circ_mv,turnover_rate_f",
        )
        if df is not None and len(df) >= 1000:
            return
        errors.append(f"tushare daily_basic returned too few rows for {expected_date}")
    except Exception as exc:
        errors.append(f"tushare: {exc}")

    try:
        for symbol in ("000001", "600519", "000300"):
            hist = get_efinance_quote_history(symbol, start_date="20200101")
            latest = pd.to_datetime(hist["日期"]).max().strftime("%Y%m%d")
            if latest >= expected_date:
                return
        errors.append(f"efinance history is older than {expected_date}")
    except Exception as exc:
        errors.append(f"efinance: {exc}")

    raise RuntimeError("No free after-hours provider confirmed the latest daily bar: " + " | ".join(errors))


def ensure_freshness(mode: str) -> tuple[str, str]:
    _load_env_file(ENV_FILE)

    expected = _expected_cache_date(mode)
    latest = _latest_cached_date()

    if latest < expected:
        raise RuntimeError(f"Cache freshness check failed for {mode}: latest={latest}, expected>={expected}")

    now = datetime.now(_BJT)

    if mode == "intraday":
        if 9 <= now.hour < 15:
            _check_intraday_provider()
    else:
        _check_after_hours_provider(expected)

    return expected, latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-effort freshness checks before running the local scanner.")
    parser.add_argument("--mode", choices=["intraday", "after_hours"], required=True)
    args = parser.parse_args()

    expected, latest = ensure_freshness(args.mode)
    print(f"freshness_ok mode={args.mode} latest={latest} expected={expected}")


if __name__ == "__main__":
    main()
