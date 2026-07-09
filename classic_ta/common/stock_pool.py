"""Stock universe and intraday quote helpers."""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

COL_CODE = "\u4ee3\u7801"
COL_NAME = "\u540d\u79f0"
COL_INDUSTRY = "\u884c\u4e1a"
COL_LAST = "\u6700\u65b0\u4ef7"
COL_OPEN = "\u4eca\u5f00"
COL_HIGH = "\u6700\u9ad8"
COL_LOW = "\u6700\u4f4e"
COL_PREV_CLOSE = "\u6628\u6536"
COL_VOLUME = "\u6210\u4ea4\u91cf"
COL_AMOUNT = "\u6210\u4ea4\u989d"
COL_CHANGE_PCT = "\u6da8\u8dcc\u5e45"
COL_TURNOVER = "\u6362\u624b\u7387"
COL_VOL_RATIO = "\u91cf\u6bd4"


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _has_col(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns


def _row_get(row, column: str, default=None):
    if column in row.index:
        return row[column]
    return default


def _to_ts_code(code: str) -> str | None:
    code = str(code)
    if code.startswith("6"):
        return f"{code}.SH"
    if code.startswith("0") or code.startswith("3"):
        return f"{code}.SZ"
    return None


def _calc_intraday_qfq_ratio(df: pd.DataFrame, realtime_quote: dict, today: pd.Timestamp) -> float:
    """Map raw realtime prices into the cached forward-adjusted price space."""
    prev_close_raw = _to_float(realtime_quote.get("PrevClose"), 0.0)
    if prev_close_raw <= 0 or df is None or len(df) == 0:
        return 1.0

    ref_idx = -1
    if len(df) >= 2 and df.index[-1] == today:
        ref_idx = -2

    prev_close_adj = _to_float(df.iloc[ref_idx].get("Close"), 0.0)
    if prev_close_adj <= 0:
        return 1.0

    return prev_close_adj / prev_close_raw


def get_all_a_stocks():
    """Return all listed A-shares as ``(ts_code, name, industry)`` tuples."""
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        if df is not None and len(df) > 100:
            stocks = []
            for _, row in df.iterrows():
                code = str(_row_get(row, COL_CODE, ""))
                name = str(_row_get(row, COL_NAME, ""))
                if not code or name.startswith("ST") or name.startswith("*ST") or name.startswith("N"):
                    continue
                ts_code = _to_ts_code(code)
                if ts_code is None:
                    continue
                stocks.append((ts_code, name, str(_row_get(row, COL_INDUSTRY, ""))))
            if len(stocks) > 100:
                logger.info("akshare stock list fetched: %s", len(stocks))
                return stocks
    except Exception as exc:
        logger.warning("akshare stock list failed: %s", exc)

    try:
        import tushare as ts

        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            return _fallback_duckdb_stocks()

        pro = ts.pro_api(token)
        stock_basic = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,industry,list_date",
        )
        a_stocks = stock_basic[
            stock_basic["ts_code"].str.endswith(".SH") | stock_basic["ts_code"].str.endswith(".SZ")
        ]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("ST")]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("*ST")]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("N")]
        a_stocks = a_stocks[a_stocks["list_date"] < "20250101"]
        return [(row["ts_code"], row["name"], row.get("industry", "")) for _, row in a_stocks.iterrows()]
    except Exception as exc:
        logger.warning("tushare stock list failed: %s", exc)
        return _fallback_duckdb_stocks()


def _fallback_duckdb_stocks():
    """Fallback to the local DuckDB cache when online providers are unavailable."""
    try:
        import duckdb

        from classic_ta.stock_data_duckdb import DUCKDB_PATH

        if not DUCKDB_PATH.exists():
            logger.warning("DuckDB cache file does not exist")
            return []

        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        result = conn.execute(
            "SELECT DISTINCT ts_code FROM daily_data "
            "WHERE ts_code LIKE '%.SH' OR ts_code LIKE '%.SZ' "
            "ORDER BY ts_code"
        ).fetchall()
        conn.close()

        stocks = []
        for (ts_code,) in result:
            code = ts_code.split(".")[0]
            if code.startswith("8") or code.startswith("9"):
                continue
            if _to_ts_code(code) is None:
                continue
            stocks.append((ts_code, code, ""))

        logger.info("DuckDB fallback stock list fetched: %s", len(stocks))
        print(f"  Using DuckDB stock cache fallback ({len(stocks)} symbols)", flush=True)
        return stocks
    except Exception as exc:
        logger.warning("DuckDB fallback failed: %s", exc)
        return []


def batch_prefilter_stocks():
    """Prefilter the spot universe using batch realtime quotes."""
    try:
        import akshare as ak

        print("  Fetching akshare spot quotes for prefilter...", flush=True)
        df = ak.stock_zh_a_spot_em()
        if df is None or len(df) == 0:
            print("  akshare returned empty spot data", flush=True)
            return None
        print(f"  Retrieved {len(df)} spot rows", flush=True)

        df = df[~df[COL_NAME].str.startswith("ST", na=False)]
        df = df[~df[COL_NAME].str.startswith("*ST", na=False)]
        df = df[~df[COL_NAME].str.startswith("N", na=False)]
        df = df[~df[COL_NAME].str.contains("\u9000", na=False)]
        if _has_col(df, COL_VOLUME):
            df = df[df[COL_VOLUME] > 0]
        df = df[~df[COL_CODE].str.startswith("8", na=False)]
        df = df[~df[COL_CODE].str.startswith("9", na=False)]

        df["ts_code"] = df[COL_CODE].apply(_to_ts_code)
        df = df[df["ts_code"].notna()]

        if _has_col(df, COL_LAST):
            df = df[df[COL_LAST] >= 3]

        logger.info("Batch prefilter passed: %s", len(df))
        print(f"  Prefilter complete: {len(df)} symbols", flush=True)
        return df
    except Exception as exc:
        logger.warning("Batch prefilter failed: %s", exc)
        print(f"  Prefilter failed: {exc}", flush=True)
        return None


def get_realtime_quotes():
    """Return realtime quotes keyed by ts_code."""
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        if df is None or len(df) == 0:
            return {}

        quotes = {}
        for _, row in df.iterrows():
            code = str(_row_get(row, COL_CODE, ""))
            ts_code = _to_ts_code(code)
            if ts_code is None:
                continue

            close = _to_float(_row_get(row, COL_LAST, 0))
            if close <= 0:
                continue

            quotes[ts_code] = {
                "Open": _to_float(_row_get(row, COL_OPEN, 0)),
                "High": _to_float(_row_get(row, COL_HIGH, 0)),
                "Low": _to_float(_row_get(row, COL_LOW, 0)),
                "Close": close,
                "PrevClose": _to_float(_row_get(row, COL_PREV_CLOSE, 0)),
                "Volume": _to_float(_row_get(row, COL_VOLUME, 0)),
                "Amount": _to_float(_row_get(row, COL_AMOUNT, 0)),
                "change_pct": _to_float(_row_get(row, COL_CHANGE_PCT, 0)),
                "turnover": _to_float(_row_get(row, COL_TURNOVER, 0)),
                "vol_ratio_rt": _to_float(_row_get(row, COL_VOL_RATIO, 0)),
            }

        logger.info("Realtime quotes fetched: %s", len(quotes))
        return quotes
    except Exception as exc:
        logger.warning("Realtime quotes failed: %s", exc)
        return {}


def append_realtime_bar(df, realtime_quote, today_str=None):
    """Append or overwrite today's bar using qfq-mapped realtime prices."""
    if not realtime_quote or realtime_quote.get("Close", 0) <= 0:
        return df

    try:
        today = pd.Timestamp(today_str or datetime.now().strftime("%Y-%m-%d"))
        ratio = _calc_intraday_qfq_ratio(df, realtime_quote, today)
        open_adj = realtime_quote["Open"] * ratio
        high_adj = realtime_quote["High"] * ratio
        low_adj = realtime_quote["Low"] * ratio
        close_adj = realtime_quote["Close"] * ratio

        if len(df) > 0 and df.index[-1] == today:
            df.loc[today, "Open"] = open_adj
            df.loc[today, "High"] = high_adj
            df.loc[today, "Low"] = low_adj
            df.loc[today, "Close"] = close_adj
            df.loc[today, "Volume"] = realtime_quote["Volume"]
            df.loc[today, "Amount"] = realtime_quote["Amount"]
            return df

        new_row = pd.Series(
            {
                "Open": open_adj,
                "High": high_adj,
                "Low": low_adj,
                "Close": close_adj,
                "Volume": realtime_quote["Volume"],
                "Amount": realtime_quote["Amount"],
            },
            name=today,
        )
        return pd.concat([df, new_row.to_frame().T])
    except Exception:
        return df
