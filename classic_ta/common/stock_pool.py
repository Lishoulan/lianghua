"""Stock universe and intraday quote helpers."""

from __future__ import annotations

import logging
import os
import re
import time
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


def _normalize_spot_code(code: str) -> str:
    """统一现货代码格式：新浪带市场前缀（sh600000），东财是裸代码（600000）。"""
    code = str(code).strip()
    if len(code) == 8 and code[:2].isalpha():
        return code[2:]
    return code


def _get_spot_df_em():
    """全市场现货 —— 东财源（首选）。

    东财在境外/云主机 IP 上经常被限流或直接返回非 JSON 内容，
    此时自动降级到新浪源。
    """
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        if df is not None and len(df) > 100:
            return df
        logger.warning("东财现货返回异常（%s 行）", 0 if df is None else len(df))
    except Exception as exc:
        logger.warning("东财现货不可用: %s", exc)
    return None


def _get_spot_df_sina():
    """全市场现货 —— 新浪源（akshare 封装）。

    与东财源有两处差异，必须处理，否则数据会错：
      1. 代码带市场前缀（sh600000），要剥掉；
      2. 成交量单位是「股」，而东财和缓存日线用的都是「手」，要换算。

    注意：akshare 的这个封装打的是 vip.stock.finance.sina.com.cn:80（明文 HTTP），
    境外 IP 上会 read timeout，所以只作为第三顺位备源。
    """
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot()
        if df is None or len(df) == 0:
            logger.warning("新浪现货返回空")
            return None

        df = df.copy()
        if _has_col(df, COL_CODE):
            df[COL_CODE] = df[COL_CODE].map(_normalize_spot_code)
        if _has_col(df, COL_VOLUME):
            df[COL_VOLUME] = df[COL_VOLUME].astype(float) / 100.0

        logger.info("新浪现货可用: %s 只", len(df))
        return df
    except Exception as exc:
        logger.warning("新浪现货不可用: %s", exc)
        return None


TENCENT_URL = "https://qt.gtimg.cn/q={codes}"
TENCENT_BATCH = 60
TENCENT_PATTERN = re.compile(r'v_(\w+)="([^"]*)"')


def _a_share_symbols():
    """A股代码列表（腾讯格式 sh600000 / sz000001），优先用本地 DuckDB 缓存。"""
    symbols = []
    for ts_code, _name, _industry in _fallback_duckdb_stocks():
        code, _, market = str(ts_code).partition(".")
        if not code or not market:
            continue
        symbols.append(f"{market.lower()}{code}")
    return symbols


def _get_spot_df_tencent():
    """全市场现货 —— 腾讯源（HTTPS 443 直连，境外 IP 友好）。

    字段取自 qt.gtimg.cn 的标准行情格式：
      [1]名称 [2]代码 [3]最新价 [4]昨收 [5]今开 [6]成交量(手)
      [30]时间 [32]涨跌% [33]最高 [34]最低 [37]成交额(万) [38]换手率

    成交量单位是「手」，与东财和缓存日线一致，无需换算。
    """
    try:
        import requests

        symbols = _a_share_symbols()
        if not symbols:
            logger.warning("腾讯现货：本地缓存里没有 A 股代码列表")
            return None

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

        rows = []
        for start in range(0, len(symbols), TENCENT_BATCH):
            batch = symbols[start:start + TENCENT_BATCH]
            try:
                resp = session.get(TENCENT_URL.format(codes=",".join(batch)), timeout=15)
                resp.encoding = "gbk"
            except Exception as exc:
                logger.warning("腾讯现货批次失败(%s): %s", start, exc)
                continue

            for _sym, payload in TENCENT_PATTERN.findall(resp.text):
                fields = payload.split("~")
                if len(fields) < 40:
                    continue
                try:
                    close = float(fields[3])
                    prev_close = float(fields[4])
                    volume = float(fields[6])
                except (ValueError, IndexError):
                    continue
                if close <= 0 or prev_close <= 0:
                    continue
                amount = 0.0
                if len(fields) > 35 and "/" in fields[35]:
                    parts = fields[35].split("/")
                    if len(parts) > 2:
                        try:
                            amount = float(parts[2])
                        except ValueError:
                            amount = 0.0
                rows.append({
                    COL_CODE: fields[2],
                    COL_NAME: fields[1],
                    COL_LAST: close,
                    COL_OPEN: float(fields[5] or 0),
                    COL_HIGH: float(fields[33] or 0),
                    COL_LOW: float(fields[34] or 0),
                    COL_PREV_CLOSE: prev_close,
                    COL_VOLUME: volume,                 # 单位：手
                    COL_AMOUNT: amount,                 # 单位：元
                    COL_CHANGE_PCT: (close - prev_close) / prev_close * 100,
                    COL_TURNOVER: float(fields[38] or 0) if len(fields) > 38 else 0.0,
                    COL_VOL_RATIO: 0.0,                 # 腾讯不提供量比（下游未使用该字段）
                })

        if not rows:
            logger.warning("腾讯现货返回空")
            return None

        df = pd.DataFrame(rows)
        logger.info("腾讯现货可用: %s 只", len(df))
        return df
    except Exception as exc:
        logger.warning("腾讯现货不可用: %s", exc)
        return None


# 现货 DataFrame 缓存：一次推送流程里会多次取现货（预热/新鲜度/实时行情/预筛选），
# 不缓存的话每个源都要重试一遍，白白多花几分钟。
_SPOT_CACHE = {"df": None, "ts": 0.0}
SPOT_CACHE_TTL = 300.0


def _get_spot_df():
    """全市场现货 DataFrame：东财 → 腾讯 → 新浪 依次降级。"""
    df = _get_spot_df_em()
    if df is not None:
        return df

    print("  ⚠️ 东财实时行情不可用，降级到腾讯源", flush=True)
    df = _get_spot_df_tencent()
    if df is not None:
        print(f"  ✅ 腾讯实时行情可用 ({len(df)} 只)", flush=True)
        return df

    print("  ⚠️ 腾讯实时行情不可用，降级到新浪源", flush=True)
    df = _get_spot_df_sina()
    if df is not None:
        print(f"  ✅ 新浪实时行情可用 ({len(df)} 只)", flush=True)
    return df


def _get_spot_df_cached():
    """带 TTL 缓存的现货获取（同一进程内多次调用只拉一次）。"""
    now = time.time()
    cached = _SPOT_CACHE.get("df")
    if cached is not None and now - _SPOT_CACHE.get("ts", 0) < SPOT_CACHE_TTL:
        return cached

    df = _get_spot_df()
    if df is not None:
        _SPOT_CACHE["df"] = df
        _SPOT_CACHE["ts"] = now
    return df


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
    df = _get_spot_df()
    if df is not None:
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
            logger.info("现货股票列表: %s 只", len(stocks))
            return stocks

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
        print("  Fetching spot quotes for prefilter...", flush=True)
        df = _get_spot_df_cached()
        if df is None or len(df) == 0:
            print("  spot provider returned empty data", flush=True)
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
    """Return realtime quotes keyed by ts_code.

    数据源链路：东财 -> 新浪。CI 环境（GitHub runner）常被东财限流，
    没有降级源的话盘中模式会静默退化成盘后模式。
    """
    try:
        df = _get_spot_df_cached()
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
