from __future__ import annotations

from datetime import datetime

import pandas as pd


def _first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _rename_if_present(df: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for target, candidates in mapping.items():
        source = _first_present(df, candidates)
        if source is not None:
            rename_map[source] = target
    return df.rename(columns=rename_map)


def get_efinance_realtime_quotes() -> pd.DataFrame:
    import efinance as ef

    df = ef.stock.get_realtime_quotes()
    if df is None or len(df) < 1000:
        raise RuntimeError("efinance realtime quotes returned too few rows")

    df = _rename_if_present(
        df.copy(),
        {
            "代码": ["股票代码", "代码"],
            "名称": ["股票名称", "名称"],
            "最新价": ["最新价"],
            "涨跌幅": ["涨跌幅"],
            "涨跌额": ["涨跌额"],
            "成交量": ["成交量"],
            "成交额": ["成交额"],
            "今开": ["今开"],
            "最高": ["最高"],
            "最低": ["最低"],
            "昨收": ["昨收"],
            "换手率": ["换手率"],
            "量比": ["量比"],
            "市盈率-动态": ["市盈率", "市盈率-动态"],
            "市净率": ["市净率"],
            "总市值": ["总市值"],
            "流通市值": ["流通市值"],
            "涨速": ["涨速"],
            "5分钟涨跌": ["5分钟涨跌"],
            "60日涨跌幅": ["60日涨跌幅"],
            "年初至今涨跌幅": ["年初至今涨跌幅"],
        },
    )
    return df


def get_efinance_quote_history(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    adjust: str = "",
) -> pd.DataFrame:
    import efinance as ef

    fqt = {"": 0, "qfq": 1, "hfq": 2}.get((adjust or "").lower(), 0)
    beg = start_date or "19700101"
    end = end_date or datetime.now().strftime("%Y%m%d")

    df = ef.stock.get_quote_history(symbol, beg=beg, end=end, klt=101, fqt=fqt)
    if df is None or len(df) == 0:
        raise RuntimeError(f"efinance quote history is empty for {symbol}")

    df = _rename_if_present(
        df.copy(),
        {
            "日期": ["日期"],
            "开盘": ["开盘"],
            "收盘": ["收盘"],
            "最高": ["最高"],
            "最低": ["最低"],
            "成交量": ["成交量"],
            "成交额": ["成交额"],
            "振幅": ["振幅"],
            "涨跌幅": ["涨跌幅"],
            "涨跌额": ["涨跌额"],
            "换手率": ["换手率"],
        },
    )
    return df


def get_latest_daily_bar_date_via_efinance(symbol: str = "000001") -> str:
    df = get_efinance_quote_history(symbol, start_date="20200101")
    if "日期" not in df.columns:
        raise RuntimeError("efinance quote history is missing 日期 column")
    latest = pd.to_datetime(df["日期"]).max()
    return latest.strftime("%Y%m%d")


def patch_akshare_with_efinance_fallback() -> None:
    import akshare as ak

    original_spot = ak.stock_zh_a_spot_em

    def patched_spot(*args, **kwargs):
        try:
            df = original_spot(*args, **kwargs)
            if df is not None and len(df) >= 1000:
                return df
        except Exception:
            pass
        return get_efinance_realtime_quotes()

    ak.stock_zh_a_spot_em = patched_spot

    if hasattr(ak, "stock_zh_a_hist"):
        original_hist = ak.stock_zh_a_hist

        def patched_hist(*args, **kwargs):
            try:
                df = original_hist(*args, **kwargs)
                if df is not None and len(df) > 0:
                    return df
            except Exception:
                pass

            symbol = kwargs.get("symbol")
            if symbol is None and args:
                symbol = args[0]
            if symbol is None:
                raise RuntimeError("stock_zh_a_hist fallback requires a symbol")

            period = kwargs.get("period", "daily")
            if period != "daily":
                raise RuntimeError("efinance fallback currently supports daily history only")

            start_date = kwargs.get("start_date")
            end_date = kwargs.get("end_date")
            adjust = kwargs.get("adjust", "")
            return get_efinance_quote_history(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )

        ak.stock_zh_a_hist = patched_hist
