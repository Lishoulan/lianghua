"""BaoStock 前复权数据源接入模块

BaoStock 是免费、开源的 A 股证券数据接口（自有 TCP 服务，非爬虫）：
- 官网: http://baostock.com
- 无需注册、无需 token、无配额限制
- 原生支持前复权（adjustflag="2"）/ 后复权（"3"）/ 不复权（"1"）
- 提供全市场股票列表 query_all_stock 和逐股 K 线 query_history_k_data_plus

在降级链路中的定位（akshare → baostock → tushare）：
- akshare 基于东财爬虫，2025 年后 IP 封禁/限流加剧，GitHub Actions 海外 IP 不稳定
- baostock 走自有 TCP 服务，不依赖网页爬虫，海外可达性优于爬虫类
- tushare 需要 token 且有配额限制，作为最终兜底

代码格式约定（YYYYMMDD <-> baostock 的 YYYY-MM-DD）:
- 项目内部统一用 YYYYMMDD（tushare 风格）
- baostock 接口要求 YYYY-MM-DD，本模块内部转换

ts_code 转换约定:
- 项目用 "600519.SH" / "000001.SZ"（tushare 风格）
- baostock 用 "sh.600519" / "sz.000001"
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

_BJT = ZoneInfo("Asia/Shanghai")


def _to_baostock_code(ts_code: str) -> str | None:
    """tushare 风格 ts_code -> baostock 风格 code

    "600519.SH" -> "sh.600519"
    "000001.SZ" -> "sz.000001"
    "688981.SH" -> "sh.688981"
    "300750.SZ" -> "sz.300750"
    """
    if not ts_code or "." not in ts_code:
        return None
    code, market = ts_code.split(".", 1)
    market = market.strip().upper()
    # 沪市: SH/SSH, 深市: SZ/SZSE
    if market in ("SH", "SSH"):
        return f"sh.{code}"
    if market in ("SZ", "SZSE"):
        return f"sz.{code}"
    return None


def _to_ts_code(bs_code: str) -> str:
    """baostock 风格 code -> tushare 风格 ts_code（反向转换）"""
    if not bs_code or "." not in bs_code:
        return bs_code
    market, code = bs_code.split(".", 1)
    market = market.strip().lower()
    if market == "sh":
        return f"{code}.SH"
    if market == "sz":
        return f"{code}.SZ"
    return bs_code


def _to_baostock_date(date_str: str) -> str:
    """YYYYMMDD -> YYYY-MM-DD"""
    if not date_str or len(date_str) != 8:
        return date_str
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


def _to_compact_date(date_str: str) -> str:
    """YYYY-MM-DD -> YYYYMMDD"""
    if not date_str or len(date_str) != 10:
        return date_str
    return date_str.replace("-", "")


def fetch_qfq_history(
    ts_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    adjustflag: str = "2",
) -> pd.DataFrame | None:
    """通过 BaoStock 获取单只股票前复权日线数据

    Args:
        ts_code: tushare 风格代码，如 "600519.SH"
        start_date: 起始日期 YYYYMMDD，默认 20210101
        end_date: 结束日期 YYYYMMDD，默认今天
        adjustflag: 复权标志，"2"=前复权(默认), "1"=不复权, "3"=后复权

    Returns:
        pd.DataFrame (Date 索引, Open/High/Low/Close/Volume 列) 或 None
        列名与 _fetch_raw_stock_data 输出格式完全一致，便于无缝替换。
    """
    try:
        import baostock as bs
    except ImportError:
        logger.debug("baostock 未安装，跳过该数据源")
        return None

    bs_code = _to_baostock_code(ts_code)
    if bs_code is None:
        logger.warning(f"baostock: 无法转换 ts_code={ts_code}")
        return None

    if start_date is None:
        start_date = "20210101"
    if end_date is None:
        end_date = datetime.now(_BJT).strftime("%Y%m%d")

    bs_start = _to_baostock_date(start_date)
    bs_end = _to_baostock_date(end_date)

    # BaoStock 需要显式 login/logout，使用局部连接避免影响全局状态
    lg = None
    try:
        lg = bs.login()
        if getattr(lg, "error_code", "0") != "0":
            logger.warning(f"baostock login 失败: {getattr(lg, 'error_msg', 'unknown')}")
            return None

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
            start_date=bs_start,
            end_date=bs_end,
            frequency="d",
            adjustflag=adjustflag,  # "2" = 前复权
        )

        if getattr(rs, "error_code", "0") != "0":
            logger.warning(f"baostock query 失败 {ts_code}: {getattr(rs, 'error_msg', 'unknown')}")
            return None

        data_list = []
        while (rs.error_code == "0") and rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return None

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 类型转换（baostock 返回值均为字符串）
        for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 过滤停牌日（volume=0 或 OHLC 为 0/NaN）
        df = df[df["volume"] > 0]
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[(df["open"] > 0) & (df["close"] > 0)]
        if df.empty:
            return None

        # 标准化为项目统一格式：Date 索引 + Open/High/Low/Close/Volume
        df["Date"] = pd.to_datetime(df["date"])
        df = df.set_index("Date").sort_index()
        result = pd.DataFrame({
            "Open": df["open"].astype(float),
            "High": df["high"].astype(float),
            "Low": df["low"].astype(float),
            "Close": df["close"].astype(float),
            "Volume": df["volume"].astype(float),
        })
        return result

    except Exception as e:
        logger.warning(f"baostock 获取 {ts_code} 异常: {e}")
        return None
    finally:
        try:
            if lg is not None:
                bs.logout()
        except Exception:
            pass


def query_all_stock_codes(trade_date: str | None = None) -> list[str]:
    """获取指定交易日全市场股票代码列表（tushare 风格）

    Args:
        trade_date: 交易日 YYYYMMDD，默认今天

    Returns:
        list[str]: 如 ["600519.SH", "000001.SZ", ...]，失败返回空列表
    """
    try:
        import baostock as bs
    except ImportError:
        return []

    if trade_date is None:
        trade_date = datetime.now(_BJT).strftime("%Y%m%d")
    bs_date = _to_baostock_date(trade_date)

    lg = None
    try:
        lg = bs.login()
        if getattr(lg, "error_code", "0") != "0":
            return []

        rs = bs.query_all_stock(day=bs_date)
        if getattr(rs, "error_code", "0") != "0":
            return []

        data_list = []
        while (rs.error_code == "0") and rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return []

        df = pd.DataFrame(data_list, columns=rs.fields)
        # 过滤: 仅保留 A 股股票（code 以 sh.6/sz.0/sz.3/sh.688 开头），排除指数/基金
        if "code" not in df.columns:
            return []

        def _is_a_share(bs_code: str) -> bool:
            if not bs_code or "." not in bs_code:
                return False
            market, num = bs_code.split(".", 1)
            if market == "sh":
                return num.startswith("6")  # 60xxxx / 688xxx
            if market == "sz":
                return num.startswith(("0", "3")) and not num.startswith("39")  # 00xxxx/30xxxx，排除 39 指数
            return False

        codes = df[df["code"].apply(_is_a_share)]["code"].tolist()
        return [_to_ts_code(c) for c in codes]

    except Exception as e:
        logger.warning(f"baostock query_all_stock 异常: {e}")
        return []
    finally:
        try:
            if lg is not None:
                bs.logout()
        except Exception:
            pass


def health_check() -> bool:
    """BaoStock 连通性健康检查（用于数据源预检）

    Returns:
        bool: True 表示可连通并获取数据
    """
    try:
        import baostock as bs
    except ImportError:
        return False

    lg = None
    try:
        lg = bs.login()
        if getattr(lg, "error_code", "0") != "0":
            return False
        # 用平安银行单股最近 5 天探测
        rs = bs.query_history_k_data_plus(
            "sz.000001",
            "date,close",
            start_date=_to_baostock_date(
                (datetime.now(_BJT)).strftime("%Y%m%d")
            ),
            end_date=_to_baostock_date(
                (datetime.now(_BJT)).strftime("%Y%m%d")
            ),
            frequency="d",
            adjustflag="2",
        )
        return getattr(rs, "error_code", "1") == "0"
    except Exception:
        return False
    finally:
        try:
            if lg is not None:
                bs.logout()
        except Exception:
            pass


if __name__ == "__main__":
    # 自测：获取贵州茅台最近 10 天前复权数据
    logging.basicConfig(level=logging.INFO)
    print("=== BaoStock 前复权数据源自测 ===")
    print(f"连通性: {health_check()}")
    df = fetch_qfq_history("600519.SH", start_date="20260601")
    if df is not None:
        print(f"贵州茅台前复权日线（最近5行）:\n{df.tail()}")
    else:
        print("获取失败")
