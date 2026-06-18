"""
股票池获取与预筛选模块

从 v63_daily_push.py 提取的公共股票池逻辑。
支持 akshare（优先，无限流）和 tushare（降级）两种数据源。
"""
import os
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


def get_all_a_stocks():
    """获取全市场A股列表

    优先使用akshare（无限流），降级使用tushare。
    排除ST、*ST、N开头、北交所（8/9开头）股票。

    Returns:
        list[tuple]: [(ts_code, name, industry), ...]
    """
    # 尝试 akshare
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is not None and len(df) > 100:
            stocks = []
            for _, row in df.iterrows():
                code = str(row.get("代码", ""))
                name = str(row.get("名称", ""))
                if not code or name.startswith("ST") or name.startswith("*ST") or name.startswith("N"):
                    continue
                if code.startswith("6"):
                    ts_code = f"{code}.SH"
                elif code.startswith("0") or code.startswith("3"):
                    ts_code = f"{code}.SZ"
                else:
                    continue
                industry = str(row.get("行业", ""))
                stocks.append((ts_code, name, industry))
            if len(stocks) > 100:
                logger.info(f"akshare获取股票列表: {len(stocks)}只")
                return stocks
    except Exception as e:
        logger.warning(f"akshare获取股票列表失败: {e}")

    # 降级到tushare
    try:
        import tushare as ts
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            return []
        pro = ts.pro_api(token)
        stock_basic = pro.stock_basic(exchange="", list_status="L",
                                       fields="ts_code,symbol,name,industry,list_date")
        a_stocks = stock_basic[
            (stock_basic["ts_code"].str.endswith(".SH"))
            | (stock_basic["ts_code"].str.endswith(".SZ"))
        ]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("ST")]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("*ST")]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("N")]
        a_stocks = a_stocks[a_stocks["list_date"] < "20250101"]
        return [(row["ts_code"], row["name"], row.get("industry", "")) for _, row in a_stocks.iterrows()]
    except Exception as e:
        logger.warning(f"获取股票列表失败: {e}")
        return []


def batch_prefilter_stocks():
    """用akshare批量获取全市场实时行情，快速预筛选潜在信号股

    预筛选规则：
      - 排除ST、北交所、停牌
      - 涨跌幅 < 5%（排除已大涨）
      - 价格 3~100元
      - 换手率 >= 0.5%
      - 涨跌幅 > -5%（排除暴跌）

    Returns:
        pd.DataFrame or None: 预筛选后的行情数据（含ts_code列）
    """
    try:
        import akshare as ak
        print("  📡 正在获取akshare全市场行情（预筛选）...", flush=True)
        df = ak.stock_zh_a_spot_em()
        if df is None or len(df) == 0:
            print("  ⚠️ akshare返回空数据", flush=True)
            return None
        print(f"  📡 获取到{len(df)}只股票行情", flush=True)

        # 基本过滤
        df = df[~df["名称"].str.startswith("ST", na=False)]
        df = df[~df["名称"].str.startswith("*ST", na=False)]
        df = df[~df["名称"].str.startswith("N", na=False)]
        df = df[~df["名称"].str.contains("退", na=False)]
        if "成交量" in df.columns:
            df = df[df["成交量"] > 0]
        df = df[~df["代码"].str.startswith("8", na=False)]
        df = df[~df["代码"].str.startswith("9", na=False)]

        # 转换为ts_code格式
        def to_ts_code(code):
            code = str(code)
            if code.startswith("6"):
                return f"{code}.SH"
            elif code.startswith("0") or code.startswith("3"):
                return f"{code}.SZ"
            return None

        df["ts_code"] = df["代码"].apply(to_ts_code)
        df = df[df["ts_code"].notna()]

        # 快速预筛选
        if "涨跌幅" in df.columns:
            df = df[df["涨跌幅"] < 5]
        if "最新价" in df.columns:
            df = df[(df["最新价"] >= 3) & (df["最新价"] <= 100)]
        if "换手率" in df.columns:
            df = df[df["换手率"] >= 0.5]
        if "涨跌幅" in df.columns:
            df = df[df["涨跌幅"] > -5]

        logger.info(f"批量预筛选: {len(df)}只（排除ST/停牌/北交所/已大涨/极端价格/低换手）")
        print(f"  ✅ 预筛选完成: {len(df)}只通过初筛", flush=True)
        return df
    except Exception as e:
        logger.warning(f"批量预筛选失败: {e}")
        print(f"  ❌ 预筛选失败: {e}", flush=True)
        return None


def get_realtime_quotes():
    """获取全市场实时行情（akshare）

    Returns:
        dict: {ts_code: {Open, High, Low, Close, Volume, Amount, change_pct, turnover, vol_ratio_rt}}
    """
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is None or len(df) == 0:
            return {}

        quotes = {}
        for _, row in df.iterrows():
            code = str(row.get("代码", ""))
            if code.startswith("6"):
                ts_code = f"{code}.SH"
            elif code.startswith("0") or code.startswith("3"):
                ts_code = f"{code}.SZ"
            else:
                continue

            try:
                close = float(row.get("最新价", 0))
                if close <= 0:
                    continue
                quotes[ts_code] = {
                    "Open": float(row.get("今开", 0)),
                    "High": float(row.get("最高", 0)),
                    "Low": float(row.get("最低", 0)),
                    "Close": close,
                    "Volume": float(row.get("成交量", 0)),
                    "Amount": float(row.get("成交额", 0)),
                    "change_pct": float(row.get("涨跌幅", 0)),
                    "turnover": float(row.get("换手率", 0)),
                    "vol_ratio_rt": float(row.get("量比", 0)) if row.get("量比", 0) else 0,
                }
            except (ValueError, TypeError):
                continue

        logger.info(f"实时行情获取: {len(quotes)}只")
        return quotes
    except Exception as e:
        logger.warning(f"实时行情获取失败: {e}")
        return {}


def append_realtime_bar(df, realtime_quote, today_str=None):
    """将akshare实时行情拼接到日线数据末尾

    Args:
        df: 已计算指标的DataFrame
        realtime_quote: {Open, High, Low, Close, Volume, Amount, ...}
        today_str: 今日日期字符串

    Returns:
        pd.DataFrame: 拼接后的DataFrame
    """
    if not realtime_quote or realtime_quote.get("Close", 0) <= 0:
        return df

    try:
        today = pd.Timestamp(today_str or datetime.now().strftime("%Y-%m-%d"))

        if len(df) > 0 and df.index[-1] == today:
            df.iloc[-1]["Open"] = realtime_quote["Open"]
            df.iloc[-1]["High"] = realtime_quote["High"]
            df.iloc[-1]["Low"] = realtime_quote["Low"]
            df.iloc[-1]["Close"] = realtime_quote["Close"]
            df.iloc[-1]["Volume"] = realtime_quote["Volume"]
            df.iloc[-1]["Amount"] = realtime_quote["Amount"]
            return df

        new_row = pd.Series({
            "Open": realtime_quote["Open"],
            "High": realtime_quote["High"],
            "Low": realtime_quote["Low"],
            "Close": realtime_quote["Close"],
            "Volume": realtime_quote["Volume"],
            "Amount": realtime_quote["Amount"],
        }, name=today)

        df = pd.concat([df, new_row.to_frame().T])
        return df
    except Exception as e:
        return df
