"""
潜伏模型V6.4 每日实盘推送（精细动态评分版）
===========================
基于V6.4优化参数 + 精细动态评分 + 微信推送格式

优化参数:
  - 评分阈值 3→5, J值超卖 25→10, SOS窗口 12→8
  - 行业RS前30%→前20%, J极度超卖 0→3

精细动态评分:
  - OAMV牛市: 评分>=5 或 (评分=4且J<3且量比<0.6)
  - OAMV熊市: 评分>=6
  - 所有信号 J<10

推送内容：
  1. OAMV活跃市值择时状态
  2. 行业热度分析（行业动量排名、冷热分布、轮动信号）
  3. 潜伏买入信号（含行业过滤+精细动态评分）
  4. 持仓监控（4级退出：硬止损→吊灯止盈→Buy Climax→时间止损）

推送渠道：Server酱（微信）
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import tushare as ts
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 重试机制
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")

def _get_pro():
    """延迟初始化tushare pro，避免模块导入时token不存在报错"""
    if TUSHARE_TOKEN is None:
        raise ValueError("TUSHARE_TOKEN环境变量未设置")
    return ts.pro_api(TUSHARE_TOKEN)

SERVERCHAN_KEYS = [k.strip() for k in os.getenv("SERVERCHAN_KEY", "").split(",") if k.strip()]

RESULT_DIR = Path(__file__).parent.parent / "results" / "v63_daily"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# V6.4模型导入
from classic_ta.v60_ambush_model import IndicatorCalcBase, DEFAULT_PARAMS
from classic_ta.v61_ambush_model import V61_PARAMS
from classic_ta.v62_ambush_model import (
    compute_industry_momentum, build_industry_allow_matrix,
)
from classic_ta.v63_ambush_model import (
    Detect_AmbushSignal_V63, add_micro_confirm_indicators,
    calc_volatility_parity_shares, V63_PARAMS,
)
from classic_ta.v64_ambush_model import (
    add_inst_support_indicators, Detect_AmbushSignal_V64,
    V64_PARAMS,
)

# ══════════════════════════════════════════════════════════
#  优化参数（回测验证最优组合）
# ══════════════════════════════════════════════════════════
BEST_PARAMS = V64_PARAMS.copy()
BEST_PARAMS.update({
    # 评分阈值 3→5
    "entry_quality_min_score": 5,
    # J值超卖阈值 25→10
    "ambush_j_oversold": 10,
    # SOS后等待窗口 12→8
    "ambush_window": 8,
    # 行业RS前30%→前20%
    "industry_rs_top_pct": 0.20,
    # J值极度超卖 0→3
    "eq_j_extreme": 3,
})

# 精细动态评分参数
DYNAMIC_SCORE_PARAMS = {
    "bull_min_score": 5,           # 牛市最低评分
    "bull_score4_j_max": 3,        # 牛市评分=4时J值上限
    "bull_score4_vol_ratio_max": 0.60,  # 牛市评分=4时量比上限
    "bear_min_score": 6,           # 熊市最低评分
    "j_hard_cap": 10,             # 所有信号J值硬上限
}

# 股票日线数据增量缓存
from classic_ta.stock_data_duckdb import get_stock_data_cached, get_cache_stats


# ══════════════════════════════════════════════════════════
#  Server酱推送
# ══════════════════════════════════════════════════════════

def send_serverchan(title, desp):
    if not SERVERCHAN_KEYS:
        print("SERVERCHAN_KEY未配置，跳过推送")
        return False
    success_count = 0
    for key in SERVERCHAN_KEYS:
        url = f"https://sctapi.ftqq.com/{key}.send"
        data = {"title": title, "desp": desp}
        try:
            resp = requests.post(url, data=data, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                if result.get("code") == 0:
                    success_count += 1
        except Exception as e:
            print(f"  Server酱推送失败: {e}")
    print(f"  Server酱推送: {success_count}/{len(SERVERCHAN_KEYS)}")
    return success_count > 0


# ══════════════════════════════════════════════════════════
#  OAMV活跃市值择时
# ══════════════════════════════════════════════════════════

def get_oamv_status():
    """获取OAMV活跃市值当前状态（全市场真实活跃市值，复刻指南针）"""
    from ml_strategy.oamv_filter import OAMVHysteresisFilter
    from ml_strategy.market_amv_cache import get_market_amv_series

    try:
        # 获取全市场活跃市值时间序列 (circ_mv × turnover_rate 聚合)
        amv_series = get_market_amv_series()
        if amv_series is None or len(amv_series) < 40:
            print("  全市场活跃市值数据不足，回退到成交额代理")
            # 回退: 使用沪深300成交额代理
            end_date = pd.Timestamp.now().strftime("%Y%m%d")
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime("%Y%m%d")
            index_df = _get_pro().index_daily(ts_code="000300.SH", start_date=start_date, end_date=end_date)
            if index_df is None or len(index_df) < 40:
                return None
            index_df = index_df.sort_values("trade_date").reset_index(drop=True)
            index_df["Date"] = pd.to_datetime(index_df["trade_date"], format="%Y%m%d")
            index_df.set_index("Date", inplace=True)
            index_df["amount"] = index_df["amount"].astype(float)
            oamv = OAMVHysteresisFilter(
                upper_threshold=2.0, lower_threshold=-1.0,
                cost_ma_period=42, smooth_method='sma', smooth_period=15,
            )
            oamv.fit(index_df)
            data_source = "成交额代理(amount)"
        else:
            # 优化后参数: SMA(15)平滑 + CostMA(42), 阈值+2.0%/-1.0%
            oamv = OAMVHysteresisFilter(
                upper_threshold=2.0, lower_threshold=-1.0,
                cost_ma_period=42, roc_period=1,
                weekly_ema_period=5, weekly_use_ema=True,
                smooth_method='sma', smooth_period=15,
                cost_ma_method='sma',
            )
            oamv.fit(amv_series=amv_series)
            data_source = "优化后活筹(SMA15+CostMA42|+2.0/-1.0)"

        state_df = oamv.get_state_df()
        if state_df is None or len(state_df) == 0:
            return None

        latest_date = state_df.index[-1]
        latest_state = int(state_df.loc[latest_date, "oamv_state"])
        latest_x = float(state_df.loc[latest_date, "oamv_x"])
        daily_allowed = latest_state == 1
        weekly_allowed = oamv.is_trading_allowed(latest_date, require_weekly=True)

        recent_states = []
        for i in range(min(5, len(state_df))):
            d = state_df.index[-(i+1)]
            s = int(state_df.loc[d, "oamv_state"])
            x = float(state_df.loc[d, "oamv_x"])
            recent_states.append({
                "date": d.strftime("%m-%d"),
                "state": "允许" if s == 1 else "禁止",
                "x": round(x, 2),
            })

        transitions = oamv.get_transition_dates()
        last_transition = None
        if transitions:
            lt = transitions[-1]
            last_transition = {
                "date": lt["date"].strftime("%Y-%m-%d") if hasattr(lt["date"], "strftime") else str(lt["date"]),
                "to_state": "允许" if lt["to"] == 1 else "禁止",
            }

        return {
            "latest_date": latest_date.strftime("%Y-%m-%d"),
            "daily_allowed": daily_allowed,
            "weekly_allowed": weekly_allowed,
            "can_open_position": weekly_allowed,
            "latest_x": round(latest_x, 2),
            "data_source": data_source,
            "recent_states": recent_states,
            "last_transition": last_transition,
            "trend_label": "活跃上升" if latest_state == 1 else "萎缩下降",
        }
    except Exception as e:
        print(f"  OAMV计算失败: {e}")
        return None


# ══════════════════════════════════════════════════════════
#  行业热度分析
# ══════════════════════════════════════════════════════════

def compute_industry_analysis(signals_data, industry_map):
    """
    行业间分析：计算每个行业的动量、信号数量、热度排名、轮动信号

    参数:
      signals_data: {ts_code: df} 已计算的指标+信号数据
      industry_map: {ts_code: industry_name} 股票→行业映射

    返回:
      industry_stats: [{name, momentum, momentum_change, rotation, signal_count, stock_count, hot_cold}]
    """
    mom_days = BEST_PARAMS.get("industry_momentum_days", 10)

    # 1. 计算行业动量
    mom_df = compute_industry_momentum(signals_data, industry_map, mom_days)

    # 2. 获取最新动量值和动量变化
    if mom_df.empty or len(mom_df) < 6:
        return []

    latest_mom = mom_df.iloc[-1]
    # 5日前的动量（用于计算轮动）
    lookback = min(5, len(mom_df) - 1)
    prev_mom = mom_df.iloc[-1 - lookback]

    # 3. 统计每个行业的信号数量
    industry_signal_count = defaultdict(int)
    industry_stock_count = defaultdict(int)

    for ts_code, df in signals_data.items():
        industry = industry_map.get(ts_code, "")
        if not industry:
            continue
        industry_stock_count[industry] += 1
        if len(df) > 0 and df.iloc[-1].get("ambush_signal", False):
            industry_signal_count[industry] += 1

    # 4. 构建行业统计
    industry_stats = []
    for industry in mom_df.columns:
        momentum = float(latest_mom.get(industry, 0))
        prev_momentum = float(prev_mom.get(industry, 0))
        momentum_change = momentum - prev_momentum
        signal_count = industry_signal_count.get(industry, 0)
        stock_count = industry_stock_count.get(industry, 0)

        # 热度分类
        if momentum > 0.05:
            hot_cold = "火热"
        elif momentum > 0.02:
            hot_cold = "偏热"
        elif momentum > 0:
            hot_cold = "微热"
        elif momentum > -0.02:
            hot_cold = "微冷"
        elif momentum > -0.05:
            hot_cold = "偏冷"
        else:
            hot_cold = "冰冷"

        # 轮动信号
        was_hot = prev_momentum > BEST_PARAMS.get("industry_momentum_threshold", 0.02)
        is_hot = momentum > BEST_PARAMS.get("industry_momentum_threshold", 0.02)
        if not was_hot and is_hot:
            rotation = "轮入"
        elif was_hot and not is_hot:
            rotation = "轮出"
        elif is_hot and momentum_change > 0.01:
            rotation = "加速"
        elif is_hot and momentum_change < -0.01:
            rotation = "减速"
        elif not is_hot and momentum_change > 0.01:
            rotation = "回暖"
        elif not is_hot and momentum_change < -0.01:
            rotation = "恶化"
        else:
            rotation = "平稳"

        industry_stats.append({
            "name": industry,
            "momentum": round(momentum * 100, 2),
            "momentum_change": round(momentum_change * 100, 2),
            "rotation": rotation,
            "signal_count": signal_count,
            "stock_count": stock_count,
            "hot_cold": hot_cold,
        })

    # 按动量排序
    industry_stats.sort(key=lambda x: x["momentum"], reverse=True)
    return industry_stats


# ══════════════════════════════════════════════════════════
#  全市场扫描
# ══════════════════════════════════════════════════════════

def get_all_a_stocks():
    """获取全市场A股列表 —— 优先akshare（无限流），备用tushare"""
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
                # 转换为ts_code格式
                if code.startswith("6"):
                    ts_code = f"{code}.SH"
                elif code.startswith("0") or code.startswith("3"):
                    ts_code = f"{code}.SZ"
                else:
                    continue
                industry = str(row.get("行业", ""))
                stocks.append((ts_code, name, industry))
            if len(stocks) > 100:
                print(f"  akshare获取股票列表: {len(stocks)}只")
                return stocks
    except Exception as e:
        print(f"  akshare获取股票列表失败: {e}")
    # 降级到tushare
    try:
        stock_basic = _get_pro().stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,industry,list_date")
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
        print(f"获取股票列表失败: {e}")
        return []


def get_stock_data(ts_code):
    """获取单只股票日线数据（带增量缓存，首次运行后大幅加速）"""
    return get_stock_data_cached(ts_code, min_rows=130)


def get_realtime_quotes():
    """
    获取全市场实时行情（akshare），返回 {ts_code: {Open,High,Low,Close,Volume,Amount,...}} 字典
    盘中推送时用于拼接当日实时K线到昨日缓存数据上
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

        print(f"  实时行情获取: {len(quotes)}只", flush=True)
        return quotes
    except Exception as e:
        print(f"  实时行情获取失败: {e}", flush=True)
        return {}


def append_realtime_bar(df, realtime_quote, today_str=None):
    """
    将akshare实时行情拼接到日线数据末尾，形成盘中实时K线

    参数:
      df: 已计算指标的DataFrame（昨日或更早数据）
      realtime_quote: {Open, High, Low, Close, Volume, Amount, ...}
      today_str: 今日日期字符串，默认当天

    返回:
      拼接后的DataFrame（最后一行为实时K线），或原始df（拼接失败时）
    """
    if not realtime_quote or realtime_quote.get("Close", 0) <= 0:
        return df

    try:
        today = pd.Timestamp(today_str or datetime.now().strftime("%Y-%m-%d"))

        # 如果最后一行已经是今天，更新它而不是追加
        if len(df) > 0 and df.index[-1] == today:
            df.iloc[-1]["Open"] = realtime_quote["Open"]
            df.iloc[-1]["High"] = realtime_quote["High"]
            df.iloc[-1]["Low"] = realtime_quote["Low"]
            df.iloc[-1]["Close"] = realtime_quote["Close"]
            df.iloc[-1]["Volume"] = realtime_quote["Volume"]
            df.iloc[-1]["Amount"] = realtime_quote["Amount"]
            return df

        # 构造今日实时K线行
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


def analyze_signal_detail(df, signal_idx):
    """对信号日进行详细分析"""
    row = df.iloc[signal_idx]
    prev = df.iloc[signal_idx - 1] if signal_idx > 0 else row
    body = abs(row["Close"] - row["Open"])
    amplitude = row["High"] - row["Low"]
    lower_shadow = min(row["Close"], row["Open"]) - row["Low"]
    upper_shadow = row["High"] - max(row["Close"], row["Open"])

    analysis = {}

    # 威科夫解读
    wyckoff_parts = []
    if row.get("tag_sos_anchor", False):
        wyckoff_parts.append("SOS需求确认(主力入场)")
    if row.get("tag_no_supply", False):
        wyckoff_parts.append("No Supply供应枯竭")
    if row.get("tag_test", False):
        wyckoff_parts.append("Test测试柱(需求保护)")
    window = BEST_PARAMS["ambush_window"]
    if signal_idx >= window:
        recent_sos = df.iloc[signal_idx-window:signal_idx+1]["tag_sos_anchor"].any()
        if recent_sos and not row.get("tag_sos_anchor", False):
            wyckoff_parts.append(f"近{window}日有SOS锚定(LPS回踩)")
    if row["J"] < BEST_PARAMS["ambush_j_oversold"]:
        wyckoff_parts.append(f"J={row['J']:.0f}情绪冰点(超卖)")
    analysis["wyckoff"] = wyckoff_parts if wyckoff_parts else ["标准潜伏信号"]

    # VPA量价解读
    vpa_parts = []
    vol_ratio = row["Volume"] / row["volume_ma"] if row["volume_ma"] > 0 else 0
    if vol_ratio < BEST_PARAMS["ambush_vol_shrink"]:
        vpa_parts.append(f"缩量({vol_ratio:.1%}均量=供应枯竭)")
    else:
        vpa_parts.append(f"量比{vol_ratio:.2f}")
    if body / (row["Close"] + 1e-8) < BEST_PARAMS["ambush_body_pct"]:
        vpa_parts.append("小实体(多空平衡/拒绝下跌)")
    if lower_shadow > body * 1.5:
        vpa_parts.append("下影线支撑(需求托底)")
    analysis["vpa"] = vpa_parts

    # 蜡烛图解读
    candle_parts = []
    if body < amplitude * 0.1:
        candle_parts.append("十字星(方向选择)")
    elif row["Close"] > row["Open"] and (row["Close"] - row["Low"]) / (amplitude + 1e-8) > 0.7:
        candle_parts.append("阳线收高(强势)")
    elif row["Close"] < row["Open"] and lower_shadow > body * 2:
        candle_parts.append("锤子线(潜在反转)")
    if not candle_parts:
        candle_parts.append("阳线" if row["Close"] >= row["Open"] else "阴线")
    analysis["candle"] = candle_parts

    # 支撑/阻力
    analysis["support"] = round(float(row["yellow_line"] - 0.5 * row["atr14"]), 2)
    analysis["resistance"] = round(float(row["yellow_line"] + 1.5 * row["atr14"]), 2)

    return analysis


def batch_prefilter_stocks():
    """用akshare批量获取全市场实时行情，快速预筛选潜在信号股"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is None or len(df) == 0:
            return None
        # 基本过滤：排除ST、北交所、停牌
        df = df[~df["名称"].str.startswith("ST", na=False)]
        df = df[~df["名称"].str.startswith("*ST", na=False)]
        df = df[~df["名称"].str.startswith("N", na=False)]
        df = df[~df["名称"].str.contains("退", na=False)]
        # 排除停牌（成交量为0）
        if "成交量" in df.columns:
            df = df[df["成交量"] > 0]
        # 排除北交所（代码8/9开头）
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
        print(f"  批量预筛选: {len(df)}只（排除ST/停牌/北交所/已大涨/极端价格/低换手）")
        return df
    except Exception as e:
        print(f"  批量预筛选失败: {e}")
        return None


def _fetch_and_process_one_core(ts_code, name, industry, best_params, realtime_quote=None):
    """获取单只股票数据并计算指标的核心逻辑（不含重试和异常捕获）"""
    df = get_stock_data(ts_code)
    if df is None:
        return (ts_code, name, industry, None)
    # 盘中模式：拼接实时K线到昨日缓存
    if realtime_quote is not None:
        df = append_realtime_bar(df, realtime_quote)
    df = IndicatorCalcBase(df)
    df = add_micro_confirm_indicators(df)
    df = add_inst_support_indicators(df, best_params)
    df = Detect_AmbushSignal_V64(df, best_params)
    if df is None or len(df) < 130:
        return (ts_code, name, industry, None)
    return (ts_code, name, industry, df)


def _fetch_and_process_one_with_retry(ts_code, name, industry, best_params, realtime_quote=None):
    """带重试的单只股票处理（使用tenacity自动重试网络异常）"""
    if TENACITY_AVAILABLE:
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
               retry=retry_if_exception_type((ConnectionError, TimeoutError)))
        def _retry_wrapper():
            return _fetch_and_process_one_core(ts_code, name, industry, best_params, realtime_quote)
        return _retry_wrapper()
    else:
        return _fetch_and_process_one_core(ts_code, name, industry, best_params, realtime_quote)


def _fetch_and_process_one(ts_code, name, industry, best_params, realtime_quote=None):
    """获取单只股票数据并计算指标，返回 (ts_code, name, industry, df_or_None)"""
    try:
        return _fetch_and_process_one_with_retry(ts_code, name, industry, best_params, realtime_quote)
    except Exception as e:
        print(f"  股票处理异常 {ts_code}({name}): {e}", flush=True)
        return (ts_code, name, industry, None)


# 断点续传状态文件路径
SCAN_STATUS_FILE = RESULT_DIR / "scan_status.json"


def _load_scan_status():
    """加载断点续传状态：返回已完成的股票代码集合"""
    if SCAN_STATUS_FILE.exists():
        try:
            with open(SCAN_STATUS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            completed = set(data.get("completed", []))
            print(f"  断点续传: 发现{len(completed)}只已完成股票", flush=True)
            return completed
        except Exception as e:
            print(f"  断点续传状态加载失败: {e}", flush=True)
    return set()


def _save_scan_status(completed_set):
    """保存断点续传状态"""
    try:
        SCAN_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SCAN_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump({"completed": list(completed_set)}, f, ensure_ascii=False)
    except Exception as e:
        print(f"  断点续传状态保存失败: {e}", flush=True)


def _clear_scan_status():
    """扫描完成后删除状态文件"""
    try:
        if SCAN_STATUS_FILE.exists():
            SCAN_STATUS_FILE.unlink()
            print("  断点续传状态文件已清理", flush=True)
    except Exception as e:
        print(f"  断点续传状态文件清理失败: {e}", flush=True)


def scan_market(oamv_weekly_allowed_dates=None, industry_allow_matrix=None, industry_map=None, prefilter_df=None, realtime_quotes=None):
    """全市场扫描潜伏信号（并发获取数据 + 断点续传 + 盘中实时拼接）"""
    all_stocks = get_all_a_stocks()
    if not all_stocks:
        print("无法获取股票列表")
        return [], {}

    # 使用批量预筛选结果过滤股票列表
    if prefilter_df is not None:
        prefilter_codes = set(prefilter_df["ts_code"].tolist())
        original_count = len(all_stocks)
        all_stocks = [(tc, n, ind) for tc, n, ind in all_stocks if tc in prefilter_codes]
        print(f"  预筛选后股票数: {len(all_stocks)}/{original_count}", flush=True)

    # 断点续传：跳过已完成的股票
    completed_set = _load_scan_status()
    if completed_set:
        before_count = len(all_stocks)
        all_stocks = [(tc, n, ind) for tc, n, ind in all_stocks if tc not in completed_set]
        print(f"  断点续传: 跳过{before_count - len(all_stocks)}只已完成股票，剩余{len(all_stocks)}只", flush=True)

    is_intraday = realtime_quotes is not None and len(realtime_quotes) > 0
    total = len(all_stocks)
    print(f"扫描股票数(预筛选后): {total} | 并发数: 10 | 模式: {'盘中实时' if is_intraday else '盘后完整'}", flush=True)

    signals = []
    all_signals_data = {}
    processed = 0
    errors = 0
    start_time = time.time()

    # 并发获取数据（10个线程）
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for ts_code, name, industry in all_stocks:
            # 盘中模式：每只股票传入对应的实时行情
            rt_quote = realtime_quotes.get(ts_code) if is_intraday else None
            future = executor.submit(_fetch_and_process_one, ts_code, name, industry, BEST_PARAMS, rt_quote)
            futures[future] = (ts_code, name, industry)

        for future in as_completed(futures):
            ts_code, name, industry = futures[future]
            processed += 1

            if processed % 200 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / processed * (total - processed) if processed > 0 else 0
                print(f"  进度: {processed}/{total} ({processed/total*100:.1f}%) | "
                      f"信号:{len(signals)} | 失败:{errors} | ETA:{eta:.0f}s", flush=True)

            try:
                result_ts_code, result_name, result_industry, df = future.result()
            except Exception:
                errors += 1
                completed_set.add(ts_code)
                continue

            if df is None:
                errors += 1
                completed_set.add(ts_code)
                continue

            # 保存用于行业分析
            all_signals_data[ts_code] = df

            latest = df.iloc[-1]
            if pd.isna(latest.get("yellow_line")) or pd.isna(latest.get("white_line")):
                continue

            signal_date = df.index[-1]
            if oamv_weekly_allowed_dates is not None:
                if signal_date not in oamv_weekly_allowed_dates:
                    continue

            if latest.get("ambush_signal", False):
                # 行业热度过滤
                ind_allowed = True
                if industry_allow_matrix is not None and industry and industry in industry_allow_matrix.columns:
                    try:
                        ind_val = industry_allow_matrix[industry].reindex([signal_date])
                        if not ind_val.empty and not ind_val.iloc[0]:
                            ind_allowed = False
                    except:
                        pass

                if not ind_allowed:
                    continue

                prev = df.iloc[-2]
                change_pct = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
                vol_ratio = latest["Volume"] / latest["volume_ma"] if latest["volume_ma"] > 0 else 0

                detail = analyze_signal_detail(df, len(df) - 1)

                window = BEST_PARAMS["ambush_window"]
                sos_dates = []
                for j in range(max(0, len(df) - window), len(df)):
                    if df.iloc[j].get("tag_sos_anchor", False):
                        sos_dates.append(df.index[j].strftime("%m-%d"))

                # 入场质量评分
                eq_score = int(latest.get("entry_quality_score", 0)) if "entry_quality_score" in df.columns else 0

                # 止损价计算
                hard_stop = round(float(latest["Close"] * 0.85), 2)  # 硬止损: -15%
                chandelier_init = round(float(latest["Close"] - 3 * latest["atr14"]), 2)  # 吊灯线初始

                signal_info = {
                    "code": ts_code,
                    "name": name,
                    "industry": industry,
                    "price": round(float(latest["Close"]), 2),
                    "change_pct": round(float(change_pct), 2),
                    "white_line": round(float(latest["white_line"]), 2),
                    "yellow_line": round(float(latest["yellow_line"]), 2),
                    "J": round(float(latest["J"]), 1),
                    "atr14": round(float(latest["atr14"]), 2),
                    "vol_ratio": round(float(vol_ratio), 2),
                    "sos_dates": sos_dates,
                    "analysis": detail,
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    # V6.4：入场质量评分
                    "entry_quality_score": eq_score,
                    # 止损参考
                    "hard_stop": hard_stop,
                    "chandelier_init": chandelier_init,
                    # 主力托底评分（兼容旧字段）
                    "inst_support_score": int(latest.get("inst_support_score", 0)),
                    "factor_a": bool(latest.get("factor_a_vol_stable", False)),
                    "factor_b": bool(latest.get("factor_b_vp_divergence", False)),
                    "factor_c": bool(latest.get("factor_c_support_hold", False)),
                    "factor_d": bool(latest.get("factor_d_intraday_accum", False)),
                }
                signals.append(signal_info)
                print(f"  潜伏信号: {name}({ts_code}) [{industry}] {latest['Close']:.2f} {change_pct:+.2f}% "
                      f"J:{latest['J']:.1f} 量比:{vol_ratio:.2f} 评分:{eq_score}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n扫描完成! 耗时: {elapsed/60:.1f}min | 信号: {len(signals)}只 | 错误: {errors}", flush=True)

    # 扫描完成，清理断点续传状态文件
    _clear_scan_status()

    return signals, all_signals_data


# ══════════════════════════════════════════════════════════
#  精细动态评分过滤
# ══════════════════════════════════════════════════════════

def apply_dynamic_score_filter(signals, oamv_status):
    """
    精细动态评分过滤：
    - OAMV牛市: 评分>=5 或 (评分=4且J<3且量比<0.6)
    - OAMV熊市: 评分>=6
    - 所有信号 J<10
    """
    if not signals:
        return signals

    is_bull = oamv_status and oamv_status.get("can_open_position", False)
    dsp = DYNAMIC_SCORE_PARAMS

    filtered = []
    for s in signals:
        j = s.get("J", 99)
        eq = s.get("entry_quality_score", 0)
        vr = s.get("vol_ratio", 1.0)

        # J值硬上限
        if j >= dsp["j_hard_cap"]:
            continue

        if is_bull:
            # 牛市：评分>=5 直接通过
            if eq >= dsp["bull_min_score"]:
                filtered.append(s)
            # 评分=4 但J<3且量比<0.6（极度冰点+极度缩量）也通过
            elif eq == 4 and j < dsp["bull_score4_j_max"] and vr < dsp["bull_score4_vol_ratio_max"]:
                filtered.append(s)
        else:
            # 熊市：只允许评分>=6
            if eq >= dsp["bear_min_score"]:
                filtered.append(s)

    return filtered


# ══════════════════════════════════════════════════════════
#  推送消息构建（微信格式）
# ══════════════════════════════════════════════════════════

def build_push_message(oamv_status, signals, industry_stats, is_intraday=False):
    today = datetime.now().strftime("%Y-%m-%d")

    # ── 市场情绪判断 ──
    if oamv_status:
        can_open = oamv_status["can_open_position"]
        if can_open:
            sentiment = "偏多"
            sentiment_icon = "🟢"
        else:
            sentiment = "偏空"
            sentiment_icon = "🔴"
    else:
        can_open = True
        sentiment = "中性"
        sentiment_icon = "⚪"

    mode_tag = "盘中实时" if is_intraday else "盘后完整"
    title = f"{sentiment_icon} 量化潜伏 {today} | {len(signals)}信号 | {sentiment} | {mode_tag}"

    lines = []

    # ── 头部 ──
    lines.append(f"今日信号:{len(signals)}只")
    if is_intraday:
        lines.append(f"模式:盘中实时扫描(数据截至{datetime.now().strftime('%H:%M')})")

    # ── OAMV市场环境 ──
    if oamv_status:
        oamv_label = "牛市(允许开仓)" if can_open else "熊市(控制仓位)"
        lines.append(f"OAMV:{oamv_label}|趋势:{oamv_status['trend_label']}|X:{oamv_status['latest_x']}")
        if oamv_status.get("last_transition"):
            lt = oamv_status["last_transition"]
            lines.append(f"趋势切换:{lt['date']}→{lt['to_state']}")
    else:
        lines.append("OAMV:环境评估中")

    lines.append("")

    # ── 行业风向（精简） ──
    if industry_stats:
        hot = [s for s in industry_stats if s["momentum"] > 0]
        rotation_in = [s for s in industry_stats if s["rotation"] == "轮入"]
        rotation_out = [s for s in industry_stats if s["rotation"] == "轮出"]

        if hot:
            hot_names = "、".join(s["name"] for s in hot[:6])
            lines.append(f"强势行业({len(hot)}个):{hot_names}")
        if rotation_in:
            ri_names = "、".join(s["name"] for s in rotation_in[:4])
            lines.append(f"轮入:{ri_names}")
        if rotation_out:
            ro_names = "、".join(s["name"] for s in rotation_out[:4])
            lines.append(f"轮出:{ro_names}")
        lines.append("")

    # ── 潜伏信号（完整复刻微信格式） ──
    if not can_open:
        lines.append("⚠️当前环境偏弱，以下标的仅供跟踪观察")
        lines.append("")

    for i, s in enumerate(signals, 1):
        # 1. 股票名称+代码+行业
        lines.append(f"{i}.{s['name']}({s['code']}){s['industry']}")

        # 2. 价格+涨跌
        change_sign = "+" if s['change_pct'] >= 0 else ""
        lines.append(f"价格:{s['price']:.2f}|涨跌:{change_sign}{s['change_pct']:.2f}%")

        # 3. 白线+黄线+关系
        ma_rel = "白>黄" if s['white_line'] > s['yellow_line'] else "白<黄" if s['white_line'] < s['yellow_line'] else "白=黄"
        lines.append(f"白线:{s['white_line']:.2f}|黄线:{s['yellow_line']:.2f}|{ma_rel}")

        # 4. J值+量比+ATR
        lines.append(f"·J值:{s['J']:.1f}|量比:{s['vol_ratio']:.2f}|ATR:{s['atr14']:.2f}")

        # 5. SOS锚定日
        if s.get('sos_dates'):
            lines.append(f"SOS锚定日:{','.join(s['sos_dates'])}")

        # 6. 威科夫解读
        analysis = s.get('analysis', {})
        wyckoff = analysis.get('wyckoff', [])
        if wyckoff:
            lines.append(f"·威科夫:{';'.join(wyckoff)}")

        # 7. VPA量价解读
        vpa = analysis.get('vpa', [])
        if vpa:
            lines.append(f"VPA量价:{';'.join(vpa)}")

        # 8. 蜡烛图解读
        candle = analysis.get('candle', [])
        if candle:
            lines.append(f"·蜡烛图:{';'.join(candle)}")

        # 9. 支撑+阻力
        support = analysis.get('support', s['yellow_line'])
        resistance = analysis.get('resistance', s['yellow_line'])
        lines.append(f"支撑:{support}|阻力:{resistance}")

        # 10. T+1参考买入+硬止损+吊灯线初始
        lines.append(f"·T+1参考买入:{s['price']:.2f}(开盘价)|硬止损:{s['hard_stop']:.2f}|吊灯线初始:{s['chandelier_init']:.2f}")

        # 11. 评分信息
        eq = s.get('entry_quality_score', 0)
        lines.append(f"·评分:{eq}/8|模型:潜伏模型V6.4|理论:威科夫LPS+VPA量价|择时:OAMV+行业动量|退出:4级(硬止损→吊灯→BC→时间)")

        # 空行分隔
        lines.append("")

    # ── 页脚 ──
    if not signals:
        if can_open:
            lines.append("今日无符合条件的标的")
        else:
            lines.append("环境偏弱，暂无值得关注的标的")
        lines.append("")

    lines.append("·量化潜伏系统·多维度量化筛选")
    lines.append("·以上内容为系统量化输出，不构成投资建议，据此操作风险自担")

    desp = "\n".join(lines)
    return title, desp


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def prewarm_data():
    """数据预热：确保缓存就绪，提前获取关键数据"""
    print("\n[预热] 检查数据缓存...", flush=True)

    # 1. 检查DuckDB缓存状态
    cache_stats = get_cache_stats()
    cache_count = cache_stats.get("count", 0)
    cache_size = cache_stats.get("size_mb", 0)
    print(f"  DuckDB缓存: {cache_count}只股票 | {cache_size}MB", flush=True)

    # 2. 检查akshare可用性（盘中推送依赖实时数据）
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is not None and len(df) > 100:
            print(f"  akshare实时行情: 可用 ({len(df)}只)", flush=True)
        else:
            print("  akshare实时行情: 数据异常", flush=True)
    except Exception as e:
        print(f"  akshare实时行情: 不可用 ({e})", flush=True)

    # 3. 检查tushare可用性
    try:
        pro = _get_pro()
        test_df = pro.trade_cal(exchange="SSE", is_open="1", limit=1)
        print(f"  tushare接口: 可用", flush=True)
    except Exception as e:
        print(f"  tushare接口: 不可用 ({e})", flush=True)

    # 4. 判断当前时段
    now = datetime.now()
    hour = now.hour
    if 9 <= hour < 15:
        print(f"  当前时段: 盘中({hour}:00) → 使用akshare实时数据+昨日信号", flush=True)
    elif hour >= 15:
        print(f"  当前时段: 盘后({hour}:00) → 使用tushare完整日线数据", flush=True)
    else:
        print(f"  当前时段: 盘前({hour}:00) → 使用缓存数据", flush=True)

    print("[预热] 完成", flush=True)


def daily_push():
    print("=" * 80, flush=True)
    print("潜伏模型V6.4 每日实盘推送（精细动态评分版）", flush=True)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"优化参数: 评分≥5 | J<10 | window=8 | industry_top=20% | eq_j_extreme=3", flush=True)
    print(f"动态评分: 牛市≥5或(4+J<3+量比<0.6) | 熊市≥6 | J<10", flush=True)
    print("=" * 80, flush=True)

    # 0. 数据预热
    prewarm_data()

    # 0.5 判断盘中/盘后模式
    now = datetime.now()
    is_intraday = 9 <= now.hour < 15  # 9:00~14:59 为盘中
    if is_intraday:
        print(f"\n>>> 盘中实时模式 <<<", flush=True)
    else:
        print(f"\n>>> 盘后完整模式 <<<", flush=True)

    # 1. OAMV活跃市值择时
    print("\n[1/6] 计算OAMV活跃市值择时...", flush=True)
    oamv_status = get_oamv_status()
    if oamv_status:
        can_open = oamv_status["can_open_position"]
        print(f"  择时状态: {'允许开仓(牛市)' if can_open else '禁止开仓(熊市)'} | "
              f"OAMV={oamv_status['latest_x']} | {oamv_status['trend_label']}", flush=True)
    else:
        print("  OAMV计算失败", flush=True)

    # 2. 获取行业分类
    print("\n[2/6] 获取行业分类...", flush=True)
    try:
        basic = _get_pro().stock_basic(fields="ts_code,industry", list_status="L")
        industry_map = dict(zip(basic["ts_code"], basic["industry"]))
        print(f"  行业映射: {len(industry_map)}只股票", flush=True)
    except Exception as e:
        print(f"  行业分类获取失败: {e}", flush=True)
        industry_map = {}

    # 3. 盘中模式：获取实时行情
    realtime_quotes = None
    if is_intraday:
        print("\n[3/6] 获取akshare实时行情（盘中拼接）...", flush=True)
        realtime_quotes = get_realtime_quotes()
        if realtime_quotes:
            print(f"  实时行情: {len(realtime_quotes)}只 → 将拼接为今日实时K线", flush=True)
        else:
            print("  实时行情获取失败，回退到盘后模式（使用昨日缓存）", flush=True)
            is_intraday = False
    else:
        print("\n[3/6] 盘后模式，跳过实时行情获取", flush=True)

    # 4. 全市场扫描
    print("\n[4/6] 全市场扫描潜伏信号...", flush=True)
    cache_stats = get_cache_stats()
    print(f"  股票缓存: {cache_stats.get('count', 0)}只 | {cache_stats.get('size_mb', 0)}MB", flush=True)
    print("  批量预筛选全市场行情...", flush=True)
    prefilter_df = batch_prefilter_stocks()
    signals, all_signals_data = scan_market(
        industry_allow_matrix=None,  # 先扫描所有信号，行业过滤在后面做
        industry_map=industry_map,
        prefilter_df=prefilter_df,
        realtime_quotes=realtime_quotes,
    )

    # 5. 行业热度分析 + 行业过滤
    print("\n[5/6] 行业热度分析...", flush=True)
    industry_stats = []
    industry_allow_matrix = None
    if all_signals_data and industry_map:
        industry_stats = compute_industry_analysis(all_signals_data, industry_map)
        print(f"  行业数: {len(industry_stats)}", flush=True)

        # 构建行业允许买入矩阵
        mom_days = BEST_PARAMS.get("industry_momentum_days", 10)
        mom_threshold = BEST_PARAMS.get("industry_momentum_threshold", 0.0)
        mom_df = compute_industry_momentum(all_signals_data, industry_map, mom_days)
        if not mom_df.empty:
            industry_allow_matrix = build_industry_allow_matrix(mom_df, mom_threshold)

        # 用行业过滤重新筛选信号
        filtered_signals = []
        for s in signals:
            industry = s.get("industry", "")
            if industry_allow_matrix is not None and industry and industry in industry_allow_matrix.columns:
                try:
                    signal_date = pd.Timestamp(s["signal_date"])
                    ind_val = industry_allow_matrix[industry].reindex([signal_date])
                    if not ind_val.empty and not ind_val.iloc[0]:
                        continue  # 行业动量不足，过滤
                except:
                    pass
            filtered_signals.append(s)
        print(f"  行业过滤: {len(signals)}只 → {len(filtered_signals)}只", flush=True)
        signals = filtered_signals

    # 6. 精细动态评分过滤
    print("\n[6/6] 精细动态评分过滤...", flush=True)
    before_dynamic = len(signals)
    signals = apply_dynamic_score_filter(signals, oamv_status)
    print(f"  动态评分过滤: {before_dynamic}只 → {len(signals)}只", flush=True)
    if oamv_status:
        is_bull = oamv_status.get("can_open_position", False)
        print(f"  OAMV状态: {'牛市' if is_bull else '熊市'} | "
              f"规则: {'评分≥5或(4+J<3+量比<0.6)' if is_bull else '评分≥6'} | J<10", flush=True)

    # 构建推送消息
    print("\n构建推送消息...", flush=True)
    title, desp = build_push_message(oamv_status, signals, industry_stats, is_intraday=is_intraday)

    # 保存结果
    result = {
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "V6.4-精细动态评分",
        "mode": "盘中实时" if is_intraday else "盘后完整",
        "params": {
            "entry_quality_min_score": BEST_PARAMS["entry_quality_min_score"],
            "ambush_j_oversold": BEST_PARAMS["ambush_j_oversold"],
            "ambush_window": BEST_PARAMS["ambush_window"],
            "industry_rs_top_pct": BEST_PARAMS["industry_rs_top_pct"],
            "eq_j_extreme": BEST_PARAMS["eq_j_extreme"],
        },
        "dynamic_score_rules": DYNAMIC_SCORE_PARAMS,
        "oamv_status": oamv_status,
        "signal_count": len(signals),
        "signals": signals,
        "industry_stats": industry_stats[:30],
    }
    result_file = RESULT_DIR / f"daily_push_{datetime.now().strftime('%Y%m%d')}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"  结果已保存: {result_file}", flush=True)

    # 推送
    print("\n推送微信...", flush=True)
    send_serverchan(title, desp)
    print("推送完成", flush=True)

    # 摘要
    print(f"\n{'='*80}")
    print(f"OAMV择时: {'允许(牛市)' if oamv_status and oamv_status['can_open_position'] else '禁止(熊市)'}")
    print(f"潜伏信号: {len(signals)}只")
    for s in signals:
        eq = s.get('entry_quality_score', 0)
        print(f"  - {s['name']}({s['code']}) [{s['industry']}] {s['price']:.2f} J:{s['J']:.1f} 评分:{eq}")
    if industry_stats:
        hot = [s for s in industry_stats if s["momentum"] > 0]
        rot_in = [s for s in industry_stats if s["rotation"] == "轮入"]
        rot_out = [s for s in industry_stats if s["rotation"] == "轮出"]
        print(f"行业热度: {len(hot)}个偏热 / {len(industry_stats)-len(hot)}个偏冷 | 轮入:{len(rot_in)} 轮出:{len(rot_out)}")
        if rot_in:
            print(f"  轮入: {', '.join(s['name'] for s in rot_in[:5])}")
        if rot_out:
            print(f"  轮出: {', '.join(s['name'] for s in rot_out[:5])}")


if __name__ == "__main__":
    daily_push()
