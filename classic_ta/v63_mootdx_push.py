"""
潜伏模型V6.3 盘前扫描推送 - 含波动率平价仓位
==========================================================
基于V6.3最佳参数 + 行业动量过滤 + OAMV择时 + 微观确认 + 波动率平价仓位
使用akshare数据源（前复权）+ 并发获取 + 批量预筛选

推送内容：
  1. OAMV活跃市值择时状态
  2. 行业热度分析（行业动量排名、冷热分布、轮动信号）
  3. 潜伏买入信号（含行业动量过滤）
  4. 持仓监控（4级退出）

用法：
    python -m classic_ta.v63_mootdx_push              # 全市场扫描
    python -m classic_ta.v63_mootdx_push --test 100    # 测试模式
"""

import sys
import os
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd
import tushare as ts
import requests
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 重试机制
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

# ─── 项目路径 ───
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

# ─── 日志配置 ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("V63_Mootdx_Push")

# ─── Server酱 ───
SERVERCHAN_KEYS = [k.strip() for k in os.getenv("SERVERCHAN_KEY", "").split(",") if k.strip()]

# ─── 输出目录 ───
RESULT_DIR = Path(__file__).parent.parent / "results" / "v63_mootdx"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ─── V6.3模型 ───
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

BEST_PARAMS = V64_PARAMS.copy()

# ══════════════════════════════════════════════════════════
#  Mootdx测速优选
# ══════════════════════════════════════════════════════════

def mootdx_bestip():
    """动态寻找延迟最低的通达信行情节点"""
    try:
        from mootdx.quotes import Quotes
        best = Quotes.bestip()
        logger.info(f"Mootdx最优节点: {best}")
        return best
    except Exception as e:
        logger.warning(f"Mootdx测速失败: {e}")
        return None


# 股票日线数据增量缓存
from classic_ta.stock_data_duckdb import get_stock_data_cached, get_cache_stats


# ══════════════════════════════════════════════════════════
#  Server酱推送
# ══════════════════════════════════════════════════════════

def send_serverchan(title, desp):
    if not SERVERCHAN_KEYS:
        logger.info("Server酱未配置，跳过推送")
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
            logger.warning(f"Server酱推送失败: {e}")
    logger.info(f"Server酱推送: {success_count}/{len(SERVERCHAN_KEYS)}")
    return success_count > 0


# ══════════════════════════════════════════════════════════
#  数据获取层（akshare优先 + tushare降级 + 并发 + 预筛选）
# ══════════════════════════════════════════════════════════

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")

def _get_pro():
    """延迟初始化tushare pro"""
    if TUSHARE_TOKEN is None:
        raise ValueError("TUSHARE_TOKEN环境变量未设置")
    return ts.pro_api(TUSHARE_TOKEN)


def get_all_a_stocks():
    """获取全市场A股列表 —— 优先akshare（无限流），备用tushare"""
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
        logger.warning(f"获取股票列表失败: {e}")
        return []


def get_stock_data(ts_code):
    """获取单只股票日线数据（带增量缓存，首次运行后大幅加速）"""
    return get_stock_data_cached(ts_code, min_rows=130)


def batch_prefilter_stocks():
    """用akshare批量获取全市场实时行情，快速预筛选

    过滤条件：
    - 排除ST股、退市股、停牌股、北交所
    - 排除非强势行业（仅保留当日行业涨跌幅排名前50%的行业）
    """
    try:
        import akshare as ak

        # ── 第一步：获取行业板块涨跌幅，确定强势行业 ──
        strong_industries = set()
        try:
            industry_df = ak.stock_board_industry_name_em()
            if industry_df is not None and len(industry_df) > 0:
                # 列名可能为: 板块名称, 涨跌幅
                change_col = None
                name_col = None
                for col in industry_df.columns:
                    if "涨跌幅" in str(col):
                        change_col = col
                    if "板块名称" in str(col) or "名称" in str(col):
                        name_col = col
                if change_col and name_col:
                    industry_df = industry_df.copy()
                    industry_df[change_col] = pd.to_numeric(industry_df[change_col], errors="coerce")
                    industry_df = industry_df.dropna(subset=[change_col])
                    # 取涨跌幅排名前50%的行业作为强势行业
                    median_change = industry_df[change_col].median()
                    strong = industry_df[industry_df[change_col] >= median_change]
                    strong_industries = set(strong[name_col].tolist())
                    logger.info(f"行业筛选: {len(strong_industries)}/{len(industry_df)}个行业为强势 "
                                f"(涨跌幅 >= {median_change:.2f}%)")
        except Exception as e:
            logger.warning(f"行业板块数据获取失败，跳过行业过滤: {e}")

        # ── 第二步：获取全市场实时行情 ──
        df = ak.stock_zh_a_spot_em()
        if df is None or len(df) == 0:
            return None
        original_count = len(df)

        # 排除ST、退市、停牌、北交所
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

        # ── 第三步：排除非强势行业 ──
        if strong_industries and "行业" in df.columns:
            before_count = len(df)
            df = df[df["行业"].isin(strong_industries)]
            removed = before_count - len(df)
            logger.info(f"行业过滤: 排除{removed}只非强势行业股票，保留{len(df)}只")
        elif strong_industries:
            # akshare的stock_zh_a_spot_em可能没有"行业"列，需要通过行业成分股反查
            try:
                strong_codes = set()
                for ind_name in strong_industries:
                    try:
                        cons = ak.stock_board_industry_cons_em(symbol=ind_name)
                        if cons is not None and len(cons) > 0:
                            code_col = None
                            for col in cons.columns:
                                if "代码" in str(col):
                                    code_col = col
                                    break
                            if code_col:
                                strong_codes.update(cons[code_col].astype(str).tolist())
                    except Exception:
                        continue
                if strong_codes:
                    before_count = len(df)
                    df = df[df["代码"].astype(str).isin(strong_codes)]
                    removed = before_count - len(df)
                    logger.info(f"行业过滤(成分股反查): 排除{removed}只，保留{len(df)}只")
            except Exception as e:
                logger.warning(f"行业成分股反查失败: {e}")

        logger.info(f"批量预筛选: {len(df)}只/{original_count}只 "
                    f"（排除ST/退市/停牌/北交所/非强势行业）")
        return df
    except Exception as e:
        logger.warning(f"批量预筛选失败: {e}")
        return None



def _fetch_and_process_one_core(ts_code, name, industry, best_params):
    """获取单只股票数据并计算指标的核心逻辑（不含重试和异常捕获）"""
    df = get_stock_data(ts_code)
    if df is None:
        return (ts_code, name, industry, None)
    df = IndicatorCalcBase(df)
    df = add_micro_confirm_indicators(df)
    df = add_inst_support_indicators(df, best_params)
    df = Detect_AmbushSignal_V64(df, best_params)
    if df is None or len(df) < 130:
        return (ts_code, name, industry, None)
    return (ts_code, name, industry, df)


def _fetch_and_process_one_with_retry(ts_code, name, industry, best_params):
    """带重试的单只股票处理（使用tenacity自动重试网络异常）"""
    if TENACITY_AVAILABLE:
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
               retry=retry_if_exception_type((ConnectionError, TimeoutError)))
        def _retry_wrapper():
            return _fetch_and_process_one_core(ts_code, name, industry, best_params)
        return _retry_wrapper()
    else:
        return _fetch_and_process_one_core(ts_code, name, industry, best_params)


def _fetch_and_process_one(ts_code, name, industry, best_params):
    """获取单只股票数据并计算指标，返回 (ts_code, name, industry, df_or_None)

    健壮性保证：
    - 单只股票计算崩溃仅记录ERROR日志，绝对不阻断其他股票
    - 对Detect_AmbushSignal_V64的调用包裹try...except
    - 网络异常自动重试3次（需安装tenacity）
    """
    try:
        return _fetch_and_process_one_with_retry(ts_code, name, industry, best_params)
    except Exception as e:
        logger.error(f"股票处理异常 {ts_code}({name}): {e}", exc_info=False)
        return (ts_code, name, industry, None)


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
            logger.warning("全市场活跃市值数据不足，回退到成交额代理")
            # 回退: 使用沪深300成交额代理
            pro = _get_pro()
            end_date = pd.Timestamp.now().strftime("%Y%m%d")
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime("%Y%m%d")
            index_df = pro.index_daily(ts_code="000300.SH", start_date=start_date, end_date=end_date)
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
            # 超额年化+8.12%, 回撤-9.2%, 仅31次切换
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
            d = state_df.index[-(i + 1)]
            s = int(state_df.loc[d, "oamv_state"])
            x = float(state_df.loc[d, "oamv_x"])
            recent_states.append({"date": d.strftime("%m-%d"), "state": "允许" if s == 1 else "禁止", "x": round(x, 2)})

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
        logger.warning(f"OAMV计算失败: {e}")
        return None


# ══════════════════════════════════════════════════════════
#  行业热度分析
# ══════════════════════════════════════════════════════════

def compute_industry_analysis(signals_data, industry_map):
    """行业间分析：计算每个行业的动量、信号数量、热度排名、轮动信号"""
    mom_days = BEST_PARAMS.get("industry_momentum_days", 10)
    mom_df = compute_industry_momentum(signals_data, industry_map, mom_days)

    if mom_df.empty or len(mom_df) < 6:
        return [], None

    latest_mom = mom_df.iloc[-1]
    lookback = min(5, len(mom_df) - 1)
    prev_mom = mom_df.iloc[-1 - lookback]

    industry_signal_count = defaultdict(int)
    industry_stock_count = defaultdict(int)
    for ts_code, df in signals_data.items():
        industry = industry_map.get(ts_code, "")
        if not industry:
            continue
        industry_stock_count[industry] += 1
        if len(df) > 0 and df.iloc[-1].get("ambush_signal", False):
            industry_signal_count[industry] += 1

    industry_stats = []
    for industry in mom_df.columns:
        momentum = float(latest_mom.get(industry, 0))
        prev_momentum = float(prev_mom.get(industry, 0))
        momentum_change = momentum - prev_momentum
        signal_count = industry_signal_count.get(industry, 0)
        stock_count = industry_stock_count.get(industry, 0)

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

    industry_stats.sort(key=lambda x: x["momentum"], reverse=True)

    # 构建行业允许买入矩阵
    mom_threshold = BEST_PARAMS.get("industry_momentum_threshold", 0.0)
    industry_allow_matrix = build_industry_allow_matrix(mom_df, mom_threshold)

    return industry_stats, industry_allow_matrix


# ══════════════════════════════════════════════════════════
#  信号详细分析
# ══════════════════════════════════════════════════════════

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
        recent_sos = df.iloc[signal_idx - window:signal_idx + 1]["tag_sos_anchor"].any()
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
    if upper_shadow > body * 1.5:
        vpa_parts.append("上影线压力(供应出现)")
    analysis["vpa"] = vpa_parts

    # 蜡烛图解读
    candle_parts = []
    close_pos = (row["Close"] - row["Low"]) / (amplitude + 1e-8)
    if body < amplitude * 0.1:
        candle_parts.append("十字星(方向选择)")
    elif body < amplitude * 0.3:
        if row["Close"] > row["Open"]:
            candle_parts.append("小阳线(温和看多)")
        else:
            candle_parts.append("小阴线(温和看空)")
    elif row["Close"] > row["Open"] and close_pos > 0.7:
        candle_parts.append("阳线收高(强势)")
    elif row["Close"] < row["Open"] and lower_shadow > body * 2:
        candle_parts.append("锤子线(潜在反转)")
    if not candle_parts:
        if row["Close"] >= row["Open"]:
            candle_parts.append("阳线")
        else:
            candle_parts.append("阴线")
    analysis["candle"] = candle_parts

    analysis["support"] = round(float(row["yellow_line"] - 0.5 * row["atr14"]), 2)
    analysis["resistance"] = round(float(row["yellow_line"] + 1.5 * row["atr14"]), 2)

    return analysis


# ══════════════════════════════════════════════════════════
#  全市场扫描
# ══════════════════════════════════════════════════════════

# 断点续传状态文件路径
SCAN_STATUS_FILE = RESULT_DIR / "scan_status.json"


def _load_scan_status():
    """加载断点续传状态：返回已完成的股票代码集合"""
    if SCAN_STATUS_FILE.exists():
        try:
            with open(SCAN_STATUS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            completed = set(data.get("completed", []))
            logger.info(f"断点续传: 发现{len(completed)}只已完成股票")
            return completed
        except Exception as e:
            logger.warning(f"断点续传状态加载失败: {e}")
    return set()


def _save_scan_status(completed_set):
    """保存断点续传状态"""
    try:
        SCAN_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SCAN_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump({"completed": list(completed_set)}, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"断点续传状态保存失败: {e}")


def _clear_scan_status():
    """扫描完成后删除状态文件"""
    try:
        if SCAN_STATUS_FILE.exists():
            SCAN_STATUS_FILE.unlink()
            logger.info("断点续传状态文件已清理")
    except Exception as e:
        logger.warning(f"断点续传状态文件清理失败: {e}")


def scan_market(max_stocks=None, industry_allow_matrix=None, industry_map=None, prefilter_df=None):
    """全市场扫描潜伏信号（并发获取数据 + 断点续传）"""
    all_stocks = get_all_a_stocks()
    if not all_stocks:
        logger.error("无法获取股票列表")
        return [], {}

    if max_stocks:
        all_stocks = all_stocks[:max_stocks]

    # 使用批量预筛选结果过滤股票列表
    if prefilter_df is not None:
        prefilter_codes = set(prefilter_df["ts_code"].tolist())
        original_count = len(all_stocks)
        all_stocks = [(tc, n, ind) for tc, n, ind in all_stocks if tc in prefilter_codes]
        logger.info(f"预筛选后股票数: {len(all_stocks)}/{original_count}")

    # 断点续传：跳过已完成的股票
    completed_set = _load_scan_status()
    if completed_set:
        before_count = len(all_stocks)
        all_stocks = [(tc, n, ind) for tc, n, ind in all_stocks if tc not in completed_set]
        logger.info(f"断点续传: 跳过{before_count - len(all_stocks)}只已完成股票，剩余{len(all_stocks)}只")

    total = len(all_stocks)
    logger.info(f"扫描股票数(预筛选后): {total} | 并发数: 10")

    signals = []
    all_signals_data = {}
    processed = 0
    errors = 0
    start_time = time.time()

    # 并发获取数据（10个线程）
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for ts_code, name, industry in all_stocks:
            future = executor.submit(_fetch_and_process_one, ts_code, name, industry, BEST_PARAMS)
            futures[future] = (ts_code, name, industry)

        for future in as_completed(futures):
            ts_code, name, industry = futures[future]
            processed += 1

            if processed % 200 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / processed * (total - processed) if processed > 0 else 0
                logger.info(f"进度: {processed}/{total} ({processed/total*100:.1f}%) | "
                            f"信号:{len(signals)} | 失败:{errors} | ETA:{eta:.0f}s")

            try:
                result_ts_code, result_name, result_industry, df = future.result()
            except Exception:
                errors += 1
                # 即使异常也标记为已完成，避免反复重试同一只股票
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

            if latest.get("ambush_signal", False):
                # V6.3：行业热度过滤
                signal_date = df.index[-1]
                ind_allowed = True
                if industry_allow_matrix is not None and industry and industry in industry_allow_matrix.columns:
                    try:
                        ind_val = industry_allow_matrix[industry].reindex([signal_date])
                        if not ind_val.empty and not ind_val.iloc[0]:
                            ind_allowed = False
                    except Exception:
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
                    # V6.4：主力托底评分
                    "inst_support_score": int(latest.get("inst_support_score", 0)),
                    "factor_a": bool(latest.get("factor_a_vol_stable", False)),
                    "factor_b": bool(latest.get("factor_b_vp_divergence", False)),
                    "factor_c": bool(latest.get("factor_c_support_hold", False)),
                    "factor_d": bool(latest.get("factor_d_intraday_accum", False)),
                }
                signals.append(signal_info)
                logger.info(f"潜伏信号: {name}({ts_code}) [{industry}] {latest['Close']:.2f} "
                            f"{change_pct:+.2f}% J:{latest['J']:.1f} 量比:{vol_ratio:.2f}")

    elapsed = time.time() - start_time
    logger.info(f"扫描完成! 耗时: {elapsed/60:.1f}min | 信号: {len(signals)}只 | 错误: {errors}")

    # 扫描完成，清理断点续传状态文件
    _clear_scan_status()

    return signals, all_signals_data


# ══════════════════════════════════════════════════════════
#  推送消息构建
# ══════════════════════════════════════════════════════════

def build_push_message(oamv_status, signals, industry_stats):
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

    title = f"{sentiment_icon} 量化潜伏·盘前 {today} | {len(signals)}信号 | {sentiment}"

    parts = []

    # ── 头部 ──
    parts.append(f"## 📊 量化潜伏 · 盘前扫描")
    parts.append(f"**{today}** | 市场情绪: **{sentiment}** {sentiment_icon}")
    parts.append("")
    parts.append("---")
    parts.append("")

    # ── 市场环境 ──
    parts.append("### 🌡️ 市场环境")
    if oamv_status:
        parts.append(f"- 活跃度趋势: **{oamv_status['trend_label']}**")
        parts.append(f"- 操作建议: **{'可积极布局' if can_open else '建议观望，控制仓位'}**")
        if oamv_status.get("last_transition"):
            lt = oamv_status["last_transition"]
            parts.append(f"- 趋势切换于 {lt['date']}")
    else:
        parts.append("- 环境评估中，暂默认允许操作")
    parts.append("")
    parts.append("---")
    parts.append("")

    # ── 行业风向 ──
    if industry_stats:
        parts.append("### 🏭 行业风向")

        # 热门行业 Top8
        hot = [s for s in industry_stats if s["momentum"] > 0]
        hot_count = len(hot)
        cold_count = len(industry_stats) - hot_count
        if hot:
            parts.append(f"**强势行业 Top8** ({hot_count}个偏强)")
            for s in hot[:8]:
                icon = {"火热": "🔥", "偏热": "🟠", "微热": "🟡"}.get(s["hot_cold"], "⚪")
                arrow = "↑" if s["momentum_change"] > 0 else "↓" if s["momentum_change"] < 0 else "→"
                sig_mark = f" ✦{s['signal_count']}信号" if s["signal_count"] > 0 else ""
                parts.append(f"- {icon} {s['name']} {s['momentum']:+.2f}% {arrow}{sig_mark}")
            parts.append("")

        # 轮动信号
        rotation_in = [s for s in industry_stats if s["rotation"] == "轮入"]
        rotation_out = [s for s in industry_stats if s["rotation"] == "轮出"]
        warming = [s for s in industry_stats if s["rotation"] == "回暖"]

        if rotation_in or rotation_out:
            parts.append("**资金动向**")
            if rotation_in:
                names = "、".join(s["name"] for s in rotation_in[:6])
                parts.append(f"- 🔄 轮入: {names}")
            if rotation_out:
                names = "、".join(s["name"] for s in rotation_out[:5])
                parts.append(f"- ⚠️ 轮出: {names}")
            if warming:
                names = "、".join(s["name"] for s in warming[:4])
                parts.append(f"- 🌱 回暖: {names}")
            parts.append("")

        # 弱势行业
        cold = [s for s in industry_stats if s["momentum"] <= 0]
        if cold:
            parts.append(f"**弱势行业** ({cold_count}个偏弱)")
            for s in cold[-5:]:
                icon = {"冰冷": "❄️", "偏冷": "🔵", "微冷": "🔷"}.get(s["hot_cold"], "⚪")
                parts.append(f"- {icon} {s['name']} {s['momentum']:+.2f}%")
            parts.append("")

        parts.append("---")
        parts.append("")

    # ── 潜伏信号 ──
    parts.append("### 🎯 潜伏信号")
    if not can_open:
        parts.append("> ⚠️ 当前环境偏弱，以下标的仅供跟踪观察")
        parts.append("")

    if signals:
        parts.append(f"今日筛选 **{len(signals)}** 只:")
        parts.append("")
        for i, s in enumerate(signals, 1):
            score = s.get('inst_support_score', 0)
            score_bar = '★' * score + '☆' * (3 - score)
            factors = []
            if s.get('factor_b'):
                factors.append('量价背离')
            if s.get('factor_c'):
                factors.append('支撑不破')
            if s.get('factor_d'):
                factors.append('日内承接')
            factor_str = '+'.join(factors) if factors else '无'
            parts.append(f"**{i}. {s['name']}** ({s['code']}) [{s['industry']}]")
            parts.append(f"  收盘 {s['price']:.2f} | {s['change_pct']:+.2f}% | 量比 {s['vol_ratio']:.2f}")
            parts.append(f"  托底评分 {score_bar} ({score}/3) | {factor_str}")
            parts.append("")
    else:
        if can_open:
            parts.append("今日无符合条件的标的")
        else:
            parts.append("环境偏弱，暂无值得关注的标的")
        parts.append("")

    parts.append("---")
    parts.append("")

    # ── 页脚 ──
    parts.append("*量化潜伏系统 · 多维度量化筛选*")
    parts.append("*以上内容为系统量化输出，不构成投资建议，据此操作风险自担*")

    desp = "\n".join(parts)
    return title, desp


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def daily_push(max_stocks=None):
    logger.info("=" * 60)
    logger.info("潜伏模型V6.4.5 盘前扫描推送 (Mootdx版+主力托底)")
    logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 0. Mootdx测速优选
    logger.info("[0/4] Mootdx测速优选...")
    mootdx_bestip()

    # 1. OAMV择时
    logger.info("[1/4] 计算OAMV活跃市值择时...")
    oamv_status = get_oamv_status()
    if oamv_status:
        logger.info(f"  择时状态: {'允许' if oamv_status['can_open_position'] else '禁止'} | "
                    f"OAMV={oamv_status['latest_x']} | {oamv_status['trend_label']}")

    # 2. 获取行业分类
    logger.info("[2/4] 获取行业分类...")
    try:
        basic = _get_pro().stock_basic(fields="ts_code,industry", list_status="L")
        industry_map = dict(zip(basic["ts_code"], basic["industry"]))
        logger.info(f"  行业映射: {len(industry_map)}只股票")
    except Exception as e:
        logger.warning(f"  行业分类获取失败: {e}")
        industry_map = {}

    # 3. 全市场扫描
    logger.info("[3/4] 全市场扫描潜伏信号...")
    cache_stats = get_cache_stats()
    logger.info(f"  股票缓存: {cache_stats.get('count', 0)}只 | {cache_stats.get('size_mb', 0)}MB")
    logger.info("  批量预筛选全市场行情...")
    prefilter_df = batch_prefilter_stocks()
    signals, all_signals_data = scan_market(
        max_stocks=max_stocks,
        industry_allow_matrix=None,
        industry_map=industry_map,
        prefilter_df=prefilter_df,
    )

    # 4. 行业热度分析 + 行业过滤
    logger.info("[4/4] 行业热度分析...")
    industry_stats = []
    if all_signals_data and industry_map:
        industry_stats, industry_allow_matrix = compute_industry_analysis(
            all_signals_data, industry_map
        )
        logger.info(f"  行业数: {len(industry_stats)}")

        # 用行业过滤重新筛选信号
        if industry_allow_matrix is not None:
            filtered_signals = []
            for s in signals:
                industry = s.get("industry", "")
                try:
                    signal_date = pd.Timestamp(s["signal_date"])
                    ind_val = industry_allow_matrix[industry].reindex([signal_date])
                    if not ind_val.empty and not ind_val.iloc[0]:
                        logger.info(f"  行业过滤: {s['name']} 行业={industry} 动量不足")
                        continue
                except Exception:
                    pass
                filtered_signals.append(s)
            removed = len(signals) - len(filtered_signals)
            if removed > 0:
                logger.info(f"  行业过滤: {len(signals)}只 -> {len(filtered_signals)}只 (过滤{removed}只)")
            signals = filtered_signals

    # 构建推送消息
    logger.info("构建推送消息...")
    title, desp = build_push_message(oamv_status, signals, industry_stats)

    # 保存结果
    result = {
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "V6.4.5",
        "data_source": "akshare",
        "oamv_status": oamv_status,
        "signal_count": len(signals),
        "signals": signals,
        "industry_stats": industry_stats[:30],
    }
    result_file = RESULT_DIR / f"mootdx_push_{datetime.now().strftime('%Y%m%d')}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"结果已保存: {result_file}")

    # 推送
    logger.info("推送微信...")
    send_serverchan(title, desp)
    logger.info("推送完成")

    # 摘要
    logger.info(f"OAMV择时: {'允许' if oamv_status and oamv_status['can_open_position'] else '禁止'}")
    logger.info(f"潜伏信号: {len(signals)}只")
    for s in signals:
        logger.info(f"  - {s['name']}({s['code']}) [{s['industry']}] {s['price']:.2f} J:{s['J']:.1f}")
    if industry_stats:
        hot = [s for s in industry_stats if s["momentum"] > 0]
        rot_in = [s for s in industry_stats if s["rotation"] == "轮入"]
        logger.info(f"行业热度: {len(hot)}个偏热 / {len(industry_stats)-len(hot)}个偏冷 | 轮入:{len(rot_in)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="潜伏模型V6.4.5 盘前扫描推送 (Mootdx版+主力托底)")
    parser.add_argument("--test", type=int, default=None, help="测试模式：仅扫描前N只股票")
    args = parser.parse_args()

    daily_push(max_stocks=args.test)
