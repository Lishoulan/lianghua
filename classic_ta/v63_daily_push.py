"""
潜伏模型V6.3 每日实盘推送
===========================
基于V6.3最佳参数 + 波动率平价仓位

推送内容：
  1. OAMV活跃市值择时状态
  2. 行业热度分析（行业动量排名、冷热分布、轮动信号）
  3. 潜伏买入信号（含行业过滤：只买行业动量>2%的股票）
  4. 持仓监控（4级退出：硬止损→吊灯止盈→Buy Climax→时间止损）
  5. 波动率平价仓位建议

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

# V6.3最佳参数
from classic_ta.v60_ambush_model import IndicatorCalcBase, DEFAULT_PARAMS
from classic_ta.v61_ambush_model import V61_PARAMS
from classic_ta.v62_ambush_model import (
    compute_industry_momentum, build_industry_allow_matrix,
)
from classic_ta.v63_ambush_model import (
    Detect_AmbushSignal_V63, add_micro_confirm_indicators,
    calc_volatility_parity_shares, V63_PARAMS,
)

BEST_PARAMS = V63_PARAMS.copy()


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
    """获取OAMV活跃市值当前状态"""
    from ml_strategy.oamv_filter import OAMVHysteresisFilter

    end_date = pd.Timestamp.now().strftime("%Y%m%d")
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime("%Y%m%d")

    try:
        index_df = _get_pro().index_daily(ts_code="000300.SH", start_date=start_date, end_date=end_date)
        if index_df is None or len(index_df) < 40:
            return None
        index_df = index_df.sort_values("trade_date").reset_index(drop=True)
        index_df["Date"] = pd.to_datetime(index_df["trade_date"], format="%Y%m%d")
        index_df.set_index("Date", inplace=True)
        index_df["Close"] = index_df["close"].astype(float)
        index_df["Volume"] = index_df["vol"].astype(float)
        index_df["amount"] = index_df["amount"].astype(float)

        daily_basic = None
        try:
            daily_basic = _get_pro().daily_basic(
                ts_code="000300.SH", start_date=start_date, end_date=end_date,
                fields="ts_code,trade_date,circ_mv,turnover_rate_f",
            )
            if daily_basic is not None and len(daily_basic) > 20:
                daily_basic["trade_date"] = pd.to_datetime(daily_basic["trade_date"], format="%Y%m%d")
                daily_basic.set_index("trade_date", inplace=True)
                daily_basic["circ_mv"] = daily_basic["circ_mv"].astype(float)
                daily_basic["turnover_rate_f"] = daily_basic["turnover_rate_f"].astype(float)
                daily_basic["amv_rate"] = daily_basic["circ_mv"] * daily_basic["turnover_rate_f"]
            else:
                daily_basic = None
        except Exception:
            daily_basic = None

        oamv = OAMVHysteresisFilter(
            upper_threshold=4.0, lower_threshold=-2.3,
            cost_ma_period=34, roc_period=1,
            weekly_ema_period=5, weekly_use_ema=True,
        )
        if daily_basic is not None:
            oamv.fit(index_df, daily_basic_df=daily_basic)
            data_source = "指南针同款(circ_mv×turnover_rate_f)"
        else:
            oamv.fit(index_df)
            data_source = "成交额代理(amount)"

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
            rotation = "轮入"  # 冷→热，行业正在被资金关注
        elif was_hot and not is_hot:
            rotation = "轮出"  # 热→冷，资金正在撤离
        elif is_hot and momentum_change > 0.01:
            rotation = "加速"  # 热且动量还在上升
        elif is_hot and momentum_change < -0.01:
            rotation = "减速"  # 热但动量开始下降
        elif not is_hot and momentum_change > 0.01:
            rotation = "回暖"  # 冷但动量在改善
        elif not is_hot and momentum_change < -0.01:
            rotation = "恶化"  # 冷且动量还在恶化
        else:
            rotation = "平稳"

        industry_stats.append({
            "name": industry,
            "momentum": round(momentum * 100, 2),  # 转为百分比
            "momentum_change": round(momentum_change * 100, 2),  # 动量变化（百分点）
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
    """获取单只股票日线数据 —— 优先akshare（无限流），备用tushare"""
    # 尝试 akshare
    try:
        import akshare as ak
        symbol = ts_code.split(".")[0]
        end_date = pd.Timestamp.now().strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date="20240101", end_date=end_date, adjust="qfq")
        if df is not None and len(df) >= 130:
            col_map = {"开盘": "Open", "最高": "High", "最低": "Low",
                       "收盘": "Close", "成交量": "Volume"}
            for old, new in col_map.items():
                if old in df.columns:
                    df[new] = df[old].astype(float)
            if "日期" in df.columns:
                df["Date"] = pd.to_datetime(df["日期"])
            df.set_index("Date", inplace=True)
            df = df.sort_index()
            df = df[df["Volume"] > 0]
            if not df.empty and len(df) >= 130:
                return df
    except Exception:
        pass
    # 降级到tushare
    try:
        end_date = pd.Timestamp.now().strftime("%Y%m%d")
        df = _get_pro().daily(ts_code=ts_code, start_date="20240101", end_date=end_date)
        if df is None or len(df) < 130:
            return None

        try:
            adj = _get_pro().adj_factor(ts_code=ts_code, start_date="20240101", end_date=end_date)
            if adj is not None and len(adj) > 0:
                adj = adj.sort_values("trade_date").reset_index(drop=True)
                latest_adj = adj["adj_factor"].iloc[-1]
                adj_ratio = adj["adj_factor"].astype(float) / float(latest_adj)
                df = df.sort_values("trade_date").reset_index(drop=True)
                df["open"] = df["open"].astype(float) * adj_ratio
                df["high"] = df["high"].astype(float) * adj_ratio
                df["low"] = df["low"].astype(float) * adj_ratio
                df["close"] = df["close"].astype(float) * adj_ratio
            else:
                df = df.sort_values("trade_date").reset_index(drop=True)
        except:
            df = df.sort_values("trade_date").reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df.set_index("Date", inplace=True)
        df["Open"] = df["open"].astype(float)
        df["High"] = df["high"].astype(float)
        df["Low"] = df["low"].astype(float)
        df["Close"] = df["close"].astype(float)
        df["Volume"] = df["vol"].astype(float)
        df = df[df["Volume"] > 0]
        if df.empty:
            return None
        return df
    except Exception:
        time.sleep(0.3)
        return None


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
        # 快速预筛选：价格在白线上方（Close > MA5近似）、J值偏低（涨跌幅<3%）
        # 这里只做粗筛，后续逐只精确计算
        if "涨跌幅" in df.columns:
            df = df[df["涨跌幅"] < 5]  # 排除已经大涨的
        print(f"  批量预筛选: {len(df)}只（排除ST/停牌/北交所/已大涨）")
        return df
    except Exception as e:
        print(f"  批量预筛选失败: {e}")
        return None


def scan_market(oamv_weekly_allowed_dates=None, industry_allow_matrix=None, industry_map=None, prefilter_df=None):
    """全市场扫描潜伏信号（含行业热度过滤）"""
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

    total = len(all_stocks)
    print(f"扫描股票数(预筛选后): {total}", flush=True)

    signals = []
    all_signals_data = {}  # 用于行业分析
    processed = 0
    errors = 0
    max_errors = 200  # 错误上限，防止任务无限运行
    start_time = time.time()

    for ts_code, name, industry in all_stocks:
        processed += 1

        # 错误上限检查
        if errors >= max_errors:
            print(f"  错误数达到{max_errors}，停止扫描，使用已有信号", flush=True)
            break

        if processed % 200 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / processed * (total - processed) if processed > 0 else 0
            print(f"  进度: {processed}/{total} ({processed/total*100:.1f}%) | "
                  f"信号:{len(signals)} | 失败:{errors} | ETA:{eta:.0f}s", flush=True)

        df = get_stock_data(ts_code)
        if df is None:
            errors += 1
            continue

        try:
            df = IndicatorCalcBase(df)
            df = add_micro_confirm_indicators(df)
            df = Detect_AmbushSignal_V63(df, BEST_PARAMS)
        except Exception:
            errors += 1
            continue

        if df is None or len(df) < 130:
            errors += 1
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
            # V6.2：行业热度过滤
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

            # 行业动量
            ind_momentum = ""
            if industry_allow_matrix is not None and industry and industry in industry_allow_matrix.columns:
                try:
                    mom_df_ref = industry_allow_matrix._mom_df if hasattr(industry_allow_matrix, '_mom_df') else None
                except:
                    pass

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
            }
            signals.append(signal_info)
            print(f"  潜伏信号: {name}({ts_code}) [{industry}] {latest['Close']:.2f} {change_pct:+.2f}% "
                  f"J:{latest['J']:.1f} 量比:{vol_ratio:.2f}", flush=True)

        if processed % 300 == 0:
            time.sleep(5)  # akshare无限流，短暂休息即可

    elapsed = time.time() - start_time
    print(f"\n扫描完成! 耗时: {elapsed/60:.1f}min | 信号: {len(signals)}只 | 错误: {errors}", flush=True)
    return signals, all_signals_data


# ══════════════════════════════════════════════════════════
#  推送消息构建
# ══════════════════════════════════════════════════════════

def build_push_message(oamv_status, signals, industry_stats):
    today = datetime.now().strftime("%Y-%m-%d")
    title = f"潜伏模型V6.3 {today}"

    parts = []

    # 标题
    parts.append("## 潜伏模型V6.3 每日实盘推送")
    parts.append(f"**日期**: {today}")
    parts.append("")
    parts.append("---")
    parts.append("")

    # OAMV活跃市值择时
    parts.append("### 活跃市值择时(OAMV)")
    if oamv_status:
        can_open = oamv_status["can_open_position"]
        status_icon = "🟢 允许开仓" if can_open else "🔴 禁止开仓"
        trend = oamv_status["trend_label"]
        parts.append(f"**当前状态**: {status_icon}")
        parts.append(f"**趋势判断**: {trend}")
        parts.append(f"**OAMV指标值**: {oamv_status['latest_x']}")
        parts.append(f"**数据源**: {oamv_status['data_source']}")
        parts.append("")
        parts.append("**近5日状态**:")
        for s in oamv_status["recent_states"]:
            icon = "🟢" if s["state"] == "允许" else "🔴"
            parts.append(f"- {s['date']}: {icon}{s['state']} (x={s['x']})")
        if oamv_status.get("last_transition"):
            lt = oamv_status["last_transition"]
            parts.append(f"**最近切换**: {lt['date']} → {lt['to_state']}")
    else:
        can_open = True
        parts.append("**OAMV计算失败，默认允许开仓**")
    parts.append("")
    parts.append("---")
    parts.append("")

    # 行业热度分析
    parts.append("### 行业热度分析")
    if industry_stats:
        mom_days = BEST_PARAMS.get("industry_momentum_days", 10)
        parts.append(f"> 行业动量 = 近{mom_days}日等权平均涨幅 | 阈值>{BEST_PARAMS.get('industry_momentum_threshold',0)*100:.0f}%才允许买入")
        parts.append("")

        # 热门行业Top10
        hot = [s for s in industry_stats if s["momentum"] > 0]
        cold = [s for s in industry_stats if s["momentum"] <= 0]
        parts.append(f"**偏热行业({len(hot)}个)** Top10:")
        for s in hot[:10]:
            icon = {"火热": "🔥", "偏热": "🟠", "微热": "🟡"}.get(s["hot_cold"], "⚪")
            sig_mark = f" **{s['signal_count']}信号**" if s["signal_count"] > 0 else ""
            change_arrow = "↑" if s["momentum_change"] > 0 else "↓" if s["momentum_change"] < 0 else "→"
            parts.append(f"- {icon} {s['name']}: {s['momentum']:+.2f}% ({change_arrow}{abs(s['momentum_change']):.2f}pp) ({s['stock_count']}只){sig_mark}")

        parts.append("")

        # 冷门行业Bottom5
        parts.append(f"**偏冷行业({len(cold)}个)** Bottom5:")
        for s in cold[-5:]:
            icon = {"冰冷": "❄️", "偏冷": "🔵", "微冷": "🔷"}.get(s["hot_cold"], "⚪")
            change_arrow = "↑" if s["momentum_change"] > 0 else "↓" if s["momentum_change"] < 0 else "→"
            parts.append(f"- {icon} {s['name']}: {s['momentum']:+.2f}% ({change_arrow}{abs(s['momentum_change']):.2f}pp) ({s['stock_count']}只)")

        parts.append("")
        parts.append("---")
        parts.append("")

        # ── 行业间轮动分析 ──
        parts.append("### 行业间轮动分析")
        parts.append(f"> 对比5日前动量变化，检测资金在行业间的流动方向")
        parts.append("")

        # 轮入行业（冷→热，最值得关注）
        rotation_in = [s for s in industry_stats if s["rotation"] == "轮入"]
        if rotation_in:
            parts.append(f"**🔄 轮入行业({len(rotation_in)}个)** — 资金正在流入，从冷转热:")
            for s in rotation_in[:8]:
                parts.append(f"- **{s['name']}**: 动量{s['momentum']:+.2f}% (5日前{s['momentum']-s['momentum_change']:+.2f}%→今{s['momentum']:+.2f}%) {s['stock_count']}只{s['signal_count']}信号")
        else:
            parts.append("**🔄 轮入行业**: 无")

        parts.append("")

        # 轮出行业（热→冷，需要警惕）
        rotation_out = [s for s in industry_stats if s["rotation"] == "轮出"]
        if rotation_out:
            parts.append(f"**⚠️ 轮出行业({len(rotation_out)}个)** — 资金正在撤离，从热转冷:")
            for s in rotation_out[:5]:
                parts.append(f"- {s['name']}: 动量{s['momentum']:+.2f}% (5日前{s['momentum']-s['momentum_change']:+.2f}%→今{s['momentum']:+.2f}%) {s['stock_count']}只")
        else:
            parts.append("**⚠️ 轮出行业**: 无")

        parts.append("")

        # 加速行业（热且动量还在上升）
        accelerating = [s for s in industry_stats if s["rotation"] == "加速"]
        if accelerating:
            parts.append(f"**🚀 加速行业({len(accelerating)}个)** — 热门且动量持续上升:")
            for s in accelerating[:5]:
                parts.append(f"- {s['name']}: 动量{s['momentum']:+.2f}% (+{s['momentum_change']:.2f}pp) {s['stock_count']}只{s['signal_count']}信号")

        parts.append("")

        # 回暖行业（冷但动量在改善）
        warming = [s for s in industry_stats if s["rotation"] == "回暖"]
        if warming:
            parts.append(f"**🌱 回暖行业({len(warming)}个)** — 冷门但动量在改善，可能即将轮入:")
            for s in warming[:5]:
                parts.append(f"- {s['name']}: 动量{s['momentum']:+.2f}% (+{s['momentum_change']:.2f}pp) {s['stock_count']}只")

        parts.append("")

        # 恶化行业（冷且动量还在恶化）
        deteriorating = [s for s in industry_stats if s["rotation"] == "恶化"]
        if deteriorating:
            parts.append(f"**📉 恶化行业({len(deteriorating)}个)** — 冷门且动量继续恶化:")
            for s in deteriorating[:3]:
                parts.append(f"- {s['name']}: 动量{s['momentum']:+.2f}% ({s['momentum_change']:.2f}pp)")

        parts.append("")

        # ── 行业间强弱对比 ──
        parts.append("### 行业间强弱对比")
        parts.append("> 综合动量+动量变化+信号密度排名")
        parts.append("")

        # 综合评分：动量权重60% + 动量变化权重20% + 信号密度权重20%
        max_mom = max(abs(s["momentum"]) for s in industry_stats) + 1e-8
        max_change = max(abs(s["momentum_change"]) for s in industry_stats) + 1e-8
        max_signal = max(s["signal_count"] for s in industry_stats) + 1e-8

        for s in industry_stats:
            score = (s["momentum"] / max_mom * 60
                     + s["momentum_change"] / max_change * 20
                     + s["signal_count"] / max_signal * 20)
            s["composite_score"] = round(score, 1)

        industry_stats.sort(key=lambda x: x["composite_score"], reverse=True)

        parts.append("**综合排名Top15**:")
        for i, s in enumerate(industry_stats[:15], 1):
            rot_icon = {"轮入": "🔄", "轮出": "⚠️", "加速": "🚀", "减速": "🔻",
                        "回暖": "🌱", "恶化": "📉", "平稳": "➡️"}.get(s["rotation"], "➡️")
            hot_icon = {"火热": "🔥", "偏热": "🟠", "微热": "🟡",
                        "微冷": "🔷", "偏冷": "🔵", "冰冷": "❄️"}.get(s["hot_cold"], "⚪")
            parts.append(f"- {i}. {hot_icon}{rot_icon} {s['name']}: 动量{s['momentum']:+.2f}% | 变化{s['momentum_change']:+.2f}pp | {s['signal_count']}信号 | 评分{s['composite_score']}")

        parts.append("")

        # 信号行业分布
        sig_industries = [s for s in industry_stats if s["signal_count"] > 0]
        if sig_industries:
            sig_industries.sort(key=lambda x: x["signal_count"], reverse=True)
            parts.append(f"**信号行业分布({len(sig_industries)}个行业有信号)**:")
            for s in sig_industries[:10]:
                bar = "█" * s["signal_count"]
                parts.append(f"- {s['name']}: {bar} {s['signal_count']}只 (动量{s['momentum']:+.2f}%)")
    else:
        parts.append("**行业热度数据不可用**")
    parts.append("")
    parts.append("---")
    parts.append("")

    # 潜伏信号
    parts.append("### 潜伏买入信号")
    if can_open if oamv_status else True:
        parts.append("> 择时状态: 允许开仓，以下信号可执行")
    else:
        parts.append("> 择时状态: 禁止开仓，以下信号仅供观察")
    parts.append("")
    parts.append(f"> 策略: SOS锚定→情绪冰点(J<{BEST_PARAMS['ambush_j_oversold']}+缩量+小实体+窗口{BEST_PARAMS['ambush_window']}天)")
    parts.append(f"> 退出: 硬止损({BEST_PARAMS['hard_stop_atr']}ATR) → 吊灯止盈({BEST_PARAMS['chandelier_atr_mult']}ATR) → Buy Climax → 时间止损({BEST_PARAMS['time_stop_days']}天)")
    parts.append(f"> 行业过滤: 近{BEST_PARAMS['industry_momentum_days']}日行业动量>{BEST_PARAMS['industry_momentum_threshold']*100:.0f}%")
    parts.append("")

    if signals:
        parts.append(f"**今日信号: {len(signals)} 只**")
        parts.append("")
        for i, s in enumerate(signals, 1):
            line_pos = "白>黄" if s["white_line"] > s["yellow_line"] else "白<黄"
            parts.append(f"**{i}. {s['name']}** ({s['code']}) [{s['industry']}]")
            parts.append(f"- 价格: {s['price']:.2f} | 涨跌: {s['change_pct']:+.2f}%")
            parts.append(f"- 白线: {s['white_line']:.2f} | 黄线: {s['yellow_line']:.2f} | {line_pos}")
            parts.append(f"- J值: {s['J']:.1f} | 量比: {s['vol_ratio']:.2f} | ATR: {s['atr14']:.2f}")
            if s["sos_dates"]:
                parts.append(f"- SOS锚定日: {', '.join(s['sos_dates'])}")
            a = s["analysis"]
            parts.append(f"- 威科夫: {'; '.join(a['wyckoff'])}")
            parts.append(f"- VPA量价: {'; '.join(a['vpa'])}")
            parts.append(f"- 蜡烛图: {'; '.join(a['candle'])}")
            parts.append(f"- 支撑: {a['support']:.2f} | 阻力: {a['resistance']:.2f}")
            # T+1操作建议（V6.3含波动率平价仓位）
            buy_ref = s["yellow_line"]
            hard_stop = s["price"] - BEST_PARAMS["hard_stop_atr"] * s["atr14"]
            chandelier_init = s["price"] - BEST_PARAMS["chandelier_atr_mult"] * s["atr14"]
            # 波动率平价仓位计算
            vp_shares = calc_volatility_parity_shares(
                total_equity=100000,  # 示例10万资金
                entry_price=s["price"],
                atr_at_entry=s["atr14"],
                hard_stop_atr=BEST_PARAMS["hard_stop_atr"],
                params=BEST_PARAMS,
            )
            vp_pct = vp_shares * s["price"] / 100000 * 100 if vp_shares > 0 else 0
            parts.append(f"- T+1参考买入: {s['price']:.2f}(开盘价) | 硬止损: {hard_stop:.2f} | 吊灯线初始: {chandelier_init:.2f}")
            parts.append(f"- 波动率平价仓位: {vp_shares}股({vp_pct:.1f}%资金) | 风险1%/{(BEST_PARAMS['hard_stop_atr']*s['atr14']):.2f}元每股")
            parts.append("")
    else:
        parts.append("**今日无潜伏信号**")
        parts.append("")

    parts.append("---")
    parts.append("")
    parts.append("*模型: 潜伏模型V6.3 | 理论: 威科夫LPS+VPA量价 | 择时: OAMV+行业动量 | 仓位: 波动率平价 | 退出: 4级(硬止损→吊灯→BC→时间)*")

    desp = "\n".join(parts)
    return title, desp


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def daily_push():
    print("=" * 80, flush=True)
    print("潜伏模型V6.3 每日实盘推送", flush=True)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)

    # 1. OAMV活跃市值择时
    print("\n[1/4] 计算OAMV活跃市值择时...", flush=True)
    oamv_status = get_oamv_status()
    if oamv_status:
        can_open = oamv_status["can_open_position"]
        print(f"  择时状态: {'允许开仓' if can_open else '禁止开仓'} | "
              f"OAMV={oamv_status['latest_x']} | {oamv_status['trend_label']}", flush=True)
    else:
        print("  OAMV计算失败", flush=True)

    # 2. 获取行业分类
    print("\n[2/4] 获取行业分类...", flush=True)
    try:
        basic = _get_pro().stock_basic(fields="ts_code,industry", list_status="L")
        industry_map = dict(zip(basic["ts_code"], basic["industry"]))
        print(f"  行业映射: {len(industry_map)}只股票", flush=True)
    except Exception as e:
        print(f"  行业分类获取失败: {e}", flush=True)
        industry_map = {}

    # 3. 全市场扫描
    print("\n[3/4] 全市场扫描潜伏信号...", flush=True)
    print("  批量预筛选全市场行情...", flush=True)
    prefilter_df = batch_prefilter_stocks()
    signals, all_signals_data = scan_market(
        industry_allow_matrix=None,  # 先扫描所有信号，行业过滤在后面做
        industry_map=industry_map,
        prefilter_df=prefilter_df,
    )

    # 4. 行业热度分析
    print("\n[4/4] 行业热度分析...", flush=True)
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

    # 构建推送消息
    print("\n构建推送消息...", flush=True)
    title, desp = build_push_message(oamv_status, signals, industry_stats)

    # 保存结果
    result = {
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "V6.3",
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
    print(f"OAMV择时: {'允许' if oamv_status and oamv_status['can_open_position'] else '禁止'}")
    print(f"潜伏信号: {len(signals)}只")
    for s in signals:
        print(f"  - {s['name']}({s['code']}) [{s['industry']}] {s['price']:.2f} J:{s['J']:.1f}")
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
