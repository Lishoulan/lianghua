"""
潜伏模型 V6.2 —— V6.1 + P2行业热度过滤
=======================================
基于V6.1最佳参数，实施P2级优化：

  P2-维度三：行业热度过滤（Industry Momentum Filter）
    核心思想：只买入处于上涨趋势行业中的股票
    实现：
      1. 用stock_basic.industry获取每只股票的行业分类（110个行业）
      2. 每个交易日，计算每个行业近N日的等权平均涨幅（行业动量）
      3. 只买入行业动量 > threshold的股票
    原理：威科夫理论中，SOS需要行业背景支撑
          行业上涨 = 资金流入 = 主力建仓环境
          行业下跌 = 资金流出 = 个股SOS可能是假突破
    预期效果：胜率+2-3%，减少在下跌行业中接飞刀
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
from collections import defaultdict

from classic_ta.v60_ambush_model import IndicatorCalcBase, DEFAULT_PARAMS
from classic_ta.v61_ambush_model import (
    Position, TradeRecord, ExitReason,
    Detect_AmbushSignal_V61, detect_buy_climax_v61,
    StatefulTradeBacktester_V61, compute_v61_metrics, V61_PARAMS,
)


# ╔══════════════════════════════════════════════════════════════╗
# ║  V6.2 参数                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

V62_PARAMS = V61_PARAMS.copy()
V62_PARAMS.update({
    # --- V6.1最佳参数（固定）---
    "spring_test_enabled": False,
    "chandelier_atr_mult": 2.5,
    "hard_stop_atr": 2.5,
    "time_stop_days": 8,
    "ambush_j_oversold": 18,
    "ambush_window": 12,

    # --- P2：行业热度过滤参数 ---
    "industry_filter_enabled": True,       # 是否启用行业过滤
    "industry_momentum_days": 20,          # 行业动量回看天数
    "industry_momentum_threshold": 0.0,    # 行业动量阈值（0=正收益才买）
})


# ╔══════════════════════════════════════════════════════════════╗
# ║  行业动量计算                                               ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_industry_momentum(signals_cache: Dict, industry_map: Dict[str, str],
                               momentum_days: int = 20) -> pd.DataFrame:
    """
    计算每个行业每天的动量值

    参数:
      signals_cache: {ts_code: df} 预计算好的指标+信号数据
      industry_map: {ts_code: industry_name} 股票→行业映射
      momentum_days: 回看天数

    返回:
      DataFrame: index=Date, columns=行业名, values=近N日等权平均涨幅
    """
    # 收集每个行业所有股票的日收益率
    industry_returns = defaultdict(list)  # {industry: [series1, series2, ...]}

    for ts_code, df in signals_cache.items():
        industry = industry_map.get(ts_code, "")
        if not industry:
            continue
        if "Close" not in df.columns:
            continue
        daily_ret = df["Close"].pct_change()
        daily_ret.name = ts_code
        industry_returns[industry].append(daily_ret)

    # 对每个行业计算等权平均涨幅
    industry_momentum = {}
    for industry, ret_list in industry_returns.items():
        if len(ret_list) < 3:  # 至少3只股票才算行业
            continue
        # 合并所有股票的日收益率
        combined = pd.concat(ret_list, axis=1)
        # 等权平均日收益率
        avg_daily_ret = combined.mean(axis=1)
        # 近N日累计涨幅
        momentum = avg_daily_ret.rolling(momentum_days, min_periods=max(momentum_days//2, 5)).sum()
        industry_momentum[industry] = momentum

    if not industry_momentum:
        return pd.DataFrame()

    mom_df = pd.DataFrame(industry_momentum)
    return mom_df


def build_industry_allow_matrix(mom_df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    根据行业动量构建允许买入矩阵

    返回:
      DataFrame: index=Date, columns=行业名, values=bool（True=允许买入）
    """
    return mom_df > threshold


# ╔══════════════════════════════════════════════════════════════╗
# ║  V6.2 状态机回测（行业热度过滤）                              ║
# ╚══════════════════════════════════════════════════════════════╝

def StatefulTradeBacktester_V62(
    df: pd.DataFrame,
    signal_col: str = "ambush_signal",
    initial_cash: float = 100000.0,
    params: Dict[str, Any] = None,
    market_allow_buy=None,
    ts_code: str = "",
    industry_allow_buy=None,  # V6.2新增：行业允许买入Series（index=Date, values=bool）
) -> List[TradeRecord]:
    """
    V6.2 状态机回测 —— V6.1 + 行业热度过滤

    与V6.1的关键区别：
      1. 追加行业热度过滤：只在行业动量>阈值时买入
    """
    if params is None:
        params = V62_PARAMS

    cash = initial_cash
    position = None
    trades = []
    pending_signal_idx = None

    # 预计算行业允许买入的对齐索引
    ind_allow_aligned = None
    if industry_allow_buy is not None:
        common_idx = df.index.intersection(industry_allow_buy.index)
        if len(common_idx) > 0:
            ind_allow_aligned = industry_allow_buy.reindex(common_idx)

    for i in range(len(df)):
        row = df.iloc[i]
        current_price = float(row["Close"])
        current_open = float(row["Open"])

        if pd.isna(current_price) or pd.isna(row.get("white_line", np.nan)):
            continue

        allow_buy = True
        if market_allow_buy is not None:
            try:
                allow_buy = bool(market_allow_buy.iloc[i])
            except (IndexError, KeyError):
                pass

        # V6.2：行业热度过滤
        if ind_allow_aligned is not None:
            try:
                date_idx = df.index[i]
                if date_idx in ind_allow_aligned.index:
                    ind_allow = ind_allow_aligned[date_idx]
                    if pd.notna(ind_allow) and not ind_allow:
                        allow_buy = False
            except (IndexError, KeyError):
                pass

        # ── T+1执行 ──
        if pending_signal_idx is not None and position is None:
            prev_row = df.iloc[pending_signal_idx]
            prev_close = float(prev_row["Close"])
            prev_yellow = float(prev_row["yellow_line"])
            prev_atr = float(prev_row["atr14"])

            if current_open > prev_close + params["t1_high_open_atr"] * prev_atr:
                pending_signal_idx = None
                continue
            if current_open < prev_yellow:
                pending_signal_idx = None
                continue

            if allow_buy and current_open > 0:
                shares = int(initial_cash * 0.3 / current_open / 100) * 100
                if shares > 0:
                    cost = shares * current_open
                    if cost <= cash:
                        cash -= cost
                        position = Position(
                            entry_date=df.index[i].strftime("%Y-%m-%d"),
                            entry_idx=i,
                            entry_price=current_open,
                            shares=shares,
                            atr_at_entry=prev_atr,
                            yellow_at_entry=prev_yellow,
                            ts_code=ts_code,
                        )
                        position.update_chandelier(float(row["High"]),
                                                    params.get("chandelier_atr_mult", 2.5))
            pending_signal_idx = None
            continue

        # 信号检测
        if position is None and bool(row.get(signal_col, False)):
            if allow_buy:
                pending_signal_idx = i

        # ── 4级退出判断（同V6.1）──
        if position is not None:
            position.hold_days += 1
            pnl_pct = (current_price - position.entry_price) / position.entry_price

            position.update_peak(float(row["High"]))
            position.update_chandelier(float(row["High"]),
                                        params.get("chandelier_atr_mult", 2.5))

            exit_reason = None

            # 优先级1：硬止损
            atr_stop = -(params["hard_stop_atr"] * position.atr_at_entry / position.entry_price)
            if pnl_pct <= atr_stop:
                exit_reason = ExitReason.ATR_HARD_STOP

            # 优先级2：吊灯止盈
            elif (current_price < position.chandelier_line
                  and position.hold_days >= params.get("chandelier_min_days", 2)):
                exit_reason = ExitReason.CHANDELIER_EXIT

            # 优先级3：Buy Climax精细化
            elif detect_buy_climax_v61(row, position.max_profit_pct, params):
                exit_reason = ExitReason.BUY_CLIMAX

            # 优先级4：时间止损
            elif (position.hold_days >= params["time_stop_days"]
                  and pnl_pct < params.get("time_stop_loss", 0.01)):
                exit_reason = ExitReason.TIME_STOP
            elif position.hold_days >= params.get("max_hold_days", 20):
                exit_reason = ExitReason.TIME_STOP
            elif (i > 0
                  and float(row["white_line"]) < float(row["yellow_line"])
                  and float(df.iloc[i-1]["white_line"]) >= float(df.iloc[i-1]["yellow_line"])):
                exit_reason = ExitReason.TIME_STOP

            # 执行卖出
            if exit_reason is not None:
                cash += position.shares * current_price
                trades.append(TradeRecord(
                    buy_date=position.entry_date,
                    sell_date=df.index[i].strftime("%Y-%m-%d"),
                    buy_price=round(position.entry_price, 2),
                    sell_price=round(current_price, 2),
                    shares=position.shares,
                    hold_days=position.hold_days,
                    profit_pct=round(pnl_pct * 100, 2),
                    max_profit_pct=round(position.max_profit_pct * 100, 2),
                    exit_reason=exit_reason.label,
                    ts_code=ts_code,
                ))
                position = None

    return trades
