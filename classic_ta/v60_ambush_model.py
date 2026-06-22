"""
潜伏模型 V6.0 —— 威科夫+VPA 情绪冰点潜伏量化框架
=================================================
核心理念：看到SOS信号后，不追涨，而是等情绪冰点后潜伏等待拉升

策略流程：
  1. SOS锚定：识别需求大阳线（SOS），确认主力已入场
  2. 情绪冰点：SOS后1~5天内，等待J值跌入超卖区+缩量+小实体
  3. 潜伏买入：在冰点日T日确认，T+1开盘执行（含ATR化防护）
  4. 持仓退出：7级退出机制（硬止损→追踪止盈→保本→Buy Climax→时间→死叉→超时）

与V5.0的关键区别：
  V5.0 = SOS日直接买入（追涨）
  V6.0 = SOS日仅锚定 → 等情绪冰点 → 潜伏买入（低吸）

防未来数据泄漏：
  - SOS锚定使用shift(1).rolling()确保只看T日及之前
  - 潜伏信号在冰点日T日确认，T+1开盘执行
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum, auto
import json
import time
from pathlib import Path


# ╔══════════════════════════════════════════════════════════════╗
# ║  MODULE 1 — IndicatorCalcBase  指标计算工厂（同V5.0）      ║
# ╚══════════════════════════════════════════════════════════════╝

def IndicatorCalcBase(df: pd.DataFrame) -> pd.DataFrame:
    """指标计算工厂 —— 输入原始OHLCV，输出全部技术指标列"""
    df = df.copy()

    # 双线体系
    df["white_line"] = (
        df["Close"].ewm(span=10, adjust=False).mean()
        .ewm(span=10, adjust=False).mean()
    )
    df["ma14"] = df["Close"].rolling(window=14, min_periods=1).mean()
    df["ma28"] = df["Close"].rolling(window=28, min_periods=1).mean()
    df["ma57"] = df["Close"].rolling(window=57, min_periods=1).mean()
    df["ma114"] = df["Close"].rolling(window=114, min_periods=1).mean()
    df["yellow_line"] = (df["ma14"] + df["ma28"] + df["ma57"] + df["ma114"]) / 4

    # ATR14（True Range）
    prev_close = df["Close"].shift(1)
    tr = pd.DataFrame({
        "tr1": df["High"] - df["Low"],
        "tr2": (df["High"] - prev_close).abs(),
        "tr3": (df["Low"] - prev_close).abs(),
    }).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=1).mean()

    # 成交量均线
    df["volume_ma"] = df["Volume"].rolling(20, min_periods=1).mean()

    # 日收益率
    df["daily_return"] = df["Close"].pct_change() * 100

    # KDJ
    low_9 = df["Low"].rolling(window=9, min_periods=1).min()
    high_9 = df["High"].rolling(window=9, min_periods=1).max()
    rsv = (df["Close"] - low_9) / (high_9 - low_9 + 1e-8) * 100
    rsv = rsv.fillna(50)
    df["K"] = rsv.ewm(com=2, adjust=False).mean()
    df["D"] = df["K"].ewm(com=2, adjust=False).mean()
    df["J"] = (3 * df["K"] - 2 * df["D"]).clip(0, 100)

    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║  MODULE 2 — 潜伏信号引擎（参数化）                         ║
# ║  两阶段：SOS锚定 → 情绪冰点潜伏                            ║
# ╚══════════════════════════════════════════════════════════════╝

# 默认参数
DEFAULT_PARAMS = {
    # --- SOS锚定参数 ---
    "sos_body_ratio": 0.50,        # SOS实体占比阈值
    "sos_close_position": 0.60,    # SOS收盘位阈值
    "sos_vol_absolute": 0.75,     # SOS量能绝对阈值(×volume_ma)
    "sos_vol_relative": 1.5,       # SOS量能相对阈值(×枯竭柱量)
    "support_atr_mult": 1.5,       # 支撑带ATR倍数
    "deviation_atr_mult": 3.0,     # 乖离率ATR倍数
    "test_atr_touch": 0.5,         # Test柱触及黄线ATR容差
    "no_supply_vol_ratio": 0.65,   # No Supply缩量阈值
    "test_vol_ratio": 1.2,         # Test柱量能阈值

    # --- 潜伏冰点参数 ---
    "ambush_window": 5,             # SOS后潜伏等待窗口(天)
    "ambush_j_oversold": 13,        # 潜伏J值超卖阈值
    "ambush_vol_shrink": 0.70,      # 潜伏缩量阈值(×volume_ma)
    "ambush_body_pct": 0.03,        # 潜伏实体占比阈值(实体/收盘价)
    "ambush_support_atr": 0.5,      # 潜伏支撑容差(ATR倍数，在黄线上方)

    # --- 股性参数 ---
    "dragon_atr_mult": 200,        # 龙回头ATR倍数(百分比化)

    # --- 退出参数 ---
    "hard_stop_atr": 2.0,           # 硬止损ATR倍数
    "trailing_trigger": 0.15,       # 追踪止盈触发(15%)
    "trailing_drawdown": 0.05,      # 追踪止盈回撤(5%)
    "breakeven_low": 0.08,          # 保本区间下限(8%)
    "breakeven_high": 0.15,         # 保本区间上限(15%)
    "breakeven_trigger": 0.015,     # 保本触发线(1.5%)
    "time_stop_days": 8,            # 时间止损天数
    "time_stop_loss": -0.01,        # 时间止损亏损阈值
    "max_hold_days": 20,            # 超时平仓天数
    "climax_vol_mult": 2.5,         # Buy Climax量能倍数
    "climax_ret_ratio": 0.5,        # Buy Climax涨幅收窄比
    "climax_min_profit": 0.05,      # Buy Climax最低浮盈

    # --- T+1防护参数 ---
    "t1_high_open_atr": 1.5,       # T+1防高开ATR倍数
}


def Detect_AmbushSignal(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    潜伏信号引擎 —— 两阶段：SOS锚定 → 情绪冰点潜伏

    阶段一：SOS锚定（同V5.0，但不直接买入）
      - 趋势环境：白线>黄线 + Close>黄线
      - 左侧枯竭：近3日No Supply或Test
      - SOS需求确认：阳线+饱满+收偏上+放量+吃掉昨日实体
      - 安全护栏：ATR化支撑带+乖离率+股性

    阶段二：情绪冰点潜伏（V6.0核心创新）
      - 窗口期内有SOS锚定（近1~N天）
      - J值超卖：J < 阈值（情绪冰点）
      - 缩量：Volume < volume_ma × 阈值（抛压枯竭）
      - 小实体：|C-O|/C < 阈值（拒绝下跌，砸不动了）
      - 支撑有效：Close > 黄线 - ATR容差（均线托底）
    """
    if params is None:
        params = DEFAULT_PARAMS
    df = df.copy()

    # ── 蜡烛图要素 ──
    body = (df["Close"] - df["Open"]).abs()
    lower_shadow = df[["Close", "Open"]].min(axis=1) - df["Low"]
    amplitude = df["High"] - df["Low"]

    # ══════════════════════════════════════════════════════════
    #  阶段一：SOS锚定（不直接买入，仅标记）
    # ══════════════════════════════════════════════════════════

    # 趋势环境
    trend_ok = (df["white_line"] > df["yellow_line"]) & (df["Close"] > df["yellow_line"])

    # 左侧枯竭
    no_supply = (
        (df["Volume"] < df["volume_ma"] * params["no_supply_vol_ratio"])
        & (df["Volume"] < df["Volume"].shift(1))
        & (amplitude < amplitude.shift(1))
    )
    test = (
        (df["Low"] <= df["yellow_line"] + params["test_atr_touch"] * df["atr14"])
        & (df["Close"] > df["Low"])
        & (lower_shadow > body * 1.5)
        & (df["Volume"] < df["volume_ma"] * params["test_vol_ratio"])
    )
    depletion = no_supply | test
    depletion_w = depletion.shift(1).rolling(3, min_periods=1).max().astype(bool)

    # 枯竭量基准
    dep_vol = pd.Series(0.0, index=df.index)
    dep_vol.loc[no_supply] = df.loc[no_supply, "Volume"]
    dep_vol.loc[test] = df.loc[test, "Volume"]
    dep_vol_max = dep_vol.shift(1).rolling(3, min_periods=1).max()

    # SOS需求确认
    is_bullish = df["Close"] > df["Open"]
    is_solid = body / (amplitude + 1e-8) > params["sos_body_ratio"]
    close_pos = (df["Close"] - df["Low"]) / (amplitude + 1e-8)
    is_strong = close_pos > params["sos_close_position"]
    is_demand_vol = (
        (df["Volume"] > df["volume_ma"] * params["sos_vol_absolute"])
        & (df["Volume"] > dep_vol_max * params["sos_vol_relative"])
    )
    yesterday_top = np.maximum(df["Open"].shift(1), df["Close"].shift(1))
    is_breakout = df["Close"] > yesterday_top

    tag_sos = is_bullish & is_solid & is_strong & is_demand_vol & is_breakout

    # 安全护栏
    is_support = df["Close"] > df["yellow_line"] - params["support_atr_mult"] * df["atr14"]
    is_not_over = df["Close"] <= df["yellow_line"] + params["deviation_atr_mult"] * df["atr14"]

    # 股性
    amp_pct = (df["High"] - df["Low"]) / (df["Close"].shift(1) + 1e-8)
    is_active = ((df["daily_return"] > 9.5) | (amp_pct > 0.10)).rolling(20, min_periods=1).max().astype(bool)
    atr30 = df["atr14"].rolling(30, min_periods=1).mean()
    is_dragon = df["daily_return"].rolling(30, min_periods=1).max() > (atr30 / (df["Close"] + 1e-8)) * params["dragon_atr_mult"]
    is_stock_ok = is_active & is_dragon

    # SOS锚定综合
    is_sos_anchor = (trend_ok & depletion_w & tag_sos & is_support & is_not_over & is_stock_ok).fillna(False)

    # ══════════════════════════════════════════════════════════
    #  阶段二：情绪冰点潜伏（V6.0核心创新）
    # ══════════════════════════════════════════════════════════
    # 威科夫LPS思想：SOS后主力缩量洗盘，当情绪到达冰点时潜伏
    # 三本书的识别保障：
    #   威科夫：SOS确认需求入场 → LPS缩量回踩 = 最后支撑点
    #   VPA：缩量=供应枯竭，小实体=砸不动=需求随时介入
    #   蜡烛图：十字星/小阴小阳=多空平衡即将打破

    # 条件1：窗口期内有SOS锚定（近1~N天）
    window = params["ambush_window"]
    sos_in_window = is_sos_anchor.shift(1).rolling(window, min_periods=1).max().astype(bool)

    # 条件2：J值超卖（情绪冰点）
    j_oversold = df["J"] < params["ambush_j_oversold"]

    # 条件3：缩量（抛压枯竭）
    vol_shrink = df["Volume"] < df["volume_ma"] * params["ambush_vol_shrink"]

    # 条件4：小实体（拒绝下跌，砸不动了）
    tiny_body = body / (df["Close"] + 1e-8) < params["ambush_body_pct"]

    # 条件5：支撑有效（均线托底）
    support_ok = df["Close"] > df["yellow_line"] - params["ambush_support_atr"] * df["atr14"]

    # 潜伏信号综合
    df["tag_sos_anchor"] = is_sos_anchor
    df["tag_no_supply"] = no_supply
    df["tag_test"] = test
    df["ambush_signal"] = (
        sos_in_window
        & j_oversold
        & vol_shrink
        & tiny_body
        & support_ok
    ).fillna(False)

    # P2: 多日确认基础 —— 每日满足条件数（用于E7评分）
    df["near_signal_count"] = (
        sos_in_window.astype(int)
        + j_oversold.astype(int)
        + vol_shrink.astype(int)
        + tiny_body.astype(int)
        + support_ok.astype(int)
    ).fillna(0)

    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║  MODULE 3 — 状态机回测（参数化退出）                        ║
# ╚══════════════════════════════════════════════════════════════╝

class ExitReason(Enum):
    ATR_HARD_STOP = auto()
    TRAILING_TAKE = auto()
    BREAKEVEN_STOP = auto()
    BUY_CLIMAX = auto()
    TIME_STOP = auto()
    DEATH_CROSS = auto()
    MAX_HOLD = auto()

    @property
    def label(self):
        return {
            ExitReason.ATR_HARD_STOP: "硬止损",
            ExitReason.TRAILING_TAKE: "追踪止盈",
            ExitReason.BREAKEVEN_STOP: "保本止损",
            ExitReason.BUY_CLIMAX: "抢购高潮",
            ExitReason.TIME_STOP: "时间止损",
            ExitReason.DEATH_CROSS: "死叉止损",
            ExitReason.MAX_HOLD: "超时平仓",
        }[self]


@dataclass
class Position:
    entry_date: str
    entry_idx: int
    entry_price: float
    shares: int
    atr_at_entry: float
    yellow_at_entry: float
    hold_days: int = 0
    max_profit_pct: float = 0.0

    def update_peak(self, high_price: float):
        pct = (high_price - self.entry_price) / self.entry_price
        if pct > self.max_profit_pct:
            self.max_profit_pct = pct


@dataclass
class TradeRecord:
    buy_date: str
    sell_date: str
    buy_price: float
    sell_price: float
    shares: int
    hold_days: int
    profit_pct: float
    max_profit_pct: float
    exit_reason: str


def _detect_buy_climax(row, prev_rows, max_profit_pct, params):
    if max_profit_pct < params["climax_min_profit"]:
        return False
    if row["Volume"] <= row["volume_ma"] * params["climax_vol_mult"]:
        return False
    if len(prev_rows) >= 10:
        avg_ret = prev_rows["daily_return"].iloc[-10:].mean()
    else:
        avg_ret = prev_rows["daily_return"].mean() if len(prev_rows) > 0 else 0
    daily_ret = row.get("daily_return", 0)
    if avg_ret == 0 or abs(daily_ret) >= abs(avg_ret) * params["climax_ret_ratio"]:
        return False
    body_val = abs(row["Close"] - row["Open"])
    upper_shadow = row["High"] - max(row["Close"], row["Open"])
    amp = row["High"] - row["Low"]
    has_supply = (
        (upper_shadow > body_val * 1.5)
        or ((row["Close"] < row["Open"]) and (body_val / (amp + 1e-8) > 0.6))
    )
    return has_supply


def StatefulTradeBacktester(
    df: pd.DataFrame,
    signal_col: str = "ambush_signal",
    initial_cash: float = 100000.0,
    params: Dict[str, Any] = None,
    market_allow_buy=None,
) -> List[TradeRecord]:
    if params is None:
        params = DEFAULT_PARAMS

    cash = initial_cash
    position = None
    trades = []
    pending_signal_idx = None

    for i in range(len(df)):
        row = df.iloc[i]
        current_price = row["Close"]
        current_open = row["Open"]

        if pd.isna(current_price) or pd.isna(row.get("white_line", np.nan)):
            continue

        allow_buy = True
        if market_allow_buy is not None:
            try:
                allow_buy = bool(market_allow_buy.iloc[i])
            except (IndexError, KeyError):
                pass

        # T+1执行
        if pending_signal_idx is not None and position is None:
            prev_row = df.iloc[pending_signal_idx]
            prev_close = prev_row["Close"]
            prev_yellow = prev_row["yellow_line"]
            prev_atr = prev_row["atr14"]

            # 防高开
            if current_open > prev_close + params["t1_high_open_atr"] * prev_atr:
                pending_signal_idx = None
                continue
            # 防破位
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
                        )
            pending_signal_idx = None
            continue

        # 信号检测
        if position is None and bool(row.get(signal_col, False)):
            pending_signal_idx = i

        # 7级退出
        if position is not None:
            position.hold_days += 1
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            position.update_peak(row["High"])

            atr_stop = -(params["hard_stop_atr"] * position.atr_at_entry / position.entry_price)

            if pnl_pct <= atr_stop:
                exit_reason = ExitReason.ATR_HARD_STOP
            elif (position.max_profit_pct >= params["trailing_trigger"]
                  and (position.max_profit_pct - pnl_pct) >= params["trailing_drawdown"]):
                exit_reason = ExitReason.TRAILING_TAKE
            elif (params["breakeven_low"] <= position.max_profit_pct < params["breakeven_high"]
                  and pnl_pct <= params["breakeven_trigger"]):
                exit_reason = ExitReason.BREAKEVEN_STOP
            elif _detect_buy_climax(row, df.iloc[:i], position.max_profit_pct, params):
                exit_reason = ExitReason.BUY_CLIMAX
            elif position.hold_days >= params["time_stop_days"] and pnl_pct <= params["time_stop_loss"]:
                exit_reason = ExitReason.TIME_STOP
            elif (i > 0
                  and row["white_line"] < row["yellow_line"]
                  and df.iloc[i-1]["white_line"] >= df.iloc[i-1]["yellow_line"]):
                exit_reason = ExitReason.DEATH_CROSS
            elif position.hold_days >= params["max_hold_days"]:
                exit_reason = ExitReason.MAX_HOLD
            else:
                exit_reason = None

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
                ))
                position = None

    return trades


# ╔══════════════════════════════════════════════════════════════╗
# ║  便捷入口                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def run_v60_backtest(df, params=None, market_allow_buy=None):
    df = IndicatorCalcBase(df)
    df = Detect_AmbushSignal(df, params)
    trades = StatefulTradeBacktester(df, signal_col="ambush_signal", params=params, market_allow_buy=market_allow_buy)
    return trades


def compute_metrics(trades: List[TradeRecord]) -> Dict[str, Any]:
    """计算回测指标"""
    if not trades:
        return {"total": 0, "win_rate": 0, "avg_profit": 0, "profit_factor": 0, "avg_hold": 0, "avg_max_profit": 0, "total_profit": 0, "exit_counts": {}}

    total = len(trades)
    wins = [t for t in trades if t.profit_pct > 0]
    losses = [t for t in trades if t.profit_pct <= 0]
    win_rate = len(wins) / total * 100
    avg_profit = np.mean([t.profit_pct for t in trades])
    avg_win = np.mean([t.profit_pct for t in wins]) if wins else 0
    avg_lose = abs(np.mean([t.profit_pct for t in losses])) if losses else 0.01
    profit_factor = avg_win / avg_lose if avg_lose > 0 else 999
    avg_hold = np.mean([t.hold_days for t in trades])
    avg_max_profit = np.mean([t.max_profit_pct for t in trades])
    total_profit = sum(t.profit_pct for t in trades)

    exit_counts = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    return {
        "total": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "avg_profit": round(avg_profit, 2),
        "avg_win": round(avg_win, 2),
        "avg_lose": round(avg_lose, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_hold": round(avg_hold, 1),
        "avg_max_profit": round(avg_max_profit, 2),
        "total_profit": round(total_profit, 2),
        "exit_counts": exit_counts,
    }
