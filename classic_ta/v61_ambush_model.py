"""
潜伏模型 V6.1 —— V6.0 + P0级优化
===================================
基于V6.0核心逻辑（SOS锚定 + 情绪冰点潜伏），实施两大P0优化：

  P0-1：Spring Test 弹簧试探微确认（防接飞刀）
    在冰点条件中追加：下影线>实体 OR 收阳 OR J值拐头
    过滤阴跌中"砸不动但也没人买"的假冰点

  P0-2：退出机制精简 7级→4级 + 吊灯止盈 Chandelier Exit
    4级退出：硬止损 → 吊灯止盈 → Buy Climax → 时间止损
    吊灯止盈替代原追踪止盈+保本止损，ATR自适应
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum, auto
from collections import defaultdict

# 继承V6.0的指标计算
from classic_ta.v60_ambush_model import IndicatorCalcBase, DEFAULT_PARAMS


# ╔══════════════════════════════════════════════════════════════╗
# ║  V6.1 参数                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

V61_PARAMS = DEFAULT_PARAMS.copy()
V61_PARAMS.update({
    # --- V6.0优化后最佳参数（保持不变）---
    "ambush_j_oversold": 18,
    "ambush_vol_shrink": 0.8,
    "ambush_body_pct": 0.04,
    "ambush_window": 7,
    "hard_stop_atr": 2.5,
    "time_stop_days": 10,

    # --- P0-1：Spring Test 弹簧试探参数 ---
    "spring_test_enabled": True,       # 是否启用弹簧试探
    "spring_body_ratio": 1.0,          # 下影线 > 实体×N (1.0=原版)

    # --- P0-2：吊灯止盈参数（替代追踪止盈+保本止损）---
    "chandelier_atr_mult": 3.0,        # 吊灯止盈ATR倍数
    "chandelier_min_days": 2,          # 吊灯止盈最少持仓天数（防误触）

    # --- P0-2：Buy Climax精细化参数 ---
    "climax_body_pct": 0.02,          # 努力无结果：实体占比阈值
    "climax_shadow_ratio": 2.0,       # 派发特征：上影线 > 实体×N

    # --- P0-2：时间止损（合并原时间止损+超时平仓）---
    "time_stop_loss": 0.01,           # 时间止损浮盈阈值（1%以下视为死气沉沉）
    "max_hold_days": 20,              # 超时平仓天数
})


# ╔══════════════════════════════════════════════════════════════╗
# ║  P0-1：信号检测引擎（追加Spring Test）                      ║
# ╚══════════════════════════════════════════════════════════════╝

def Detect_AmbushSignal_V61(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    V6.1 潜伏信号引擎 —— 在V6.0基础上追加Spring Test微确认

    V6.0原有条件（全部保留）：
      - 窗口期内有SOS锚定
      - J值超卖（J < 18）
      - 缩量（Volume < volume_ma × 0.8）
      - 小实体（|C-O|/C < 4%）
      - 支撑有效（Close > 黄线 - 0.5ATR）

    V6.1新增条件（必须满足其一）：
      a) 下影线 > 实体长度（探底回升 = 低位有买盘承接）
      b) 收盘价 > 开盘价（收阳 = 需求微弱但已出现）
      c) J值拐头：当日J > 前日J（情绪从冰点开始回升）

    威科夫解读：Spring = 弹簧效应，价格触及支撑后需求开始介入
    VPA解读：下影线=低位有承接，收阳=需求微弱但已出现
    蜡烛图解读：锤子线/小阳线=潜在反转信号
    """
    if params is None:
        params = V61_PARAMS

    # 先用V6.0的信号检测（保留全部原有条件）
    from classic_ta.v60_ambush_model import Detect_AmbushSignal
    df = Detect_AmbushSignal(df, params)

    # 如果未启用弹簧试探，直接返回
    if not params.get("spring_test_enabled", True):
        return df

    # ── 弹簧试探微确认 ──
    body = (df["Close"] - df["Open"]).abs()
    lower_shadow = df[["Close", "Open"]].min(axis=1) - df["Low"]

    # 条件a：下影线 > 实体×spring_body_ratio（探底回升）
    body_ratio = params.get("spring_body_ratio", 1.0)
    has_lower_shadow = lower_shadow > body * body_ratio

    # 条件b：收盘价 > 开盘价（收阳线）
    is_bullish = df["Close"] > df["Open"]

    # 条件c：J值拐头（当日J > 前日J）
    j_turning_up = df["J"] > df["J"].shift(1)

    # 三选一（OR逻辑）
    spring_test = has_lower_shadow | is_bullish | j_turning_up

    # 在原有信号基础上追加弹簧试探过滤
    df["ambush_signal"] = df["ambush_signal"] & spring_test

    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║  P0-2：退出机制精简 7级→4级 + 吊灯止盈                     ║
# ╚══════════════════════════════════════════════════════════════╝

class ExitReason(Enum):
    """退出枚举"""
    BREAKEVEN_STOP = auto()      # 优先级0.5：保本止损
    ATR_HARD_STOP = auto()       # 优先级1：硬止损
    CHANDELIER_EXIT = auto()      # 优先级2：吊灯止盈
    BUY_CLIMAX = auto()          # 优先级3：Buy Climax
    VPA_DISTRIBUTION = auto()    # 优先级3.5：VPA派发信号
    EARLY_EXIT = auto()          # 优先级3.8：快速验证退出
    TIME_STOP = auto()           # 优先级4：时间止损/超时平仓

    @property
    def label(self):
        return {
            ExitReason.BREAKEVEN_STOP: "保本止损",
            ExitReason.ATR_HARD_STOP: "硬止损",
            ExitReason.CHANDELIER_EXIT: "吊灯止盈",
            ExitReason.BUY_CLIMAX: "抢购高潮",
            ExitReason.VPA_DISTRIBUTION: "VPA派发",
            ExitReason.EARLY_EXIT: "快速止损",
            ExitReason.TIME_STOP: "时间止损",
        }[self]


@dataclass
class Position:
    """持仓信息（V6.1新增吊灯止盈字段）"""
    entry_date: str
    entry_idx: int
    entry_price: float
    shares: int
    atr_at_entry: float
    yellow_at_entry: float
    hold_days: int = 0
    max_profit_pct: float = 0.0
    highest_high: float = 0.0           # 买入后最高价
    chandelier_line: float = 0.0        # 吊灯止盈线（只上移不下移）
    ts_code: str = ""

    def update_peak(self, high_price: float):
        """更新最高价和最大浮盈"""
        pct = (high_price - self.entry_price) / self.entry_price
        if pct > self.max_profit_pct:
            self.max_profit_pct = pct

    def update_chandelier(self, high_price: float, atr_mult: float):
        """
        更新吊灯止盈线

        核心逻辑：
          追踪止盈线 = 买入后最高价 - (atr_mult × 潜伏日ATR)
          止盈线只能上移，不能下移（锁定利润）

        原理：当价格创新高时，止盈线随之上移，给趋势留出N×ATR的呼吸空间；
        当价格回落跌破止盈线时，说明供应已压倒需求，趋势反转，应离场。
        """
        if high_price > self.highest_high:
            self.highest_high = high_price
        new_line = self.highest_high - atr_mult * self.atr_at_entry
        # 吊灯线只上移不下移
        if new_line > self.chandelier_line:
            self.chandelier_line = new_line


@dataclass
class TradeRecord:
    """交易记录"""
    buy_date: str
    sell_date: str
    buy_price: float
    sell_price: float
    shares: int
    hold_days: int
    profit_pct: float
    max_profit_pct: float
    exit_reason: str
    ts_code: str = ""
    stock_name: str = ""


def detect_buy_climax_v61(row, max_profit_pct: float, params: Dict) -> bool:
    """
    V6.1 Buy Climax精细化识别（三条件同时满足）

    条件1：努力没有结果
      - 成交量 > 2.5倍均量（放巨量=努力巨大）
      - 但K线实体绝对值 < 2%收盘价（价格几乎没动=结果甚微）

    条件2：派发特征
      - 上影线长度 > 实体长度的2倍（盘中冲高被打回）

    条件3：位置确认
      - 当前处于盈利状态（max_profit_pct > 5%）
    """
    if max_profit_pct < params.get("climax_min_profit", 0.05):
        return False

    body_pct = abs(float(row["Close"]) - float(row["Open"])) / (float(row["Close"]) + 1e-8)
    vol_ratio = float(row["Volume"]) / (float(row["volume_ma"]) + 1e-8)
    effort_no_result = (
        vol_ratio > params.get("climax_vol_mult", 2.5)
        and body_pct < params.get("climax_body_pct", 0.02)
    )
    if not effort_no_result:
        return False

    body_val = abs(float(row["Close"]) - float(row["Open"]))
    upper_shadow = float(row["High"]) - max(float(row["Close"]), float(row["Open"]))
    has_distribution = upper_shadow > body_val * params.get("climax_shadow_ratio", 2.0)

    return has_distribution


# ╔══════════════════════════════════════════════════════════════╗
# ║  状态机回测（V6.1：4级退出 + 吊灯止盈）                     ║
# ╚══════════════════════════════════════════════════════════════╝

def StatefulTradeBacktester_V61(
    df: pd.DataFrame,
    signal_col: str = "ambush_signal",
    initial_cash: float = 100000.0,
    params: Dict[str, Any] = None,
    market_allow_buy=None,
    ts_code: str = "",
) -> List[TradeRecord]:
    """
    V6.1 状态机回测 —— 4级退出 + 吊灯止盈

    与V6.0的关键区别：
      1. 退出从7级精简到4级
      2. 追踪止盈+保本止损 → 吊灯止盈(Chandelier Exit)
      3. 死叉止损+超时平仓 → 合并到时间止损
      4. Buy Climax升级为三条件精细化
    """
    if params is None:
        params = V61_PARAMS

    cash = initial_cash
    position = None
    trades = []
    pending_signal_idx = None

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

        # ── T+1执行 ──
        if pending_signal_idx is not None and position is None:
            prev_row = df.iloc[pending_signal_idx]
            prev_close = float(prev_row["Close"])
            prev_yellow = float(prev_row["yellow_line"])
            prev_atr = float(prev_row["atr14"])

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
                            ts_code=ts_code,
                        )
                        # 初始化吊灯止盈线
                        position.update_chandelier(float(row["High"]),
                                                    params.get("chandelier_atr_mult", 3.0))
            pending_signal_idx = None
            continue

        # 信号检测
        if position is None and bool(row.get(signal_col, False)):
            if allow_buy:
                pending_signal_idx = i

        # ── 4级退出判断 ──
        if position is not None:
            position.hold_days += 1
            pnl_pct = (current_price - position.entry_price) / position.entry_price

            # 更新最高价和吊灯止盈线
            position.update_peak(float(row["High"]))
            position.update_chandelier(float(row["High"]),
                                        params.get("chandelier_atr_mult", 3.0))

            exit_reason = None

            # ── 优先级1：硬止损（2.5ATR）──
            atr_stop = -(params["hard_stop_atr"] * position.atr_at_entry / position.entry_price)
            if pnl_pct <= atr_stop:
                exit_reason = ExitReason.ATR_HARD_STOP

            # ── 优先级2：吊灯止盈（Chandelier Exit）──
            elif (current_price < position.chandelier_line
                  and position.hold_days >= params.get("chandelier_min_days", 2)):
                exit_reason = ExitReason.CHANDELIER_EXIT

            # ── 优先级3：Buy Climax精细化 ──
            elif detect_buy_climax_v61(row, position.max_profit_pct, params):
                exit_reason = ExitReason.BUY_CLIMAX

            # ── 优先级4：时间止损（合并原时间止损+超时平仓+死叉）──
            # 4a: 持仓>=10天且浮盈<1%（死气沉沉，释放资金）
            elif (position.hold_days >= params["time_stop_days"]
                  and pnl_pct < params.get("time_stop_loss", 0.01)):
                exit_reason = ExitReason.TIME_STOP
            # 4b: 超时强制平仓
            elif position.hold_days >= params.get("max_hold_days", 20):
                exit_reason = ExitReason.TIME_STOP
            # 4c: 死叉止损（白线跌破黄线）
            elif (i > 0
                  and float(row["white_line"]) < float(row["yellow_line"])
                  and float(df.iloc[i-1]["white_line"]) >= float(df.iloc[i-1]["yellow_line"])):
                exit_reason = ExitReason.TIME_STOP

            # ── 执行卖出 ──
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


# ╔══════════════════════════════════════════════════════════════╗
# ║  便捷入口                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def run_v61_backtest(df, params=None, market_allow_buy=None, ts_code=""):
    df = IndicatorCalcBase(df)
    df = Detect_AmbushSignal_V61(df, params)
    trades = StatefulTradeBacktester_V61(
        df, signal_col="ambush_signal", params=params,
        market_allow_buy=market_allow_buy, ts_code=ts_code,
    )
    return trades


def compute_v61_metrics(trades: List[TradeRecord]) -> Dict[str, Any]:
    """计算回测指标"""
    if not trades:
        return {"total": 0, "win_rate": 0, "avg_profit": 0, "profit_factor": 0,
                "avg_hold": 0, "avg_max_profit": 0, "total_profit": 0, "exit_counts": {}}

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
