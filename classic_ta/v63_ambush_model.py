"""
潜伏模型 V6.3 —— V6.2 + 四维度深度优化
=======================================
基于V6.2（行业热度过滤），实施4个维度的量化升级：

  维度一：行业动量过滤 —— 从"绝对阈值"升级为"横截面相对强度（Cross-Sectional RS）"
    痛点：绝对阈值2%在牛市失效，在熊市错杀
    方案：每个交易日计算所有行业动量的百分位排名（Percentile Rank），
          信号触发时，该股所在行业必须处于全市场前Top 20%
          或处于"轮入加速"状态（动量5日变化>0且当前排名>50%）

  维度二：资金与头寸管理 —— 从"固定30%"升级为"波动率平价（Volatility Parity）"
    痛点：固定30%仓位未考虑个股波动率差异，资金利用率低
    方案：shares = (总资金 × 单笔风险比例) / (hard_stop_atr × ATR)
          单笔最大风险敞口=总资金1%，单只仓位上限15%

  维度三：重新激活"Spring Test" —— 智能微观止跌确认（VWAP/VCP）
    痛点：传统下影线Spring Test被优化器否决（过滤太多横盘缩量信号）
    方案：设计"右侧微确认"因子——T日收盘价站在VWAP之上
          或T日出现振幅萎缩（Inside Bar / 波动率收缩）
          比简单下影线更精准，保留横盘缩量的优质信号

  维度四：T+1订单执行与滑点控制 —— 限价单逻辑
    痛点：T+1开盘价买入容易被情绪裹挟，日内浮亏
    方案：以T日收盘价与T日黄线之间的支撑位作为限价买入点
          限价 = T日收盘价 × (1 - limit_discount)
          若T+1日最低价触及限价则成交，否则放弃
          回测中用min(开盘价, 限价)模拟成交
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict

from classic_ta.v60_ambush_model import IndicatorCalcBase, DEFAULT_PARAMS
from classic_ta.v61_ambush_model import (
    Position, TradeRecord, ExitReason,
    Detect_AmbushSignal_V61, detect_buy_climax_v61,
    StatefulTradeBacktester_V61, compute_v61_metrics, V61_PARAMS,
)
from classic_ta.v62_ambush_model import (
    compute_industry_momentum, V62_PARAMS,
)


# ╔══════════════════════════════════════════════════════════════╗
# ║  V6.3 参数                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

V63_PARAMS = V62_PARAMS.copy()
V63_PARAMS.update({
    # --- V6.2最佳参数（固定）---
    "spring_test_enabled": False,
    "chandelier_atr_mult": 3.5,
    "hard_stop_atr": 2.5,
    "time_stop_days": 10,
    "time_stop_extend_profit_pct": 5.0,    # 浮盈>5%时延长持仓
    "time_stop_extend_days": 20,           # 延长至20天
    "ambush_j_oversold": 25,
    "ambush_window": 12,

    # --- 维度一：横截面相对强度 ---
    "industry_filter_enabled": True,
    "industry_momentum_days": 10,
    "industry_rs_top_pct": 0.30,          # 行业RS排名前30%才允许买入（比V6.2的2%阈值更宽松）
    "industry_rs_or_rotation": True,       # 或者满足轮入加速条件也可买入

    # --- 维度二：波动率平价仓位 ---
    "volatility_parity_enabled": True,     # 是否启用波动率平价
    "risk_per_trade": 0.01,                # 单笔最大风险=总资金1%
    "max_position_pct": 0.15,              # 单只股票仓位上限15%
    "min_position_pct": 0.05,              # 单只股票仓位下限5%

    # --- 维度三：智能微观止跌确认 ---
    "micro_confirm_enabled": True,         # 是否启用微观止跌确认
    "micro_confirm_mode": "any",           # "any"=任一满足即可（VWAP OR VCP）
    # 子条件开关
    "micro_vwap_above": True,              # 收盘价站在VWAP之上
    "micro_inside_bar": False,             # Inside Bar（关闭）
    "micro_vcp_shrink": True,              # 波动率收缩（ATR连续3日下降）
    "micro_lower_wick": False,             # 下影线支撑（关闭）

    # --- 维度四：限价单执行 ---
    "limit_order_enabled": False,          # 默认关闭（回测中难以准确模拟，实盘手动执行）
    "limit_discount": 0.002,               # 限价折扣=T日收盘价×(1-0.2%)（微折扣，不过度等待）
    "limit_use_yellow": True,              # 限价取min(折扣价, T日黄线)
    "limit_expire_same_day": True,         # 限价单当日有效

    # --- 维度五：动态止损参数（基于ATR百分位） ---
    "dynamic_stop_enabled": True,          # 是否启用动态止损
    "dynamic_stop_atr_lookback": 60,        # ATR百分位回看天数
    "dynamic_stop_low_pctile": 30,          # 低波动百分位阈值
    "dynamic_stop_high_pctile": 70,         # 高波动百分位阈值
    "dynamic_hard_stop_low": 2.0,           # 低波动硬止损
    "dynamic_hard_stop_mid": 2.5,           # 中波动硬止损
    "dynamic_hard_stop_high": 3.0,          # 高波动硬止损
    "dynamic_chandelier_low": 3.0,          # 低波动吊灯
    "dynamic_chandelier_mid": 3.5,          # 中波动吊灯
    "dynamic_chandelier_high": 4.0,         # 高波动吊灯

    # --- 维度六：UT/AD退出保护 ---
    "utad_exit_enabled": True,              # 是否启用UT/AD退出保护
    "utad_tighten_chandelier": 1.5,         # UT/AD出现时吊灯收紧到1.5ATR
    "utad_min_profit_pct": 5.0,             # 只有浮盈>5%时UT/AD才触发收紧
    "bearish_vpa_exit_enabled": True,       # 连续熊性VPA信号触发卖出
    "bearish_vpa_consecutive_days": 2,      # 连续N日bearish_vpa_count>=threshold时卖出
    "bearish_vpa_min_count": 2,             # 每日最少熊性VPA信号数
})


# ╔══════════════════════════════════════════════════════════════╗
# ║  维度五：动态止损参数（基于ATR百分位）                       ║
# ╚══════════════════════════════════════════════════════════════╝

def calc_dynamic_stop_params(df: pd.DataFrame, idx: int, params: Dict) -> tuple:
    """根据ATR百分位动态计算止损参数

    核心思路：
      入场时，计算当前ATR14相对于过去N日ATR的百分位排名。
      低波动期 → 收窄止损（市场稳定，止损应更紧）
      高波动期 → 放宽止损（市场震荡，需要更多呼吸空间）

    参数:
      df: 含atr14列的DataFrame（已通过IndicatorCalcBase计算）
      idx: 当前交易日索引（建仓日）
      params: 参数字典

    返回: (hard_stop_atr, chandelier_atr_mult)
    """
    if not params.get("dynamic_stop_enabled", True):
        # 未启用动态止损，返回默认值
        return params["hard_stop_atr"], params.get("chandelier_atr_mult", 3.5)

    lookback = params.get("dynamic_stop_atr_lookback", 60)
    low_pctile = params.get("dynamic_stop_low_pctile", 30)
    high_pctile = params.get("dynamic_stop_high_pctile", 70)

    # 获取当前ATR14
    current_atr = float(df.iloc[idx]["atr14"])
    if pd.isna(current_atr) or current_atr <= 0:
        return params["hard_stop_atr"], params.get("chandelier_atr_mult", 3.5)

    # 计算过去N日ATR的百分位排名
    start_idx = max(0, idx - lookback)
    atr_window = df.iloc[start_idx:idx + 1]["atr14"].dropna()

    if len(atr_window) < 10:
        # 数据不足，回退到默认值
        return params["hard_stop_atr"], params.get("chandelier_atr_mult", 3.5)

    # 百分位排名：当前ATR在窗口中的排名百分比
    pctile_rank = (atr_window < current_atr).sum() / len(atr_window) * 100

    # 根据百分位选择参数
    if pctile_rank < low_pctile:
        # 低波动期：收窄止损
        hard_stop_atr = params.get("dynamic_hard_stop_low", 2.0)
        chandelier_atr_mult = params.get("dynamic_chandelier_low", 3.0)
    elif pctile_rank > high_pctile:
        # 高波动期：放宽止损
        hard_stop_atr = params.get("dynamic_hard_stop_high", 3.0)
        chandelier_atr_mult = params.get("dynamic_chandelier_high", 4.0)
    else:
        # 中波动期：默认
        hard_stop_atr = params.get("dynamic_hard_stop_mid", 2.5)
        chandelier_atr_mult = params.get("dynamic_chandelier_mid", 3.5)

    return hard_stop_atr, chandelier_atr_mult


# ╔══════════════════════════════════════════════════════════════╗
# ║  维度一：横截面相对强度（Cross-Sectional RS）                ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_industry_rs_matrix(mom_df: pd.DataFrame,
                                top_pct: float = 0.20,
                                or_rotation: bool = True,
                                momentum_change_days: int = 5) -> pd.DataFrame:
    """
    横截面相对强度：将行业动量转换为百分位排名，构建允许买入矩阵

    核心逻辑（向量化）：
      1. 每个交易日，对所有行业动量做rank(pct=True) → 得到0~1的百分位
      2. 允许买入 = RS排名 >= (1 - top_pct)，即Top 20%
      3. 如果or_rotation=True，额外允许"轮入加速"行业：
         - 动量5日变化 > 0（正在加速）
         - 且当前RS排名 > 50%（已经脱离底部）

    参数:
      mom_df: 行业动量DataFrame（index=Date, columns=行业名）
      top_pct: 允许买入的行业排名阈值（0.20=前20%）
      or_rotation: 是否允许轮入加速行业
      momentum_change_days: 轮动判断的回看天数

    返回:
      DataFrame: index=Date, columns=行业名, values=bool（True=允许买入）
    """
    # 1. 横截面百分位排名（向量化：按行rank）
    rs_rank = mom_df.rank(axis=1, pct=True, na_option="keep")

    # 2. Top N% 条件
    top_threshold = 1.0 - top_pct
    is_top = rs_rank >= top_threshold

    if not or_rotation:
        return is_top

    # 3. 轮入加速条件（向量化）
    # 动量变化 = 当日动量 - N日前动量
    mom_change = mom_df - mom_df.shift(momentum_change_days)
    is_accelerating = mom_change > 0  # 动量正在上升
    is_above_median = rs_rank > 0.50  # 已脱离底部
    is_rotation_in = is_accelerating & is_above_median

    # 4. 允许买入 = Top N% OR 轮入加速
    allow_matrix = is_top | is_rotation_in

    return allow_matrix


def build_industry_allow_matrix_v63(mom_df: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
    """
    V6.3行业允许买入矩阵（基于横截面RS）
    兼容V6.2的build_industry_allow_matrix接口
    """
    if params is None:
        params = V63_PARAMS

    if not params.get("industry_filter_enabled", True):
        # 不过滤时全部允许
        return pd.DataFrame(True, index=mom_df.index, columns=mom_df.columns)

    top_pct = params.get("industry_rs_top_pct", 0.20)
    or_rotation = params.get("industry_rs_or_rotation", True)

    return compute_industry_rs_matrix(mom_df, top_pct, or_rotation)


# ╔══════════════════════════════════════════════════════════════╗
# ║  维度二：波动率平价仓位（Volatility Parity）                  ║
# ╚══════════════════════════════════════════════════════════════╝

def calc_volatility_parity_shares(
    total_equity: float,
    entry_price: float,
    atr_at_entry: float,
    hard_stop_atr: float,
    params: Dict = None,
) -> int:
    """
    波动率平价仓位计算

    核心公式：
      单笔风险金额 = 总资金 × risk_per_trade
      每股最大亏损 = hard_stop_atr × ATR
      应买股数 = 单笔风险金额 / 每股最大亏损
      仓位限制 = [min_pct, max_pct] × 总资金 / 股价

    参数:
      total_equity: 当前总资金
      entry_price: 买入价格
      atr_at_entry: 买入时ATR
      hard_stop_atr: 硬止损ATR倍数
      params: 参数字典

    返回:
      应买股数（100股整数倍）
    """
    if params is None:
        params = V63_PARAMS

    if not params.get("volatility_parity_enabled", True):
        # 回退到固定30%仓位
        shares = int(total_equity * 0.3 / entry_price / 100) * 100
        return shares

    risk_per_trade = params.get("risk_per_trade", 0.01)
    max_pos_pct = params.get("max_position_pct", 0.15)
    min_pos_pct = params.get("min_position_pct", 0.05)

    # 单笔最大风险金额
    risk_amount = total_equity * risk_per_trade

    # 每股最大亏损（硬止损触发时的亏损）
    per_share_loss = hard_stop_atr * atr_at_entry

    if per_share_loss <= 0:
        return 0

    # 基于风险的股数
    risk_based_shares = risk_amount / per_share_loss

    # 仓位上下限约束
    max_shares = total_equity * max_pos_pct / entry_price
    min_shares = total_equity * min_pos_pct / entry_price

    shares = np.clip(risk_based_shares, min_shares, max_shares)

    # 取整到100股
    shares = int(shares / 100) * 100

    return max(shares, 0)


# ╔══════════════════════════════════════════════════════════════╗
# ║  维度三：智能微观止跌确认（VWAP / VCP / Inside Bar）        ║
# ╚══════════════════════════════════════════════════════════════╝

def add_micro_confirm_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    在DataFrame上追加微观止跌确认所需的指标列（向量化计算）

    追加列：
      - vwap: 成交量加权平均价（用amount/vol近似，或用典型价×成交量加权）
      - is_above_vwap: 收盘价站在VWAP之上
      - is_inside_bar: Inside Bar（当日高低价完全包含在前日高低价内）
      - is_vcp_shrink: 波动率收缩（ATR连续3日下降）
      - is_lower_wick_support: 下影线支撑+缩量（改良版Spring Test）
      - micro_confirm: 综合微观确认信号
    """
    # 1. VWAP（成交量加权平均价）—— 真正的累计VWAP
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical_price * df["Volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    df["vwap"] = cum_pv / cum_vol
    df["vwap"] = df["vwap"].fillna(typical_price)  # 无成交量时用典型价兜底

    # 2. 收盘价站在VWAP之上
    df["is_above_vwap"] = df["Close"] > df["vwap"]

    # 3. Inside Bar（振幅萎缩：当日高低价完全在前日高低价内）
    df["is_inside_bar"] = (
        (df["High"] <= df["High"].shift(1))
        & (df["Low"] >= df["Low"].shift(1))
    )

    # 4. 波动率收缩（ATR连续3日下降）
    df["is_vcp_shrink"] = (
        (df["atr14"] < df["atr14"].shift(1))
        & (df["atr14"].shift(1) < df["atr14"].shift(2))
    )

    # 5. 改良版下影线支撑：下影线>实体 且 缩量
    body = (df["Close"] - df["Open"]).abs()
    lower_shadow = df[["Close", "Open"]].min(axis=1) - df["Low"]
    vol_ratio = df["Volume"] / df["volume_ma"].replace(0, np.nan)
    df["is_lower_wick_support"] = (
        (lower_shadow > body * 0.8)  # 下影线>0.8倍实体（比原版1.0更宽松）
        & (vol_ratio < 0.9)          # 缩量（量比<0.9）
    )

    return df


def Detect_AmbushSignal_V63(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    V6.3 潜伏信号引擎 —— V6.1基础 + 智能微观止跌确认

    与V6.1的区别：
      1. spring_test_enabled=False（不再使用传统Spring Test）
      2. 新增micro_confirm微观止跌确认（VWAP/Inside Bar/VCP/改良下影线）
      3. 微观确认是"任一满足"模式，比传统OR逻辑更精准
    """
    if params is None:
        params = V63_PARAMS

    # 先用V6.1的信号检测（spring_test_enabled=False时等同V6.0）
    df = Detect_AmbushSignal_V61(df, params)

    # 如果未启用微观确认，直接返回
    if not params.get("micro_confirm_enabled", True):
        return df

    # 确保微观指标已计算
    if "is_above_vwap" not in df.columns:
        df = add_micro_confirm_indicators(df)

    # ── 构建微观确认条件 ──
    conditions = []

    if params.get("micro_vwap_above", True):
        conditions.append(df["is_above_vwap"])

    if params.get("micro_inside_bar", True):
        conditions.append(df["is_inside_bar"])

    if params.get("micro_vcp_shrink", True):
        conditions.append(df["is_vcp_shrink"])

    if params.get("micro_lower_wick", True):
        conditions.append(df["is_lower_wick_support"])

    if not conditions:
        return df

    # 合并条件
    if params.get("micro_confirm_mode", "any") == "any":
        micro_confirm = conditions[0]
        for c in conditions[1:]:
            micro_confirm = micro_confirm | c
    else:  # "all"
        micro_confirm = conditions[0]
        for c in conditions[1:]:
            micro_confirm = micro_confirm & c

    # 在原有信号基础上追加微观确认过滤
    df["ambush_signal"] = df["ambush_signal"] & micro_confirm
    df["micro_confirm"] = micro_confirm

    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║  维度四：限价单执行逻辑                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def calc_limit_price(prev_close: float, prev_yellow: float,
                     prev_atr: float, params: Dict = None) -> float:
    """
    计算T+1日限价买入价格

    逻辑：
      1. 基础限价 = T日收盘价 × (1 - limit_discount)
      2. 如果limit_use_yellow=True，限价 = min(基础限价, T日黄线)
         原理：黄线是支撑线，不应在支撑线下方追买
      3. 限价不能低于T日收盘价 - 2×ATR（防止限价过低永远无法成交）

    参数:
      prev_close: T日收盘价
      prev_yellow: T日黄线
      prev_atr: T日ATR
      params: 参数字典

    返回:
      限价（float）
    """
    if params is None:
        params = V63_PARAMS

    discount = params.get("limit_discount", 0.005)
    base_limit = prev_close * (1 - discount)

    if params.get("limit_use_yellow", True):
        limit_price = min(base_limit, prev_yellow)
    else:
        limit_price = base_limit

    # 限价下限保护（不能太低，否则永远无法成交）
    floor_price = prev_close - 2.0 * prev_atr
    limit_price = max(limit_price, floor_price)

    return round(limit_price, 2)


# ╔══════════════════════════════════════════════════════════════╗
# ║  V6.3 状态机回测（四维度优化）                                ║
# ╚══════════════════════════════════════════════════════════════╝

@dataclass
class PositionV63:
    """V6.3持仓信息（增加限价相关字段 + 动态止损参数）"""
    entry_date: str
    entry_idx: int
    entry_price: float
    shares: int
    atr_at_entry: float
    yellow_at_entry: float
    hold_days: int = 0
    max_profit_pct: float = 0.0
    highest_high: float = 0.0
    chandelier_line: float = 0.0
    ts_code: str = ""
    limit_price: float = 0.0           # V6.3：限价买入价
    is_limit_filled: bool = False      # V6.3：是否限价成交
    dynamic_hard_stop_atr: float = 0.0  # V6.3：该笔交易的动态硬止损ATR倍数
    dynamic_chandelier_mult: float = 0.0  # V6.3：该笔交易的动态吊灯ATR倍数
    consecutive_bearish_vpa: int = 0       # 维度六：连续熊性VPA天数计数

    def update_peak(self, high_price: float):
        pct = (high_price - self.entry_price) / self.entry_price
        if pct > self.max_profit_pct:
            self.max_profit_pct = pct

    def update_chandelier(self, high_price: float, atr_mult: float):
        if high_price > self.highest_high:
            self.highest_high = high_price
        new_line = self.highest_high - atr_mult * self.atr_at_entry
        if new_line > self.chandelier_line:
            self.chandelier_line = new_line


def StatefulTradeBacktester_V63(
    df: pd.DataFrame,
    signal_col: str = "ambush_signal",
    initial_cash: float = 100000.0,
    params: Dict[str, Any] = None,
    market_allow_buy=None,
    ts_code: str = "",
    industry_allow_buy=None,
) -> List[TradeRecord]:
    """
    V6.3 状态机回测 —— V6.2 + 四维度优化

    与V6.2的关键区别：
      1. 行业过滤从绝对阈值→横截面RS排名
      2. 仓位从固定30%→波动率平价
      3. 信号从无微确认→VWAP/Inside Bar/VCP微观确认
      4. T+1执行从开盘价→限价单
    """
    if params is None:
        params = V63_PARAMS

    cash = initial_cash
    position = None
    trades = []
    pending_signal_idx = None
    pending_limit_price = 0.0  # V6.3：待执行的限价

    # 预计算UT/AD和VPA信号（维度六）—— 模块缺失时优雅降级
    _has_utad = False
    _has_vpa = False
    if params.get("utad_exit_enabled", True) or params.get("bearish_vpa_exit_enabled", True):
        try:
            if "is_ut_ad" not in df.columns:
                from classic_ta.wyckoff_analysis import detect_ut_ad, calc_support_resistance
                if "support_level" not in df.columns:
                    df = calc_support_resistance(df)
                df = detect_ut_ad(df)
            _has_utad = "is_ut_ad" in df.columns
        except ImportError:
            pass
        try:
            if "bearish_vpa_count" not in df.columns:
                from classic_ta.volume_price_analysis import run_vpa_analysis
                df = run_vpa_analysis(df)
            _has_vpa = "bearish_vpa_count" in df.columns
        except ImportError:
            pass

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
        current_high = float(row["High"])
        current_low = float(row["Low"])

        if pd.isna(current_price) or pd.isna(row.get("white_line", np.nan)):
            continue

        allow_buy = True
        if market_allow_buy is not None:
            try:
                allow_buy = bool(market_allow_buy.iloc[i])
            except (IndexError, KeyError):
                pass

        # 行业热度过滤
        if ind_allow_aligned is not None:
            try:
                date_idx = df.index[i]
                if date_idx in ind_allow_aligned.index:
                    ind_allow = ind_allow_aligned[date_idx]
                    if pd.notna(ind_allow) and not ind_allow:
                        allow_buy = False
            except (IndexError, KeyError):
                pass

        # ── T+1执行（V6.3：限价单逻辑）──
        if pending_signal_idx is not None and position is None:
            prev_row = df.iloc[pending_signal_idx]
            prev_close = float(prev_row["Close"])
            prev_yellow = float(prev_row["yellow_line"])
            prev_atr = float(prev_row["atr14"])

            # 防高开（保留V6.2逻辑）
            if current_open > prev_close + params["t1_high_open_atr"] * prev_atr:
                pending_signal_idx = None
                continue
            # 防破位（保留V6.2逻辑）
            if current_open < prev_yellow:
                pending_signal_idx = None
                continue

            if allow_buy and current_open > 0:
                # ── 维度四：限价单执行 ──
                if params.get("limit_order_enabled", False):
                    limit_price = pending_limit_price

                    # 情况1：开盘价直接低于限价 → 以开盘价成交（更优价格）
                    if current_open <= limit_price:
                        fill_price = current_open
                    # 情况2：盘中最低价触及限价 → 以限价成交
                    elif current_low <= limit_price:
                        fill_price = limit_price
                    # 情况3：全天未触及限价 → 放弃
                    else:
                        pending_signal_idx = None
                        continue
                else:
                    # V6.2原逻辑：直接以开盘价买入
                    fill_price = current_open

                # ── 维度二：波动率平价仓位（用当前总权益计算）──
                # 维度五：动态止损参数（建仓时计算，存入Position）
                dyn_hard_stop, dyn_chandelier = calc_dynamic_stop_params(df, i, params)
                # 当前总权益 = 现金 + 已有持仓市值
                current_equity = cash
                if position is not None:
                    current_equity += position.shares * current_price
                else:
                    current_equity = initial_cash
                shares = calc_volatility_parity_shares(
                    total_equity=current_equity,
                    entry_price=fill_price,
                    atr_at_entry=prev_atr,
                    hard_stop_atr=dyn_hard_stop,
                    params=params,
                )

                if shares > 0:
                    cost = shares * fill_price
                    if cost <= cash:
                        cash -= cost
                        position = PositionV63(
                            entry_date=df.index[i].strftime("%Y-%m-%d"),
                            entry_idx=i,
                            entry_price=fill_price,
                            shares=shares,
                            atr_at_entry=prev_atr,
                            yellow_at_entry=prev_yellow,
                            ts_code=ts_code,
                            limit_price=limit_price if params.get("limit_order_enabled", False) else 0,
                            is_limit_filled=(fill_price <= pending_limit_price) if params.get("limit_order_enabled", False) else False,
                            dynamic_hard_stop_atr=dyn_hard_stop,
                            dynamic_chandelier_mult=dyn_chandelier,
                        )
                        position.update_chandelier(float(row["High"]),
                                                    dyn_chandelier)

            pending_signal_idx = None
            continue

        # 信号检测
        if position is None and bool(row.get(signal_col, False)):
            if allow_buy:
                pending_signal_idx = i
                # 预计算限价
                if params.get("limit_order_enabled", False):
                    pending_limit_price = calc_limit_price(
                        float(row["Close"]),
                        float(row["yellow_line"]),
                        float(row["atr14"]),
                        params,
                    )

        # ── 退出判断（V6.2四级 + 维度六UT/AD保护）──
        if position is not None:
            position.hold_days += 1
            pnl_pct = (current_price - position.entry_price) / position.entry_price

            position.update_peak(float(row["High"]))

            # 维度六：UT/AD吊灯收紧
            effective_chandelier_mult = position.dynamic_chandelier_mult if position.dynamic_chandelier_mult > 0 else params.get("chandelier_atr_mult", 3.5)
            if _has_utad and params.get("utad_exit_enabled", True) and position.max_profit_pct * 100 > params.get("utad_min_profit_pct", 5.0):
                if "is_ut_ad" in df.columns and bool(row.get("is_ut_ad", False)):
                    effective_chandelier_mult = params.get("utad_tighten_chandelier", 1.5)

            position.update_chandelier(float(row["High"]), effective_chandelier_mult)

            # 维度六：更新连续熊性VPA天数
            if _has_vpa and params.get("bearish_vpa_exit_enabled", True) and "bearish_vpa_count" in df.columns:
                if int(row.get("bearish_vpa_count", 0)) >= params.get("bearish_vpa_min_count", 2):
                    position.consecutive_bearish_vpa += 1
                else:
                    position.consecutive_bearish_vpa = 0
            else:
                position.consecutive_bearish_vpa = 0

            exit_reason = None

            # 优先级1：硬止损（使用该笔交易的动态硬止损参数）
            pos_hard_stop_atr = position.dynamic_hard_stop_atr if position.dynamic_hard_stop_atr > 0 else params["hard_stop_atr"]
            atr_stop = -(pos_hard_stop_atr * position.atr_at_entry / position.entry_price)
            if pnl_pct <= atr_stop:
                exit_reason = ExitReason.ATR_HARD_STOP

            # 优先级2：吊灯止盈
            elif (current_price < position.chandelier_line
                  and position.hold_days >= params.get("chandelier_min_days", 2)):
                exit_reason = ExitReason.CHANDELIER_EXIT

            # 优先级3：Buy Climax精细化
            elif detect_buy_climax_v61(row, position.max_profit_pct, params):
                exit_reason = ExitReason.BUY_CLIMAX

            # 优先级3.5：VPA派发信号（维度六）
            elif (_has_vpa and params.get("bearish_vpa_exit_enabled", True)
                  and position.consecutive_bearish_vpa >= params.get("bearish_vpa_consecutive_days", 2)):
                exit_reason = ExitReason.VPA_DISTRIBUTION

            # 优先级4：动态时间止损（浮盈>5%延长持仓）
            else:
                _float_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                _extend_threshold = params.get("time_stop_extend_profit_pct", 5.0)
                _extend_days = params.get("time_stop_extend_days", 20)
                _base_days = params["time_stop_days"]
                # 浮盈超过阈值 → 使用延长天数；否则使用基础天数
                _effective_days = _extend_days if _float_pct >= _extend_threshold else _base_days
                if (position.hold_days >= _effective_days
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


# ╔══════════════════════════════════════════════════════════════╗
# ║  便捷入口                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def run_v63_backtest(df, params=None, market_allow_buy=None, ts_code="",
                     industry_allow_buy=None):
    df = IndicatorCalcBase(df)
    df = add_micro_confirm_indicators(df)
    df = Detect_AmbushSignal_V63(df, params)
    trades = StatefulTradeBacktester_V63(
        df, signal_col="ambush_signal", params=params,
        market_allow_buy=market_allow_buy, ts_code=ts_code,
        industry_allow_buy=industry_allow_buy,
    )
    return trades
