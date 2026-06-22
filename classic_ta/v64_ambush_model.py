"""
潜伏模型 V6.4 —— V6.3 + 维度七：主力托底评分（Institutional Support Score）
============================================================================
核心思想：通过盘面走势量化"情绪冰点 + 主力托着"的组合特征，提高买入后胜率。

V6.3已有体系：
  - SOS锚定 → 确认主力已入场
  - J值超卖 + 缩量 + 小实体 → 情绪冰点
  - 微观确认（VWAP/VCP）→ 止跌信号

V6.4新增维度七 —— 主力托底评分（4个子因子，0~4分）：

  因子A：缩量企稳持续性（Sustained Volume Stabilization）
    逻辑：连续2~3天缩量 且 价格不再创新低
    含义：抛压已真正枯竭，不是暂时的"没人卖也没人买"
    评分：0/1分

  因子B：量价底背离（Volume-Price Bullish Divergence）
    逻辑：近N日价格创新低 但 OBV不创新低（或成交量未放大）
    含义：价格下跌但卖压未加剧 = 有资金在低位悄悄吃货
    评分：0/1分

  因子C：支撑反复试探不破（Support Holding）
    逻辑：近N日价格多次（2+次）触碰黄线附近但都收回
    含义：每次探到支撑都有买盘托住 = 主力在关键位护盘
    评分：0/1分

  因子D：日内承接信号（Intraday Accumulation）
    逻辑：近3日内出现 长下影线/收在日高附近/缩量小阳 等承接特征
    含义：盘中有人在低位主动买入，不让价格继续跌
    评分：0/1分

  总分 = A + B + C + D（0~4分）
  信号过滤：ambush_signal & (support_score >= threshold)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List

from classic_ta.v60_ambush_model import IndicatorCalcBase, DEFAULT_PARAMS
from classic_ta.v61_ambush_model import (
    Position, TradeRecord, ExitReason,
    Detect_AmbushSignal_V61, detect_buy_climax_v61,
    StatefulTradeBacktester_V61, compute_v61_metrics, V61_PARAMS,
)
from classic_ta.v62_ambush_model import compute_industry_momentum, V62_PARAMS
from classic_ta.v63_ambush_model import (
    add_micro_confirm_indicators,
    Detect_AmbushSignal_V63,
    StatefulTradeBacktester_V63,
    calc_volatility_parity_shares,
    calc_dynamic_stop_params,
    calc_limit_price,
    compute_industry_rs_matrix,
    build_industry_allow_matrix_v63,
    PositionV63,
    V63_PARAMS,
)


# ╔══════════════════════════════════════════════════════════════╗
# ║  V6.4 参数                                                  ║
# ╚══════════════════════════════════════════════════════════════╝

V64_PARAMS = V63_PARAMS.copy()
V64_PARAMS.update({
    # --- 入场质量评分（V6.4.8：盘面走势+量能+J值+黄白线）---
    "entry_quality_enabled": True,
    "entry_quality_min_score": 3,        # 最低入场质量分（0~8）

    # E1: J值深度（情绪冰点程度）
    "eq_j_enabled": True,
    "eq_j_extreme": 0,                   # J<0 = 极度超卖 (2分)
    "eq_j_very_oversold": 5,             # J<5 = 非常超卖 (1分)
    "eq_j_oversold": 13,                 # J<13 = 超卖 (0分)

    # E2: 量能枯竭度（成交量萎缩程度）
    "eq_vol_enabled": True,
    "eq_vol_extreme": 0.30,              # 量<30%均量 = 极度枯竭 (2分)
    "eq_vol_very_low": 0.50,             # 量<50%均量 = 非常萎缩 (1分)
    "eq_vol_low": 0.70,                  # 量<70%均量 = 萎缩 (0分)

    # E3: 盘面形态（蜡烛图质量）
    "eq_candle_enabled": True,
    "eq_candle_shadow_ratio": 2.0,       # 下影线>实体×2
    "eq_candle_body_pct": 0.015,         # 实体<1.5%
    "eq_candle_max_score": 2,            # 最多2分

    # E4: 黄白线关系（均线结构）
    "eq_ma_enabled": True,
    "eq_ma_cross_lookback": 3,           # 金叉判断回看天数
    "eq_ma_converge_atr": 0.5,           # 白线接近黄线的ATR容差

    # V6.4.9：趋势方向二级过滤（黄线上升 = 底部抬升）
    "eq_trend_dir_enabled": True,
    "eq_trend_dir_lookback": 5,          # 黄线斜率回看天数
    "eq_trend_dir_min_score": 3,         # 入场质量评分>=此值时启用趋势过滤

    # V6.4.9：评分=3子模式过滤（排除V=1且J=0的弱组合）
    "eq_sub_filter_enabled": True,
    "eq_sub_filter_min_score": 3,
    "eq_sub_filter_max_score": 3,        # 仅对score=3启用
    "eq_sub_filter_exclude_v1_j0": True, # 排除V=1且J=0

    # V6.4.10：评分=4子模式过滤（排除J1V0C1M2弱组合）
    # 回测验证: J1V0C1M2胜率21.7%, 过滤后总收益+63.7pp
    "eq_sub_filter_score4_enabled": False,  # 默认关闭，在BEST_PARAMS中启用

    # P2: 多日确认评分（E7，加分项）
    "eq_persist_enabled": False,          # 默认关闭，待回测验证
    "eq_persist_lookback": 2,             # 回看天数（T-1, T-2）
    "eq_persist_min_conds": 3,            # 近信号最少满足条件数（3/5）
    "eq_persist_score": 1,                # 加分值

    # --- 旧因子体系（保留用于对比）---
    "inst_support_enabled": False,       # 关闭旧的B+C+D评分
    "factor_d_required": False,

    # V6.4.5：快速验证退出（禁用，本策略需要耐心）
    "early_exit_enabled": False,
    "early_exit_days": 3,
    "early_exit_min_profit_pct": -0.05,

    # V6.4.5：缩短时间止损（10天→7天）
    "time_stop_days": 7,

    # V6.4.8：差异化仓位（高分信号加大仓位）
    "score_position_enabled": True,
    "score_position_mult": {0: 0.7, 1: 0.7, 2: 0.7, 3: 1.0, 4: 1.0, 5: 1.2, 6: 1.3, 7: 1.5, 8: 1.5},

    # V6.4.8：差异化持仓时间（高分信号给更多时间）
    "score_time_stop_enabled": True,
    "score_time_stop_days": {0: 5, 1: 5, 2: 5, 3: 7, 4: 7, 5: 8, 6: 9, 7: 10, 8: 10},

    # V6.4.7：保本止损（浮盈达标后止损上移到成本价）
    "breakeven_stop_enabled": True,
    "breakeven_trigger_pct": 0.03,       # 浮盈达到3%时激活保本
    "breakeven_min_profit_pct": 0.005,   # 保本时最少保留0.5%利润
})


# ╔══════════════════════════════════════════════════════════════╗
# ║  辅助指标：OBV（On-Balance Volume）                         ║
# ╚══════════════════════════════════════════════════════════════╝

def calc_obv(df: pd.DataFrame) -> pd.Series:
    """计算OBV（能量潮指标）

    逻辑：
      收盘价上涨 → OBV += 当日成交量
      收盘价下跌 → OBV -= 当日成交量
      收盘价不变 → OBV 不变

    原理：OBV持续上升说明有资金持续流入（即使价格暂时下跌）
    """
    direction = np.sign(df["Close"].diff())
    direction.iloc[0] = 0
    obv = (direction * df["Volume"]).cumsum()
    return obv


# ╔══════════════════════════════════════════════════════════════╗
# ║  入场质量评分（Entry Quality Score）V6.4.8                      ║
# ║  维度：J值深度 + 量能枯竭 + 盘面形态 + 黄白线关系              ║
# ╚══════════════════════════════════════════════════════════════╝

def add_entry_quality_indicators(df: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
    """计算入场质量评分（0~8分）

    追加列：
      - eq_j_score: J值深度得分 (0/1/2)
      - eq_vol_score: 量能枯竭得分 (0/1/2)
      - eq_candle_score: 盘面形态得分 (0/1/2)
      - eq_ma_score: 黄白线关系得分 (0/1/2)
      - entry_quality_score: 综合评分 (0~8)
    """
    if params is None:
        params = V64_PARAMS

    body = (df["Close"] - df["Open"]).abs()
    lower_shadow = df[["Close", "Open"]].min(axis=1) - df["Low"]

    # E1: J值深度 (0~2分)
    if params.get("eq_j_enabled", True):
        j_extreme = params.get("eq_j_extreme", 0)
        j_very = params.get("eq_j_very_oversold", 5)
        df["eq_j_score"] = 0
        df.loc[df["J"] < j_very, "eq_j_score"] = 1
        df.loc[df["J"] < j_extreme, "eq_j_score"] = 2
    else:
        df["eq_j_score"] = 0

    # E2: 量能枯竭度 (0~2分)
    if params.get("eq_vol_enabled", True):
        vol_extreme = params.get("eq_vol_extreme", 0.30)
        vol_very = params.get("eq_vol_very_low", 0.50)
        vol_ratio = df["Volume"] / (df["volume_ma"] + 1e-8)
        df["eq_vol_score"] = 0
        df.loc[vol_ratio < vol_very, "eq_vol_score"] = 1
        df.loc[vol_ratio < vol_extreme, "eq_vol_score"] = 2
    else:
        df["eq_vol_score"] = 0

    # E3: 盘面形态 (0~2分)
    if params.get("eq_candle_enabled", True):
        shadow_ratio = params.get("eq_candle_shadow_ratio", 2.0)
        body_pct = params.get("eq_candle_body_pct", 0.015)
        max_score = params.get("eq_candle_max_score", 2)

        has_long_shadow = lower_shadow > body * shadow_ratio
        has_small_body = body / (df["Close"] + 1e-8) < body_pct
        is_bullish = df["Close"] > df["Open"]

        candle_raw = has_long_shadow.astype(int) + has_small_body.astype(int) + is_bullish.astype(int)
        df["eq_candle_score"] = candle_raw.clip(0, max_score)
    else:
        df["eq_candle_score"] = 0

    # E4: 黄白线关系 (0~2分)
    if params.get("eq_ma_enabled", True):
        lookback = params.get("eq_ma_cross_lookback", 3)
        converge_atr = params.get("eq_ma_converge_atr", 0.5)

        white = df["white_line"]
        yellow = df["yellow_line"]

        is_above = white > yellow
        was_below = white.shift(1) <= yellow.shift(1)
        golden_cross = (white > yellow) & was_below
        golden_cross_w = golden_cross.rolling(lookback, min_periods=1).max().astype(bool)
        is_converging = (white - yellow).abs() < converge_atr * df["atr14"]

        df["eq_ma_score"] = 0
        df.loc[is_converging, "eq_ma_score"] = 1
        df.loc[golden_cross_w, "eq_ma_score"] = 1
        df.loc[is_above, "eq_ma_score"] = 2
    else:
        df["eq_ma_score"] = 0

    # 综合评分（E1-E4，0-8分）
    base_score = (
        df["eq_j_score"] + df["eq_vol_score"]
        + df["eq_candle_score"] + df["eq_ma_score"]
    )

    # E7: 多日确认评分（Persistent Near-Miss）
    # 如果T-1或T-2日也接近信号条件（≥3/5条件满足），说明蓄力更充分
    if params.get("eq_persist_enabled", False):
        lookback = params.get("eq_persist_lookback", 2)
        min_conds = params.get("eq_persist_min_conds", 3)
        persist_score_val = params.get("eq_persist_score", 1)

        # 直接计算每日满足条件数（不依赖Detect_AmbushSignal的输出）
        if "near_signal_count" in df.columns:
            nsc = df["near_signal_count"]
        else:
            # 重新计算5个基础条件
            body_p = (df["Close"] - df["Open"]).abs()
            window = params.get("ambush_window", 8)
            j_os = params.get("ambush_j_oversold", 5)
            vol_sh = params.get("ambush_vol_shrink", 0.70)
            body_pct = params.get("ambush_body_pct", 0.03)
            sup_atr = params.get("ambush_support_atr", 0.5)

            # 条件1: SOS（简化：用tag_sos_anchor如果存在）
            if "tag_sos_anchor" in df.columns:
                sos_w = df["tag_sos_anchor"].shift(1).rolling(window, min_periods=1).max().astype(bool)
            else:
                sos_w = pd.Series(True, index=df.index)  # 默认True（不限制）

            j_ok = df["J"] < j_os
            vol_ok = df["Volume"] < df["volume_ma"] * vol_sh
            body_ok = body_p / (df["Close"] + 1e-8) < body_pct
            sup_ok = df["Close"] > df["yellow_line"] - sup_atr * df["atr14"]

            nsc = (sos_w.astype(int) + j_ok.astype(int) + vol_ok.astype(int)
                   + body_ok.astype(int) + sup_ok.astype(int))

        prev_near = nsc.shift(1) >= min_conds
        prev2_near = nsc.shift(2) >= min_conds if lookback >= 2 else pd.Series(False, index=df.index)

        df["eq_persist_score"] = 0
        df.loc[prev_near | prev2_near, "eq_persist_score"] = persist_score_val
    else:
        df["eq_persist_score"] = 0

    # 最终综合评分（0-9分）
    df["entry_quality_score"] = base_score + df["eq_persist_score"]

    # E5: 趋势方向（黄线是否上升 = 底部是否抬升）
    if params.get("eq_trend_dir_enabled", True):
        lookback = params.get("eq_trend_dir_lookback", 5)
        yellow = df["yellow_line"]
        # 黄线N日前 vs 现在：上升 = True
        df["eq_trend_rising"] = yellow > yellow.shift(lookback)
    else:
        df["eq_trend_rising"] = True  # 默认不过滤

    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║  维度七：主力托底评分（Institutional Support Score）         ║
# ╚══════════════════════════════════════════════════════════════╝

def add_inst_support_indicators(df: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
    """在DataFrame上追加主力托底评分的各项指标列

    追加列：
      - obv: OBV能量潮
      - factor_a_vol_stable: 因子A - 缩量企稳
      - factor_b_vp_divergence: 因子B - 量价底背离
      - factor_c_support_hold: 因子C - 支撑试探不破
      - factor_d_intraday_accum: 因子D - 日内承接
      - inst_support_score: 综合评分（0~4）
    """
    if params is None:
        params = V64_PARAMS

    # ── 预计算共用指标 ──
    body = (df["Close"] - df["Open"]).abs()
    lower_shadow = df[["Close", "Open"]].min(axis=1) - df["Low"]
    upper_shadow = df["High"] - df[["Close", "Open"]].max(axis=1)
    amplitude = df["High"] - df["Low"]

    # OBV
    if "obv" not in df.columns:
        df["obv"] = calc_obv(df)

    # ──────────────────────────────────────────────────────────
    #  因子A：缩量企稳持续性
    #  连续N天：缩量（vol < vol_ma × ratio）且 价格未创新低
    # ──────────────────────────────────────────────────────────
    if params.get("factor_a_enabled", True):
        consec = params.get("factor_a_consec_days", 2)
        vol_ratio = params.get("factor_a_vol_ratio", 0.85)
        stable_atr = params.get("factor_a_price_stable_atr", 0.5)

        # 单日缩量
        is_vol_shrink = df["Volume"] < df["volume_ma"] * vol_ratio

        # 单日价格未创新低（相对近N日最低收盘价）
        rolling_low = df["Close"].rolling(consec + 3, min_periods=consec).min().shift(1)
        is_not_new_low = df["Close"] >= rolling_low - stable_atr * df["atr14"]

        # 单日企稳
        is_stable_day = is_vol_shrink & is_not_new_low

        # 连续N天都满足
        consec_satisfy = is_stable_day.rolling(consec, min_periods=consec).sum() >= consec
        df["factor_a_vol_stable"] = consec_satisfy
    else:
        df["factor_a_vol_stable"] = False

    # ──────────────────────────────────────────────────────────
    #  因子B：量价底背离
    #  近N日价格创新低 但 OBV/成交量 不创新低
    #  含义：价格虽跌但卖压未加剧 = 有资金在低位吸筹
    # ──────────────────────────────────────────────────────────
    if params.get("factor_b_enabled", True):
        lookback = params.get("factor_b_lookback", 10)
        use_obv = params.get("factor_b_obv_enabled", True)

        # 价格是否处于近N日低位
        at_low_pct = params.get("factor_b_at_low_pct", 1.01)
        rolling_low_close = df["Close"].rolling(lookback, min_periods=max(lookback // 2, 3)).min()
        is_at_low = df["Close"] <= rolling_low_close * at_low_pct

        if use_obv:
            # OBV背离：价格创新低 但 OBV未创新低
            rolling_low_obv = df["obv"].rolling(lookback, min_periods=max(lookback // 2, 3)).min()
            obv_margin = params.get("factor_b_obv_margin", 1.05)
            obv_not_new_low = df["obv"] > rolling_low_obv * obv_margin
            df["factor_b_vp_divergence"] = is_at_low & obv_not_new_low
        else:
            # 量能背离：价格创新低 但 成交量未放大（相对近N日平均）
            vol_ma_n = df["Volume"].rolling(lookback, min_periods=max(lookback // 2, 3)).mean()
            vol_not_expand = df["Volume"] < vol_ma_n * 1.0  # 量能未超过N日均量
            df["factor_b_vp_divergence"] = is_at_low & vol_not_expand
    else:
        df["factor_b_vp_divergence"] = False

    # ──────────────────────────────────────────────────────────
    #  因子C：支撑反复试探不破
    #  近N日价格多次触碰黄线附近 但 每次都收回（收盘>黄线）
    #  含义：主力在关键支撑位护盘，不让价格有效跌破
    # ──────────────────────────────────────────────────────────
    if params.get("factor_c_enabled", True):
        lookback = params.get("factor_c_lookback", 8)
        touch_atr = params.get("factor_c_touch_atr", 0.8)
        min_touches = params.get("factor_c_min_touches", 2)

        # 触碰黄线：最低价触及黄线±ATR容差范围内
        is_touching_support = (
            (df["Low"] <= df["yellow_line"] + touch_atr * df["atr14"])
            & (df["Low"] >= df["yellow_line"] - touch_atr * df["atr14"])
        )

        # 收回：收盘价站在黄线之上
        is_holding = df["Close"] >= df["yellow_line"] - 0.1 * df["atr14"]

        # 有效试探 = 触碰 + 收回
        is_valid_test = is_touching_support & is_holding

        # 近N日内有效试探次数
        touch_count = is_valid_test.rolling(lookback, min_periods=1).sum()
        df["factor_c_support_hold"] = touch_count >= min_touches
    else:
        df["factor_c_support_hold"] = False

    # ──────────────────────────────────────────────────────────
    #  因子D：日内承接信号
    #  近3日内出现以下承接特征之一：
    #    d1: 长下影线（lower_shadow > body × ratio）→ 低位有买盘接住
    #    d2: 收在日高附近（(C-L)/(H-L) > pct）→ 尾盘有资金拉升
    #    d3: 缩量小阳（收阳 + 缩量 + 小实体）→ 主力静默吸筹
    # ──────────────────────────────────────────────────────────
    if params.get("factor_d_enabled", True):
        lookback = params.get("factor_d_lookback", 3)
        shadow_ratio = params.get("factor_d_lower_shadow_ratio", 1.5)
        near_high_pct = params.get("factor_d_close_near_high_pct", 0.65)
        min_signals = params.get("factor_d_min_signals", 1)

        # d1: 长下影线
        d1_lower_shadow = lower_shadow > body * shadow_ratio

        # d2: 收在日高附近
        close_position = (df["Close"] - df["Low"]) / (amplitude + 1e-8)
        d2_near_high = close_position > near_high_pct

        # d3: 缩量小阳（收阳 + 量<均量×0.8 + 实体<收盘×3%）
        is_bullish = df["Close"] > df["Open"]
        is_low_vol = df["Volume"] < df["volume_ma"] * 0.8
        is_small_body = body / (df["Close"] + 1e-8) < 0.03
        d3_quiet_accum = is_bullish & is_low_vol & is_small_body

        # 综合承接信号（任一满足）
        is_accum_day = d1_lower_shadow | d2_near_high | d3_quiet_accum

        # 近N日内出现>=min_signals天承接
        accum_count = is_accum_day.rolling(lookback, min_periods=1).sum()
        df["factor_d_intraday_accum"] = accum_count >= min_signals
    else:
        df["factor_d_intraday_accum"] = False

    # ──────────────────────────────────────────────────────────
    #  综合评分（各因子等权，总分0~4分）
    # ──────────────────────────────────────────────────────────
    w_a = params.get("factor_a_weight", 1)
    w_b = params.get("factor_b_weight", 1)
    w_c = params.get("factor_c_weight", 1)
    w_d = params.get("factor_d_weight", 1)

    df["inst_support_score"] = (
        df["factor_a_vol_stable"].astype(int) * w_a
        + df["factor_b_vp_divergence"].astype(int) * w_b
        + df["factor_c_support_hold"].astype(int) * w_c
        + df["factor_d_intraday_accum"].astype(int) * w_d
    )

    return df


def Detect_AmbushSignal_V64(df: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
    """V6.4 潜伏信号引擎 —— V6.3 + 入场质量评分过滤

    在V6.3信号基础上，追加入场质量评分过滤：
      ambush_signal = V6.3信号 & (entry_quality_score >= min_score)
    """
    if params is None:
        params = V64_PARAMS

    # 先跑V6.3信号（含微观确认）
    df = Detect_AmbushSignal_V63(df, params)

    # 入场质量评分过滤（V6.4.8）
    if params.get("entry_quality_enabled", False):
        if "entry_quality_score" not in df.columns:
            df = add_entry_quality_indicators(df, params)
        min_score = params.get("entry_quality_min_score", 3)
        quality_filter = df["entry_quality_score"] >= min_score

        # V6.4.9：趋势方向二级过滤（评分>=指定值时，要求黄线上升）
        if params.get("eq_trend_dir_enabled", False):
            trend_min = params.get("eq_trend_dir_min_score", 3)
            trend_filter = (
                (df["entry_quality_score"] < trend_min)
                | df["eq_trend_rising"]
            )
            quality_filter = quality_filter & trend_filter

        # V6.4.9：评分=3子模式过滤（排除弱组合）
        if params.get("eq_sub_filter_enabled", False):
            sub_min = params.get("eq_sub_filter_min_score", 3)
            sub_max = params.get("eq_sub_filter_max_score", 3)
            is_target_score = (df["entry_quality_score"] >= sub_min) & (df["entry_quality_score"] <= sub_max)

            if params.get("eq_sub_filter_exclude_v1_j0", True):
                # 排除 V=1 且 J=0 的组合（J0V1C0M2胜率42%，远低于其他score=3）
                weak_pattern = (df["eq_vol_score"] == 1) & (df["eq_j_score"] == 0)
                sub_filter = ~is_target_score | ~weak_pattern
                quality_filter = quality_filter & sub_filter

        # V6.4.10：评分=4子模式过滤（排除J1V0C1M2弱组合）
        # 回测验证: J1V0C1M2胜率21.7%, 均收益-2.77%, 总贡献-63.7%
        # "形态好但J不深+量不枯"的4分信号质量差
        if params.get("eq_sub_filter_score4_enabled", False):
            is_score4 = df["entry_quality_score"] == 4
            weak_j1v0c1m2 = (
                (df["eq_j_score"] == 1) &
                (df["eq_vol_score"] == 0) &
                (df["eq_candle_score"] == 1) &
                (df["eq_ma_score"] == 2)
            )
            score4_filter = ~is_score4 | ~weak_j1v0c1m2
            quality_filter = quality_filter & score4_filter

        df["ambush_signal"] = df["ambush_signal"] & quality_filter

    # 旧的托底评分过滤（当启用时）
    if params.get("inst_support_enabled", False):
        if "inst_support_score" not in df.columns:
            df = add_inst_support_indicators(df, params)
        min_score = params.get("inst_support_min_score", 1)
        support_filter = df["inst_support_score"] >= min_score
        if params.get("factor_a_required", False):
            support_filter = support_filter & df["factor_a_vol_stable"]
        if params.get("factor_d_required", False):
            support_filter = support_filter & df["factor_d_intraday_accum"]
        df["ambush_signal"] = df["ambush_signal"] & support_filter

    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║  V6.4 状态机回测（增加托底评分记录）                          ║
# ╚══════════════════════════════════════════════════════════════╝

@dataclass
class PositionV64(PositionV63):
    """V6.4持仓信息（增加托底评分记录）"""
    inst_support_score: int = 0
    factor_details: str = ""  # "A1B0C1D1" 格式


def StatefulTradeBacktester_V64(
    df: pd.DataFrame,
    signal_col: str = "ambush_signal",
    initial_cash: float = 100000.0,
    params: Dict[str, Any] = None,
    market_allow_buy=None,
    ts_code: str = "",
    industry_allow_buy=None,
) -> List[TradeRecord]:
    """V6.4 状态机回测 —— V6.3 + 主力托底评分

    与V6.3的关键区别：
      1. 信号过滤增加主力托底评分
      2. TradeRecord中记录托底评分（便于分析评分与胜率的关系）
    """
    if params is None:
        params = V64_PARAMS

    cash = initial_cash
    position = None
    trades = []
    pending_signal_idx = None
    pending_limit_price = 0.0
    pending_support_score = 0
    pending_factor_details = ""

    # 预计算UT/AD和VPA信号（继承V6.3逻辑）
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
                # 限价单执行
                if params.get("limit_order_enabled", False):
                    limit_price = pending_limit_price
                    if current_open <= limit_price:
                        fill_price = current_open
                    elif current_low <= limit_price:
                        fill_price = limit_price
                    else:
                        pending_signal_idx = None
                        continue
                else:
                    fill_price = current_open

                # 波动率平价仓位 + 动态止损
                dyn_hard_stop, dyn_chandelier = calc_dynamic_stop_params(df, i, params)
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

                # V6.4.6：差异化仓位（高分信号加大仓位）
                if params.get("score_position_enabled", False):
                    score_mults = params.get("score_position_mult", {})
                    score_mult = score_mults.get(pending_support_score, 1.0)
                    shares = int(shares * score_mult)

                if shares > 0:
                    cost = shares * fill_price
                    if cost <= cash:
                        cash -= cost
                        position = PositionV64(
                            entry_date=df.index[i].strftime("%Y-%m-%d"),
                            entry_idx=i,
                            entry_price=fill_price,
                            shares=shares,
                            atr_at_entry=prev_atr,
                            yellow_at_entry=prev_yellow,
                            ts_code=ts_code,
                            limit_price=pending_limit_price if params.get("limit_order_enabled", False) else 0,
                            is_limit_filled=(fill_price <= pending_limit_price) if params.get("limit_order_enabled", False) else False,
                            dynamic_hard_stop_atr=dyn_hard_stop,
                            dynamic_chandelier_mult=dyn_chandelier,
                            inst_support_score=pending_support_score,
                            factor_details=pending_factor_details,
                        )
                        position.update_chandelier(float(row["High"]), dyn_chandelier)

            pending_signal_idx = None
            continue

        # 信号检测
        if position is None and bool(row.get(signal_col, False)):
            if allow_buy:
                pending_signal_idx = i
                # 记录入场质量评分（优先）或托底评分
                if "entry_quality_score" in df.columns and params.get("entry_quality_enabled", False):
                    pending_support_score = int(row.get("entry_quality_score", 0))
                    ej = int(row.get("eq_j_score", 0))
                    ev = int(row.get("eq_vol_score", 0))
                    ec = int(row.get("eq_candle_score", 0))
                    em = int(row.get("eq_ma_score", 0))
                    pending_factor_details = f"J{ej}V{ev}C{ec}M{em}"
                else:
                    pending_support_score = int(row.get("inst_support_score", 0))
                    a = int(row.get("factor_a_vol_stable", False))
                    b = int(row.get("factor_b_vp_divergence", False))
                    c = int(row.get("factor_c_support_hold", False))
                    d = int(row.get("factor_d_intraday_accum", False))
                    pending_factor_details = f"A{a}B{b}C{c}D{d}"
                # 预计算限价
                if params.get("limit_order_enabled", False):
                    pending_limit_price = calc_limit_price(
                        float(row["Close"]),
                        float(row["yellow_line"]),
                        float(row["atr14"]),
                        params,
                    )

        # ── 退出判断（完全继承V6.3逻辑）──
        if position is not None:
            position.hold_days += 1
            pnl_pct = (current_price - position.entry_price) / position.entry_price

            position.update_peak(float(row["High"]))

            # UT/AD吊灯收紧
            effective_chandelier_mult = position.dynamic_chandelier_mult if position.dynamic_chandelier_mult > 0 else params.get("chandelier_atr_mult", 3.5)
            if _has_utad and params.get("utad_exit_enabled", True) and position.max_profit_pct * 100 > params.get("utad_min_profit_pct", 5.0):
                if "is_ut_ad" in df.columns and bool(row.get("is_ut_ad", False)):
                    effective_chandelier_mult = params.get("utad_tighten_chandelier", 1.5)

            position.update_chandelier(float(row["High"]), effective_chandelier_mult)

            # 更新连续熊性VPA天数
            if _has_vpa and params.get("bearish_vpa_exit_enabled", True) and "bearish_vpa_count" in df.columns:
                if int(row.get("bearish_vpa_count", 0)) >= params.get("bearish_vpa_min_count", 2):
                    position.consecutive_bearish_vpa += 1
                else:
                    position.consecutive_bearish_vpa = 0
            else:
                position.consecutive_bearish_vpa = 0

            exit_reason = None

            # 优先级0.5：保本止损（V6.4.7：浮盈达标后回调到成本价即走）
            if (params.get("breakeven_stop_enabled", False)
                  and position.max_profit_pct >= params.get("breakeven_trigger_pct", 0.03)
                  and pnl_pct <= params.get("breakeven_min_profit_pct", 0.005)):
                exit_reason = ExitReason.BREAKEVEN_STOP

            # 优先级1：硬止损
            if exit_reason is None:
                pos_hard_stop_atr = position.dynamic_hard_stop_atr if position.dynamic_hard_stop_atr > 0 else params["hard_stop_atr"]
                atr_stop = -(pos_hard_stop_atr * position.atr_at_entry / position.entry_price)
                if pnl_pct <= atr_stop:
                    exit_reason = ExitReason.ATR_HARD_STOP

            # 优先级2：吊灯止盈
            if exit_reason is None and (current_price < position.chandelier_line
                  and position.hold_days >= params.get("chandelier_min_days", 2)):
                exit_reason = ExitReason.CHANDELIER_EXIT

            # 优先级3：Buy Climax精细化
            if exit_reason is None and detect_buy_climax_v61(row, position.max_profit_pct, params):
                exit_reason = ExitReason.BUY_CLIMAX

            # 优先级3.5：VPA派发信号
            if exit_reason is None and (_has_vpa and params.get("bearish_vpa_exit_enabled", True)
                  and position.consecutive_bearish_vpa >= params.get("bearish_vpa_consecutive_days", 2)):
                exit_reason = ExitReason.VPA_DISTRIBUTION

            # 优先级3.8：快速验证退出（V6.4.5：N天不涨则走）
            elif (params.get("early_exit_enabled", False)
                  and position.hold_days >= params.get("early_exit_days", 3)
                  and pnl_pct < params.get("early_exit_min_profit_pct", 0.0)):
                exit_reason = ExitReason.EARLY_EXIT

            # 优先级4：动态时间止损
            else:
                _float_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                _extend_threshold = params.get("time_stop_extend_profit_pct", 5.0)
                _extend_days = params.get("time_stop_extend_days", 20)
                # V6.4.6：差异化持仓时间（高分信号给更多时间）
                if params.get("score_time_stop_enabled", False):
                    score_days = params.get("score_time_stop_days", {})
                    _base_days = score_days.get(position.inst_support_score, params["time_stop_days"])
                else:
                    _base_days = params["time_stop_days"]
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
                    stock_name=position.factor_details,  # V6.4：在stock_name字段记录因子详情
                ))
                position = None

    return trades


# ╔══════════════════════════════════════════════════════════════╗
# ║  V6.4 回测指标分析（按评分分组统计）                          ║
# ╚══════════════════════════════════════════════════════════════╝

def analyze_support_score_impact(trades: List[TradeRecord]) -> Dict[str, Any]:
    """按评分分组分析交易表现

    支持旧格式 (A1B0C1D1) 和新格式 (J2V1C2M1)
    """
    if not trades:
        return {}

    from collections import defaultdict
    score_groups = defaultdict(list)

    for t in trades:
        details = t.stock_name or ""
        # 提取评分：求和所有数字字符
        score = sum(int(c) for c in details if c.isdigit()) if details else 0
        score_groups[score].append(t)

    result = {}
    for score, group_trades in sorted(score_groups.items()):
        wins = [t for t in group_trades if t.profit_pct > 0]
        total = len(group_trades)
        avg_profit = np.mean([t.profit_pct for t in group_trades]) if total > 0 else 0
        win_rate = len(wins) / total * 100 if total > 0 else 0

        result[f"score_{score}"] = {
            "trades": total,
            "win_rate": round(win_rate, 1),
            "avg_profit": round(float(avg_profit), 2),
            "factor_pattern": _count_patterns(group_trades),
        }

    return result


def _count_patterns(trades: List[TradeRecord]) -> Dict[str, int]:
    """统计各因子组合模式的出现次数"""
    from collections import Counter
    patterns = Counter()
    for t in trades:
        if t.stock_name:
            patterns[t.stock_name] += 1
    return dict(patterns.most_common(5))


# ╔══════════════════════════════════════════════════════════════╗
# ║  便捷入口                                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def run_v64_backtest(df, params=None, market_allow_buy=None, ts_code="",
                     industry_allow_buy=None):
    """V6.4便捷回测入口"""
    if params is None:
        params = V64_PARAMS
    df = IndicatorCalcBase(df)
    df = add_micro_confirm_indicators(df)
    df = add_inst_support_indicators(df, params)
    df = Detect_AmbushSignal_V64(df, params)
    trades = StatefulTradeBacktester_V64(
        df, signal_col="ambush_signal", params=params,
        market_allow_buy=market_allow_buy, ts_code=ts_code,
        industry_allow_buy=industry_allow_buy,
    )
    return trades
