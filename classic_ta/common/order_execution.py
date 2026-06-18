"""
限价单成交判定模块
===========================
提供生产级别的限价单成交判定逻辑，考虑滑点、安全垫和流动性。

核心改进（相对V6.3原始逻辑）：
  1. 引入 tick_buffer 安全垫：避免"擦边成交"的未来函数
  2. 引入 slippage_pct 滑点：模拟真实交易冲击成本
  3. 引入 volume_depth_check：低流动性时不成交
  4. 处理开盘跳空（低开/高开）的边界情况
"""

import logging

logger = logging.getLogger(__name__)


# 默认参数
DEFAULT_EXECUTION_PARAMS = {
    "slippage_pct": 0.001,       # 0.1% 滑点（冲击成本）
    "tick_buffer": 0.01,          # 1分钱安全垫（确保价格真正穿越限价位）
    "volume_depth_check": True,   # 启用成交量深度检查
    "min_volume_ratio": 0.5,      # 最低成交量比（相对均量），低于此不成交
}


def check_limit_order_fill(limit_price: float, next_bar_ohlcv, params: dict = None):
    """
    判定T+1日限价单是否成交，考虑滑点和边界情况

    核心逻辑：
      - 开盘低于限价 → 以开盘价+滑点成交（跳空低开场景）
      - 盘中最低价深入限价以下（含安全垫）→ 以限价成交
      - 最低价仅擦边（未穿透安全垫）→ 不成交（避免未来函数）
      - 成交量过低 → 不成交（流动性不足）

    参数:
      limit_price: 限价（T日收盘后计算的T+1日买入价）
      next_bar_ohlcv: T+1日K线数据，支持 dict 或 pd.Series
                      必须包含: open, high, low, close, volume
                      可选: volume_ma（20日均量，用于流动性检查）
      params: 执行参数字典，支持以下键:
              - slippage_pct: 滑点比例，默认0.001 (0.1%)
              - tick_buffer: 安全垫金额，默认0.01 (1分钱)
              - volume_depth_check: 是否启用成交量检查，默认True
              - min_volume_ratio: 最低成交量比例，默认0.5

    返回:
      tuple: (filled: bool, fill_price: float, fill_reason: str)
        - filled: 是否成交
        - fill_price: 实际成交价（未成交时为0.0）
        - fill_reason: 成交原因描述
          - "gap_down_fill": 开盘跳空低开成交
          - "intraday_fill": 盘中触及限价成交
          - "not_filled_no_penetration": 未穿透安全垫
          - "not_filled_low_volume": 成交量不足
          - "not_filled_gap_up": 全天高于限价

    示例:
      >>> bar = {"open": 10.05, "high": 10.20, "low": 9.88, "close": 10.10, "volume": 50000}
      >>> filled, price, reason = check_limit_order_fill(10.00, bar)
      >>> # low=9.88 < 10.00-0.01=9.99 → filled=True, price=10.00
    """
    if params is None:
        params = {}

    slippage = params.get("slippage_pct", DEFAULT_EXECUTION_PARAMS["slippage_pct"])
    tick_buffer = params.get("tick_buffer", DEFAULT_EXECUTION_PARAMS["tick_buffer"])
    volume_check = params.get("volume_depth_check", DEFAULT_EXECUTION_PARAMS["volume_depth_check"])
    min_vol_ratio = params.get("min_volume_ratio", DEFAULT_EXECUTION_PARAMS["min_volume_ratio"])

    # 提取K线数据（兼容 dict 和 pd.Series）
    if hasattr(next_bar_ohlcv, "get"):
        # dict-like
        open_p = float(next_bar_ohlcv.get("open") or next_bar_ohlcv.get("Open", 0))
        high_p = float(next_bar_ohlcv.get("high") or next_bar_ohlcv.get("High", 0))
        low_p = float(next_bar_ohlcv.get("low") or next_bar_ohlcv.get("Low", 0))
        close_p = float(next_bar_ohlcv.get("close") or next_bar_ohlcv.get("Close", 0))
        volume = float(next_bar_ohlcv.get("volume") or next_bar_ohlcv.get("Volume", 0))
        volume_ma = float(next_bar_ohlcv.get("volume_ma", 0))
    else:
        # pd.Series with attribute access
        open_p = float(getattr(next_bar_ohlcv, "Open", 0) or getattr(next_bar_ohlcv, "open", 0))
        high_p = float(getattr(next_bar_ohlcv, "High", 0) or getattr(next_bar_ohlcv, "high", 0))
        low_p = float(getattr(next_bar_ohlcv, "Low", 0) or getattr(next_bar_ohlcv, "low", 0))
        close_p = float(getattr(next_bar_ohlcv, "Close", 0) or getattr(next_bar_ohlcv, "close", 0))
        volume = float(getattr(next_bar_ohlcv, "Volume", 0) or getattr(next_bar_ohlcv, "volume", 0))
        volume_ma = float(getattr(next_bar_ohlcv, "volume_ma", 0))

    # 基本校验
    if limit_price <= 0 or open_p <= 0 or low_p <= 0:
        return (False, 0.0, "invalid_data")

    # ── 成交量深度检查（可选）──
    # 如果当日成交量 < 均量的 min_vol_ratio，认为流动性不足，限价单可能无法成交
    if volume_check and volume_ma > 0:
        vol_ratio = volume / volume_ma
        if vol_ratio < min_vol_ratio:
            logger.debug(f"限价单未成交(流动性不足): 量比={vol_ratio:.2f} < {min_vol_ratio}")
            return (False, 0.0, "not_filled_low_volume")

    # ── 情况1: 开盘跳空低开 ≤ 限价 ──
    # T+1开盘价直接低于/等于限价 → 集合竞价已满足条件
    # 实际成交价 = 开盘价 + 滑点（集合竞价可能有微小冲击）
    if open_p <= limit_price:
        fill_price = open_p * (1 + slippage)
        # 成交价不应超过限价（限价是上限）
        fill_price = min(fill_price, limit_price)
        # 成交价也不应低于跌停价（此处简化：不低于开盘价×0.9）
        fill_price = max(fill_price, open_p)
        fill_price = round(fill_price, 2)
        return (True, fill_price, "gap_down_fill")

    # ── 情况2: 盘中最低价深入限价以下（含安全垫）──
    # 要求: low < limit_price - tick_buffer
    # 含义：价格必须真正穿越限价位一定深度，而不只是"碰了一下"
    # 这避免了回测中的"虚假成交"（最低价恰好等于限价的擦边情况）
    effective_fill_threshold = limit_price - tick_buffer
    if low_p < effective_fill_threshold:
        # 以限价成交（限价挂单，只要价格到了就以限价成交）
        fill_price = round(limit_price, 2)
        return (True, fill_price, "intraday_fill")

    # ── 情况3: 最低价在限价附近但未穿透安全垫 ──
    # low_p >= limit_price - tick_buffer（即最低价距离限价不到1分钱）
    # 不判定成交，因为这种"擦边"情况在实盘中很可能不会真正成交
    if low_p <= limit_price:
        # 最低价到了限价但没穿透安全垫 → 保守不成交
        return (False, 0.0, "not_filled_no_penetration")

    # ── 情况4: 全天价格高于限价 ──
    # low_p > limit_price → 限价完全未触及
    return (False, 0.0, "not_filled_gap_up")


def calc_fill_price_with_slippage(base_price: float, slippage_pct: float = 0.001,
                                   direction: str = "buy") -> float:
    """
    计算含滑点的成交价

    参数:
      base_price: 基础价格
      slippage_pct: 滑点比例
      direction: "buy"（买入，价格上移）或 "sell"（卖出，价格下移）

    返回:
      含滑点的成交价
    """
    if direction == "buy":
        return round(base_price * (1 + slippage_pct), 2)
    else:
        return round(base_price * (1 - slippage_pct), 2)
