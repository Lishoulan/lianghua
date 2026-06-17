"""
OAMV择时状态模块

从 v63_daily_push.py / v64_daily_push.py 提取的公共OAMV择时逻辑。
支持两种数据源：
  1. 全市场活跃市值（优先，复刻指南针）
  2. 沪深300成交额代理（降级）
"""
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def get_oamv_status():
    """获取OAMV活跃市值当前状态

    优先使用全市场活跃市值缓存（circ_mv × turnover_rate 聚合），
    降级使用沪深300成交额代理。

    Returns:
        dict or None: {
            'latest_date': str,
            'daily_allowed': bool,
            'weekly_allowed': bool,
            'can_open_position': bool,
            'latest_x': float,
            'data_source': str,
            'recent_states': list,
            'last_transition': dict or None,
            'trend_label': str,
        }
    """
    from ml_strategy.oamv_filter import OAMVHysteresisFilter
    from ml_strategy.market_amv_cache import get_market_amv_series

    try:
        # 获取全市场活跃市值时间序列
        amv_series = get_market_amv_series()
        if amv_series is None or len(amv_series) < 40:
            logger.info("全市场活跃市值数据不足，回退到成交额代理")
            return _oamv_from_index()

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
            return _oamv_from_index()

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
        logger.warning(f"OAMV计算失败({e})，使用成交额代理")
        return _oamv_from_index()


def _oamv_from_index():
    """用沪深300成交额代理OAMV（降级方案）"""
    try:
        import tushare as ts
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            return None
        pro = ts.pro_api(token)

        end_date = pd.Timestamp.now().strftime("%Y%m%d")
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime("%Y%m%d")
        index_df = pro.index_daily(ts_code="000300.SH", start_date=start_date, end_date=end_date)
        if index_df is None or len(index_df) < 40:
            return None

        from ml_strategy.oamv_filter import OAMVHysteresisFilter
        index_df = index_df.sort_values("trade_date").reset_index(drop=True)
        index_df["Date"] = pd.to_datetime(index_df["trade_date"], format="%Y%m%d")
        index_df.set_index("Date", inplace=True)
        index_df["amount"] = index_df["amount"].astype(float)

        oamv = OAMVHysteresisFilter(
            upper_threshold=2.0, lower_threshold=-1.0,
            cost_ma_period=42, smooth_method='sma', smooth_period=15,
        )
        oamv.fit(index_df)
        state_df = oamv.get_state_df()
        if state_df is None or len(state_df) == 0:
            return None

        latest = state_df.iloc[-1]
        can_open = int(latest['oamv_state']) == 1

        # 最近切换
        last_transitions = []
        prev = None
        for idx, row in state_df.iterrows():
            s = int(row['oamv_state'])
            if prev is not None and s != prev:
                last_transitions.append({
                    'date': idx.strftime('%m-%d'),
                    'to': '牛市' if s == 1 else '熊市',
                })
            prev = s
        last_transitions = last_transitions[-3:]

        return {
            'can_open_position': can_open,
            'trend_label': '活跃度上升' if can_open else '活跃度下降',
            'last_transitions': last_transitions,
            'latest_x': round(float(latest.get('oamv_x', 0)), 2),
            'data_source': '成交额代理(amount)',
        }
    except Exception as e:
        logger.warning(f"成交额代理也失败({e})")
        return None
