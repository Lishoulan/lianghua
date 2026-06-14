import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class OAMVHysteresisFilter:

    def __init__(self, upper_threshold=2.0, lower_threshold=-1.0,
                 cost_ma_period=42, roc_period=1,
                 weekly_ema_period=5, weekly_use_ema=True,
                 smooth_method='sma', smooth_period=15,
                 cost_ma_method='sma'):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.cost_ma_period = cost_ma_period
        self.roc_period = roc_period
        self.weekly_ema_period = weekly_ema_period
        self.weekly_use_ema = weekly_use_ema
        # 平滑方式: 'hybrid'=0.6*MA5+0.4*MA20, 'ema'=EMA(smooth_period), 'sma'=SMA(smooth_period), 'none'=不做平滑
        self.smooth_method = smooth_method
        self.smooth_period = smooth_period
        # cost_ma方式: 'sma'=简单移动平均, 'ema'=指数移动平均
        self.cost_ma_method = cost_ma_method
        self.state_series = None
        self.x_series = None
        self.oamv_smooth = None
        self.cost_ma = None
        self.weekly_state_series = None
        self.weekly_x_series = None
        self.weekly_cost_ma = None
        self.data_source = 'proxy'

    def _apply_smoothing(self, raw_series):
        """对原始序列应用平滑"""
        if self.smooth_method == 'hybrid':
            ma5 = raw_series.rolling(window=5, min_periods=5).mean()
            ma20 = raw_series.rolling(window=20, min_periods=20).mean()
            return 0.6 * ma5 + 0.4 * ma20
        elif self.smooth_method == 'ema':
            return raw_series.ewm(span=self.smooth_period, adjust=False).mean()
        elif self.smooth_method == 'sma':
            return raw_series.rolling(window=self.smooth_period, min_periods=self.smooth_period).mean()
        elif self.smooth_method == 'none':
            return raw_series.copy()
        else:
            raise ValueError(f"Unknown smooth_method: {self.smooth_method}")

    def _apply_cost_ma(self, smooth_series):
        """对平滑后序列计算成本均线"""
        if self.cost_ma_method == 'ema':
            return smooth_series.ewm(span=self.cost_ma_period, adjust=False).mean()
        else:
            return smooth_series.rolling(window=self.cost_ma_period, min_periods=self.cost_ma_period).mean()

    def compute_oamv_proxy(self, index_df):
        df = index_df.copy()
        if 'amount' in df.columns:
            oamv = df['amount'].astype(float)
        elif 'Volume' in df.columns and 'Close' in df.columns:
            oamv = df['Volume'] * df['Close']
        else:
            raise ValueError("Need 'amount' or 'Volume'+'Close' columns")

        oamv_smooth = self._apply_smoothing(oamv)
        cost_ma = self._apply_cost_ma(oamv_smooth)
        x_t = (oamv_smooth - cost_ma) / cost_ma * 100.0

        return x_t, oamv_smooth, cost_ma

    def compute_oamv_live_chips(self, daily_basic_df):
        df = daily_basic_df.copy()

        if 'total_mv' in df.columns and 'turnover_rate_f' in df.columns:
            free_mv = df['total_mv']
            turnover_f = df['turnover_rate_f']
            live_chips = free_mv * turnover_f / 100.0
            self.data_source = 'live_chips'
        elif 'circ_mv' in df.columns and 'turnover_rate_f' in df.columns:
            circ_mv = df['circ_mv']
            turnover_f = df['turnover_rate_f']
            live_chips = circ_mv * turnover_f / 100.0
            self.data_source = 'live_chips_circ'
        elif 'amount' in df.columns:
            live_chips = df['amount'].astype(float)
            self.data_source = 'amount_fallback'
        else:
            raise ValueError("Need total_mv+turnover_rate_f or circ_mv+turnover_rate_f or amount")

        oamv_smooth = self._apply_smoothing(live_chips)
        cost_ma = self._apply_cost_ma(oamv_smooth)
        x_t = (oamv_smooth - cost_ma) / cost_ma * 100.0

        return x_t, oamv_smooth, cost_ma

    def compute_oamv_universe(self, all_stock_data, daily_basic_cache=None):
        print("Computing Universe AMV from stock pool...")
        ts_codes = list(all_stock_data.keys())

        if len(ts_codes) == 0:
            return None, None, None

        all_dates = set()
        for ts_code, info in all_stock_data.items():
            all_dates.update(info['data'].index)
        all_dates = sorted(all_dates)

        universe_amv = pd.Series(0.0, index=all_dates, dtype=float)

        if daily_basic_cache is not None and len(daily_basic_cache) > 0:
            self.data_source = 'universe_live_chips'
            for ts_code in ts_codes:
                if ts_code not in daily_basic_cache:
                    continue
                db = daily_basic_cache[ts_code]
                if db is None or db.empty:
                    continue
                info = all_stock_data[ts_code]
                df = info['data']

                common_dates = df.index.intersection(db.index)
                if len(common_dates) == 0:
                    continue

                for date in common_dates:
                    if 'total_mv' in db.columns and 'turnover_rate_f' in db.columns:
                        mv = db.loc[date, 'total_mv']
                        tr = db.loc[date, 'turnover_rate_f']
                        if pd.notna(mv) and pd.notna(tr):
                            universe_amv.loc[date] += mv * tr / 100.0
                    elif 'amount' in db.columns:
                        amt = db.loc[date, 'amount']
                        if pd.notna(amt):
                            universe_amv.loc[date] += float(amt)
        else:
            self.data_source = 'universe_amount'
            for ts_code in ts_codes:
                info = all_stock_data[ts_code]
                df = info['data']
                if 'amount' in df.columns:
                    for date in df.index:
                        val = df.loc[date, 'amount']
                        if pd.notna(val):
                            universe_amv.loc[date] += float(val)
                elif 'Volume' in df.columns and 'Close' in df.columns:
                    for date in df.index:
                        vol = df.loc[date, 'Volume']
                        close = df.loc[date, 'Close']
                        if pd.notna(vol) and pd.notna(close):
                            universe_amv.loc[date] += float(vol * close)

        oamv_smooth = self._apply_smoothing(universe_amv)
        cost_ma = self._apply_cost_ma(oamv_smooth)
        x_t = (oamv_smooth - cost_ma) / cost_ma * 100.0

        return x_t, oamv_smooth, cost_ma

    def compute_oamv_from_series(self, amv_series):
        """从预计算的活跃市值时间序列直接计算OAMV

        参数:
            amv_series: pd.Series, 索引为日期, 值为全市场活跃市值(circ_mv*turnover_rate/100)
        """
        oamv_smooth = self._apply_smoothing(amv_series)
        cost_ma = self._apply_cost_ma(oamv_smooth)
        x_t = (oamv_smooth - cost_ma) / cost_ma * 100.0

        return x_t, oamv_smooth, cost_ma

    def compute_weekly_oamv(self, daily_state_df):
        daily_x = daily_state_df['oamv_x'].copy()
        daily_x.index = pd.to_datetime(daily_x.index)

        weekly_x = daily_x.resample('W-FRI').last().dropna()

        if self.weekly_use_ema:
            weekly_cost_ma = weekly_x.ewm(span=self.weekly_ema_period, adjust=False).mean()
        else:
            weekly_cost_ma = weekly_x.rolling(window=self.weekly_ema_period, min_periods=self.weekly_ema_period).mean()

        self.weekly_cost_ma = weekly_cost_ma

        weekly_state = pd.Series(0, index=weekly_x.index, dtype=int)
        current_state = 0

        for i in range(len(weekly_x)):
            val = weekly_x.iloc[i]
            if pd.isna(val):
                weekly_state.iloc[i] = current_state
                continue

            if val >= self.upper_threshold:
                current_state = 1
            elif val <= self.lower_threshold:
                current_state = 0
            else:
                if not pd.isna(weekly_cost_ma.iloc[i]):
                    if val > weekly_cost_ma.iloc[i] and current_state == 0:
                        if i > 0 and weekly_x.iloc[i] > weekly_x.iloc[i-1]:
                            current_state = 1

            weekly_state.iloc[i] = current_state

        self.weekly_x_series = weekly_x
        self.weekly_state_series = weekly_state

        return weekly_state

    def get_weekly_state_for_date(self, date):
        if self.weekly_state_series is None:
            return 1

        date_ts = pd.Timestamp(date)
        valid = self.weekly_state_series[self.weekly_state_series.index <= date_ts]
        if len(valid) == 0:
            return 1

        return int(valid.iloc[-1])

    def apply_hysteresis(self, x_t):
        state = pd.Series(0, index=x_t.index, dtype=int)
        current_state = 0

        for i in range(len(x_t)):
            val = x_t.iloc[i]
            if pd.isna(val):
                state.iloc[i] = current_state
                continue

            if val >= self.upper_threshold:
                current_state = 1
            elif val <= self.lower_threshold:
                current_state = 0

            state.iloc[i] = current_state

        return state

    def fit(self, index_df=None, daily_basic_df=None, all_stock_data=None, daily_basic_cache=None,
            amv_series=None):
        if amv_series is not None:
            self.x_series, self.oamv_smooth, self.cost_ma = self.compute_oamv_from_series(amv_series)
            self.data_source = 'universe_cached'
        elif all_stock_data is not None:
            self.x_series, self.oamv_smooth, self.cost_ma = self.compute_oamv_universe(
                all_stock_data, daily_basic_cache)
        elif daily_basic_df is not None:
            self.x_series, self.oamv_smooth, self.cost_ma = self.compute_oamv_live_chips(daily_basic_df)
        else:
            self.x_series, self.oamv_smooth, self.cost_ma = self.compute_oamv_proxy(index_df)

        self.state_series = self.apply_hysteresis(self.x_series)

        state_df = self.get_state_df()
        self.compute_weekly_oamv(state_df)

        return self

    def get_state_df(self):
        if self.state_series is None:
            return None
        result = pd.DataFrame({
            'oamv_state': self.state_series,
            'oamv_x': self.x_series,
            'oamv_smooth': self.oamv_smooth,
            'oamv_cost_ma': self.cost_ma,
        })
        return result

    def get_state_dict(self):
        if self.state_series is None:
            return {}
        return {date: int(state) for date, state in self.state_series.items()}

    def is_trading_allowed(self, date, require_weekly=True):
        daily_ok = False
        if self.state_series is not None and date in self.state_series.index:
            daily_ok = self.state_series.loc[date] == 1

        if not daily_ok:
            return False

        if not require_weekly:
            return True

        if self.weekly_state_series is None:
            return True

        weekly_state = self.get_weekly_state_for_date(date)
        return weekly_state == 1

    def get_transition_dates(self):
        if self.state_series is None:
            return []
        transitions = []
        prev_state = 0
        for i in range(len(self.state_series)):
            curr_state = self.state_series.iloc[i]
            if curr_state != prev_state:
                transitions.append({
                    'date': self.state_series.index[i],
                    'from': prev_state,
                    'to': curr_state,
                    'x_value': self.x_series.iloc[i],
                })
                prev_state = curr_state
        return transitions

    def summary(self):
        if self.state_series is None:
            return "Not fitted yet"

        total = len(self.state_series)
        bullish = (self.state_series == 1).sum()
        bearish = (self.state_series == 0).sum()
        transitions = self.get_transition_dates()

        x_valid = self.x_series.dropna()
        buffer_zone = ((x_valid > self.lower_threshold) & (x_valid < self.upper_threshold)).sum()

        lines = [
            f"0AMV Hysteresis Filter Summary:",
            f"  Data source: {self.data_source}",
            f"  Upper threshold: +{self.upper_threshold}%",
            f"  Lower threshold: {self.lower_threshold}%",
            f"  Cost MA period: {self.cost_ma_period}",
            f"  Total days: {total}",
            f"  Bullish (State=1): {bullish} ({bullish/total*100:.1f}%)",
            f"  Bearish (State=0): {bearish} ({bearish/total*100:.1f}%)",
            f"  Buffer zone days: {buffer_zone} ({buffer_zone/len(x_valid)*100:.1f}%)",
            f"  State transitions: {len(transitions)}",
            f"  X range: [{x_valid.min():.2f}%, {x_valid.max():.2f}%]",
        ]

        if self.weekly_state_series is not None:
            w_total = len(self.weekly_state_series)
            w_bullish = (self.weekly_state_series == 1).sum()
            ma_type = "EMA" if self.weekly_use_ema else "SMA"
            lines.extend([
                f"  Weekly filter: ON ({ma_type}-{self.weekly_ema_period})",
                f"  Weekly bullish: {w_bullish}/{w_total} ({w_bullish/w_total*100:.1f}%)",
            ])
        else:
            lines.append(f"  Weekly filter: OFF")

        return "\n".join(lines)
