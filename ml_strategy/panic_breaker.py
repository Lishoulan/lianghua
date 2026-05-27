import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class MarketPanicCircuitBreaker:

    def __init__(self, breadth_threshold=0.85, limit_down_threshold=150,
                 ma_period=20, cooldown_days=5,
                 limit_down_accel_factor=3.0, breadth_deterioration_pct=0.20,
                 breadth_deterioration_window=5):
        self.breadth_threshold = breadth_threshold
        self.limit_down_threshold = limit_down_threshold
        self.ma_period = ma_period
        self.cooldown_days = cooldown_days
        self.limit_down_accel_factor = limit_down_accel_factor
        self.breadth_deterioration_pct = breadth_deterioration_pct
        self.breadth_deterioration_window = breadth_deterioration_window
        self.breadth_series = None
        self.limit_down_series = None
        self.panic_state = None
        self.market_state = None

    def compute_market_breadth(self, all_stock_data):
        print("Computing market breadth (panic circuit breaker)...")
        all_dates = set()
        for ts_code, info in all_stock_data.items():
            all_dates.update(info['data'].index)
        all_dates = sorted(all_dates)

        breadth = pd.Series(1.0, index=all_dates, dtype=float)
        limit_downs = pd.Series(0, index=all_dates, dtype=int)

        for date in all_dates:
            below_ma_count = 0
            limit_down_count = 0
            total_count = 0

            for ts_code, info in all_stock_data.items():
                df = info['data']
                if date not in df.index:
                    continue

                total_count += 1
                row = df.loc[date]

                close = row.get('Close', np.nan)
                if pd.isna(close):
                    continue

                ma_col = f'ma{self.ma_period}'
                if ma_col in df.columns:
                    ma_val = row.get(ma_col, np.nan)
                    if pd.notna(ma_val) and close < ma_val:
                        below_ma_count += 1

                pre_close = row.get('pre_close', np.nan)
                if pd.notna(pre_close) and pre_close > 0:
                    pct_change = (close - pre_close) / pre_close * 100
                    if pct_change <= -9.5:
                        limit_down_count += 1

            if total_count > 0:
                breadth.loc[date] = below_ma_count / total_count
            limit_downs.loc[date] = limit_down_count

        self.breadth_series = breadth
        self.limit_down_series = limit_downs

        panic = pd.Series(False, index=all_dates, dtype=bool)
        for i in range(len(all_dates)):
            if breadth.iloc[i] >= self.breadth_threshold:
                panic.iloc[i] = True
            elif limit_downs.iloc[i] >= self.limit_down_threshold:
                panic.iloc[i] = True

        self.panic_state = panic

        market_state = pd.Series('normal', index=all_dates, dtype=str)
        for i in range(len(all_dates)):
            if panic.iloc[i]:
                market_state.iloc[i] = 'panic'
            else:
                is_warning = False

                if i > 0:
                    prev_ld = limit_downs.iloc[i - 1]
                    curr_ld = limit_downs.iloc[i]
                    if prev_ld > 0 and curr_ld >= prev_ld * self.limit_down_accel_factor:
                        is_warning = True

                if not is_warning and i >= self.breadth_deterioration_window:
                    past_breadth = breadth.iloc[i - self.breadth_deterioration_window]
                    curr_breadth = breadth.iloc[i]
                    if past_breadth < 0.5 and (curr_breadth - past_breadth) >= self.breadth_deterioration_pct:
                        is_warning = True

                if is_warning:
                    market_state.iloc[i] = 'warning'

        self.market_state = market_state

        panic_days = panic.sum()
        warning_days = (market_state == 'warning').sum()
        total_days = len(panic)
        print(f"  Panic days: {panic_days}/{total_days} ({panic_days/total_days*100:.1f}%)")
        print(f"  Warning days: {warning_days}/{total_days} ({warning_days/total_days*100:.1f}%)")
        print(f"  Breadth range: [{breadth.min():.2%}, {breadth.max():.2%}]")
        print(f"  Max limit-downs in a day: {limit_downs.max()}")

        return panic

    def is_panic(self, date):
        if self.panic_state is None:
            return False
        date_ts = pd.Timestamp(date)
        if date_ts in self.panic_state.index:
            return bool(self.panic_state.loc[date_ts])
        return False

    def get_market_state(self, date):
        if self.market_state is None:
            return 'normal'
        date_ts = pd.Timestamp(date)
        if date_ts in self.market_state.index:
            return str(self.market_state.loc[date_ts])
        return 'normal'

    def get_panic_dates(self):
        if self.panic_state is None:
            return []
        return [d for d in self.panic_state.index[self.panic_state] if self.panic_state.loc[d]]

    def summary(self):
        if self.panic_state is None:
            return "Not computed yet"
        panic_days = self.panic_state.sum()
        warning_days = (self.market_state == 'warning').sum() if self.market_state is not None else 0
        total_days = len(self.panic_state)
        lines = [
            f"Market Panic Circuit Breaker:",
            f"  Breadth threshold: {self.breadth_threshold:.0%} below {self.ma_period}-day MA",
            f"  Limit-down threshold: {self.limit_down_threshold} stocks",
            f"  Limit-down acceleration factor: {self.limit_down_accel_factor}x",
            f"  Breadth deterioration: {self.breadth_deterioration_pct:.0%} in {self.breadth_deterioration_window} days",
            f"  Panic days: {panic_days}/{total_days} ({panic_days/total_days*100:.1f}%)",
            f"  Warning days: {warning_days}/{total_days} ({warning_days/total_days*100:.1f}%)",
        ]
        return "\n".join(lines)
