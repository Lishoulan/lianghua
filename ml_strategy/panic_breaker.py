import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "pip_libs"))


class MarketPanicCircuitBreaker:

    def __init__(self, breadth_threshold=0.85, limit_down_threshold=150,
                 ma_period=20, cooldown_days=5):
        self.breadth_threshold = breadth_threshold
        self.limit_down_threshold = limit_down_threshold
        self.ma_period = ma_period
        self.cooldown_days = cooldown_days
        self.breadth_series = None
        self.limit_down_series = None
        self.panic_state = None

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

        panic_days = panic.sum()
        total_days = len(panic)
        print(f"  Panic days: {panic_days}/{total_days} ({panic_days/total_days*100:.1f}%)")
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

    def get_panic_dates(self):
        if self.panic_state is None:
            return []
        return [d for d in self.panic_state.index[self.panic_state] if self.panic_state.loc[d]]

    def summary(self):
        if self.panic_state is None:
            return "Not computed yet"
        panic_days = self.panic_state.sum()
        total_days = len(self.panic_state)
        lines = [
            f"Market Panic Circuit Breaker:",
            f"  Breadth threshold: {self.breadth_threshold:.0%} below {self.ma_period}-day MA",
            f"  Limit-down threshold: {self.limit_down_threshold} stocks",
            f"  Panic days: {panic_days}/{total_days} ({panic_days/total_days*100:.1f}%)",
        ]
        return "\n".join(lines)
