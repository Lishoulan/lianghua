import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "pip_libs"))


class TripleBarrierLabeler:
    def __init__(self, atr_mult_upper=1.5, atr_mult_lower=0.8, max_hold_days=5):
        self.atr_mult_upper = atr_mult_upper
        self.atr_mult_lower = atr_mult_lower
        self.max_hold_days = max_hold_days

    def label_single(self, df, entry_idx, atr14_at_entry):
        if entry_idx + 1 >= len(df):
            return -1

        entry_price = df.iloc[entry_idx + 1]['Open']

        if pd.isna(entry_price) or entry_price == 0:
            return -1

        upper_barrier = entry_price + self.atr_mult_upper * atr14_at_entry
        lower_barrier = entry_price - self.atr_mult_lower * atr14_at_entry

        end_idx = min(entry_idx + self.max_hold_days, len(df) - 1)

        for offset in range(entry_idx + 1, end_idx + 1):
            day = df.iloc[offset]
            day_high = day['High']
            day_low = day['Low']
            day_open = day['Open']
            day_close = day['Close']

            hit_upper = day_high >= upper_barrier
            hit_lower = day_low <= lower_barrier

            if hit_upper and hit_lower:
                if day_open >= upper_barrier:
                    return 1
                if day_open <= lower_barrier:
                    return 0
                if abs(day_open - upper_barrier) < abs(day_open - lower_barrier):
                    return 1
                else:
                    return 0
            elif hit_upper:
                return 1
            elif hit_lower:
                return 0

        last_close = df.iloc[end_idx]['Close']
        if last_close > entry_price:
            return 1
        else:
            return 0

    def label_all(self, df, candidate_indices, atr14_series):
        results = []
        for i in candidate_indices:
            if i + 1 >= len(df):
                continue

            atr_val = atr14_series.iloc[i]
            if pd.isna(atr_val):
                continue

            entry_price = df.iloc[i + 1]['Open']
            if pd.isna(entry_price) or entry_price == 0:
                continue

            upper_barrier = entry_price + self.atr_mult_upper * atr_val
            lower_barrier = entry_price - self.atr_mult_lower * atr_val

            label = self.label_single(df, i, atr_val)
            if label == -1:
                continue

            end_idx = min(i + self.max_hold_days, len(df) - 1)
            hold_days = end_idx - i

            results.append({
                'entry_idx': i,
                'label': label,
                'entry_price': entry_price,
                'upper_barrier': upper_barrier,
                'lower_barrier': lower_barrier,
                'hold_days': hold_days,
            })

        return results
