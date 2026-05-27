import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class SterileDataCleaner:

    LIMIT_RATIO_MAIN = 0.10
    LIMIT_RATIO_KC = 0.20

    def __init__(self):
        self._limit_ratio_map = {}
        self._mask_cache = {}

    @staticmethod
    def _get_limit_ratio(ts_code):
        if ts_code.startswith('688') or ts_code.startswith('300'):
            return 0.20
        return 0.10

    @staticmethod
    def _round_price(price, direction='up'):
        if direction == 'up':
            return round(price + 0.005, 2)
        else:
            return round(price - 0.005, 2)

    def compute_limit_prices(self, prev_close, ts_code):
        ratio = self._get_limit_ratio(ts_code)
        p_up = round(prev_close * (1 + ratio) + 0.005, 2)
        p_down = round(prev_close * (1 - ratio) - 0.005, 2)
        return p_up, p_down

    def compute_tradable_mask(self, df, ts_code=''):
        df = df.copy()
        prev_close = df['Close'].shift(1)
        ratio = self._get_limit_ratio(ts_code)
        p_up = (prev_close * (1 + ratio)).round(2)
        p_down = (prev_close * (1 - ratio)).round(2)

        is_limit_up = df['Close'] >= p_up
        is_limit_down = df['Close'] <= p_down
        is_tradable = ~(is_limit_up | is_limit_down)
        is_tradable.iloc[0] = True

        df['is_tradable'] = is_tradable
        df['is_limit_up'] = is_limit_up
        df['is_limit_down'] = is_limit_down
        return df

    def sterilize(self, df, ts_code=''):
        df = self.compute_tradable_mask(df, ts_code)

        sterile_close = df['Close'].copy()
        sterile_high = df['High'].copy()
        sterile_low = df['Low'].copy()

        last_valid_close = np.nan
        for i in range(len(df)):
            if df['is_tradable'].iloc[i]:
                last_valid_close = df['Close'].iloc[i]
            else:
                if not np.isnan(last_valid_close):
                    sterile_close.iloc[i] = last_valid_close
                    sterile_high.iloc[i] = last_valid_close
                    sterile_low.iloc[i] = last_valid_close

        sterile_df = df.copy()
        sterile_df['Close_sterile'] = sterile_close
        sterile_df['High_sterile'] = sterile_high
        sterile_df['Low_sterile'] = sterile_low
        return sterile_df

    def get_feature_dataframe(self, df, ts_code=''):
        sterile_df = self.sterilize(df, ts_code)
        feature_df = sterile_df.copy()
        feature_df['Close'] = feature_df['Close_sterile']
        feature_df['High'] = feature_df['High_sterile']
        feature_df['Low'] = feature_df['Low_sterile']
        return feature_df

    def restore_real_prices(self, feature_df, original_df):
        result = feature_df.copy()
        result['Close'] = original_df['Close']
        result['High'] = original_df['High']
        result['Low'] = original_df['Low']
        return result

    def summary(self, df, ts_code=''):
        mask_df = self.compute_tradable_mask(df, ts_code)
        total = len(mask_df)
        tradable = mask_df['is_tradable'].sum()
        limit_up = mask_df['is_limit_up'].sum()
        limit_down = mask_df['is_limit_down'].sum()
        ratio = self._get_limit_ratio(ts_code)
        return (
            f"SterileDataCleaner:\n"
            f"  Code: {ts_code}, Limit ratio: {ratio:.0%}\n"
            f"  Total days: {total}\n"
            f"  Tradable: {tradable} ({tradable/total*100:.1f}%)\n"
            f"  Limit-up: {limit_up} ({limit_up/total*100:.1f}%)\n"
            f"  Limit-down: {limit_down} ({limit_down/total*100:.1f}%)"
        )
