import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class FeatureDiscretizer:

    @staticmethod
    def discretize_price_distance(dist_pct):
        if dist_pct < -1.5:
            return 0
        elif dist_pct < 2.0:
            return 1
        elif dist_pct < 6.0:
            return 2
        else:
            return 3

    @staticmethod
    def crystallize_j_value(j_value):
        if j_value <= 0:
            return 0
        elif j_value <= 20:
            return 1
        elif j_value <= 80:
            return 2
        elif j_value <= 100:
            return 3
        else:
            return 4

    @staticmethod
    def categorize_kline_pattern(row):
        body = abs(row['Close'] - row['Open'])
        total_range = row['High'] - row['Low']
        if total_range == 0:
            return 3
        upper_shadow = row['High'] - max(row['Open'], row['Close'])
        lower_shadow = min(row['Open'], row['Close']) - row['Low']
        body_ratio = body / total_range
        lower_shadow_ratio = lower_shadow / total_range
        if lower_shadow_ratio > 0.5 and body_ratio < 0.3:
            return 1
        if row['Close'] < row['Open'] and body_ratio > 0.7:
            return 2
        if body_ratio < 0.1:
            return 3
        if row['Close'] > row['Open'] and 0.3 <= body_ratio <= 0.6:
            return 4
        return 0

    def transform(self, df):
        df = df.copy()

        df['dist_pct'] = (df['Close'] - df['yellow_line']) / df['yellow_line'] * 100

        df['price_zone'] = df['dist_pct'].apply(self.discretize_price_distance)

        low_9 = df['Low'].rolling(window=9, min_periods=9).min()
        high_9 = df['High'].rolling(window=9, min_periods=9).max()
        denom = high_9 - low_9
        rsv = pd.Series(np.where(denom == 0, 50, (df['Close'] - low_9) / denom * 100),
                         index=df.index, dtype=float)
        rsv = rsv.fillna(50)

        k = pd.Series(np.nan, index=df.index, dtype=float)
        d = pd.Series(np.nan, index=df.index, dtype=float)
        k.iloc[0] = 50.0
        d.iloc[0] = 50.0
        for i in range(1, len(df)):
            k.iloc[i] = 2.0 / 3.0 * k.iloc[i - 1] + 1.0 / 3.0 * rsv.iloc[i]
            d.iloc[i] = 2.0 / 3.0 * d.iloc[i - 1] + 1.0 / 3.0 * k.iloc[i]
        j = 3.0 * k - 2.0 * d

        df['j_zone'] = j.apply(self.crystallize_j_value)

        df['j_raw'] = j

        df['k_raw'] = k
        df['d_raw'] = d

        df['j_min_5'] = j.rolling(window=5, min_periods=1).min()
        df['j_max_5'] = j.rolling(window=5, min_periods=1).max()
        df['j_rising'] = (j > j.shift(1)).astype(int)

        df['k_pattern'] = df.apply(self.categorize_kline_pattern, axis=1)

        slope_dk = df['yellow_line'].rolling(window=5, min_periods=5).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
        )
        slope_dk_norm = slope_dk / df['yellow_line'] * 100
        sign_slope = np.where(slope_dk > 0, 1, -1)
        price_dist_ratio = (df['Close'] - df['yellow_line']) / df['yellow_line']
        df['temd'] = sign_slope * price_dist_ratio * (100.0 - j)

        df['yellow_slope_sign'] = np.where(slope_dk > 0, 1, -1).astype(int)
        df['yellow_slope_norm'] = slope_dk_norm

        if 'Vol_MA20' not in df.columns:
            df['Vol_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['vol_ratio_20'] = df['Volume'] / df['Vol_MA20']

        df['vol_change'] = df['Volume'].pct_change()
        df['vol_ma5_ratio'] = df['Volume'] / df['Volume'].rolling(window=5, min_periods=1).mean()

        prev_close = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - prev_close).abs()
        tr3 = (df['Low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(window=14, min_periods=1).mean()
        df['atr_ratio'] = atr14 / df['Close']

        df['white_above_yellow'] = (df['white_line'] > df['yellow_line']).astype(int)

        df['white_yellow_dist'] = (df['white_line'] - df['yellow_line']) / df['yellow_line'] * 100

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        df['macd_value'] = 2.0 * (dif - dea)
        df['dif_value'] = dif
        df['macd_above_zero'] = (df['macd_value'] > 0).astype(int)

        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        df['rsi_value'] = 100.0 - 100.0 / (1.0 + rs)

        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)

        df['high_20'] = df['High'].rolling(window=20, min_periods=1).max()
        df['low_20'] = df['Low'].rolling(window=20, min_periods=1).min()
        df['price_position_20'] = (df['Close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)

        df['low_above_yellow'] = (df['Low'] > df['yellow_line']).astype(int)

        df['close_position_20'] = df['price_position_20']
        
        # 改进的PWVC计算：高位放量阴线才否决，且保持否决3天
        df['is_red_candle'] = (df['Close'] < df['Open']).astype(int)
        high_5 = df['High'].rolling(window=5, min_periods=1).max()
        low_5 = df['Low'].rolling(window=5, min_periods=1).min()
        df['close_position_5'] = np.where(high_5 > low_5, 
                                          (df['Close'] - low_5) / (high_5 - low_5), 
                                          0.5)
        
        # 关键修改：看最高价或开盘价的位置，而不是收盘价！
        df['high_position_5'] = np.where(high_5 > low_5, 
                                         (df['High'] - low_5) / (high_5 - low_5), 
                                         0.5)
        df['open_position_5'] = np.where(high_5 > low_5, 
                                         (df['Open'] - low_5) / (high_5 - low_5), 
                                         0.5)
        df['top_position_5'] = np.maximum(df['high_position_5'], df['open_position_5'])
        
        # 单日PWVC：只有当：放量(vol_ratio_20>1.5) AND 阴线 AND 近期高位(top_position_5>0.6)时，才触发高值
        df['pwvc_day'] = df['vol_ratio_20'] * (df['top_position_5'] - 0.5) * df['is_red_candle']
        
        # 过去3天内如果有任何一天pwvc_day>0.8，就保持否决
        df['pwvc'] = df['pwvc_day'].rolling(window=3, min_periods=1).max()
        
        df['pwvc_distribution'] = 0
        df.loc[df['pwvc'] > 1.5, 'pwvc_distribution'] = 2
        df.loc[(df['pwvc'] > 0.5) & (df['pwvc'] <= 1.5), 'pwvc_distribution'] = 1

        df['j_clean'] = df['j_raw'].clip(-20, 120)

        vol_explosion = (df['vol_ratio_20'] > 2.0).astype(int)
        vol_explosion_count_10 = vol_explosion.rolling(window=10, min_periods=1).sum()

        white_above = df['white_above_yellow']
        golden_cross = (white_above == 1) & (white_above.shift(1) == 0)
        golden_cross_recent_5 = golden_cross.rolling(window=5, min_periods=1).sum()

        accumulation_score = pd.Series(0.0, index=df.index, dtype=float)
        accumulation_score += vol_explosion_count_10.clip(upper=3) / 3.0 * 0.5
        accumulation_score += golden_cross_recent_5.clip(upper=1) * 0.5
        low_position_mask = df['close_position_20'] < 0.5
        accumulation_score = accumulation_score.where(low_position_mask, accumulation_score * 0.3)
        df['accumulation_score'] = accumulation_score

        return df

    def add_path_signatures(self, df, sig_builder=None):
        if sig_builder is None:
            from ml_strategy.path_signature import PathSignatureBuilder
            sig_builder = PathSignatureBuilder(truncation_level=2, path_dims=3, path_length=5, lead_lag=False)
        df = sig_builder.compute_signature_features(df)
        return df

    def add_market_context(self, stock_df, index_df=None, hmm_result=None,
                           industry_j_series=None):
        df = stock_df.copy()
        if index_df is not None and 'yellow_line' in index_df.columns:
            idx_yl = index_df['yellow_line'].reindex(df.index)
            idx_slope = idx_yl.rolling(window=5, min_periods=5).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
            )
            idx_slope_norm = idx_slope / (idx_yl + 1e-8) * 100
            df['index_yellow_slope'] = idx_slope_norm.fillna(0)
        else:
            df['index_yellow_slope'] = 0.0

        if hmm_result is not None:
            hmm_reindexed = hmm_result['hmm_state2_prob'].reindex(df.index).fillna(0)
            df['index_state2_prob'] = hmm_reindexed
        else:
            df['index_state2_prob'] = 0.0

        if industry_j_series is not None:
            ind_j = industry_j_series.reindex(df.index).fillna(50)
            df['industry_oversold'] = np.where(ind_j < 20, (20 - ind_j) / 20, 0.0)
        else:
            df['industry_oversold'] = 0.0

        return df
