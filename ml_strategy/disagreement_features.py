import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "pip_libs"))


class DisagreementFeatureBuilder:

    def __init__(self, ssa_window=10, ssa_signal_groups=2):
        self.ssa_window = ssa_window
        self.ssa_signal_groups = ssa_signal_groups

    def _ssa_denoise_series(self, series):
        series = np.asarray(series, dtype=float)
        mask = ~np.isnan(series)
        if mask.sum() < self.ssa_window * 2:
            return series.copy()
        clean = series.copy()
        valid_idx = np.where(mask)[0]
        valid_vals = series[valid_idx]
        L = self.ssa_window
        N = len(valid_vals)
        K = N - L + 1
        if K <= 0:
            return clean
        trajectory = np.zeros((L, K))
        for i in range(K):
            trajectory[:, i] = valid_vals[i:i + L]
        try:
            U, sigma, Vt = np.linalg.svd(trajectory, full_matrices=False)
        except np.linalg.LinAlgError:
            return clean
        n_keep = min(self.ssa_signal_groups, len(sigma))
        X_group = np.zeros((L, K))
        for idx in range(n_keep):
            X_group += sigma[idx] * np.outer(U[:, idx], Vt[idx, :])
        reconstructed = np.zeros(N)
        counts = np.zeros(N)
        for i in range(L):
            for j in range(K):
                reconstructed[i + j] += X_group[i, j]
                counts[i + j] += 1
        valid_counts = counts > 0
        reconstructed[valid_counts] /= counts[valid_counts]
        result = valid_vals.copy()
        min_len = min(len(reconstructed), len(result))
        result[:min_len] = reconstructed[:min_len]
        clean[valid_idx] = result
        return clean

    def compute_temd(self, df, j_col='j_raw', yellow_col='yellow_line'):
        df = df.copy()
        if j_col not in df.columns or yellow_col not in df.columns:
            df['temd'] = 0.0
            return df

        j_series = df[j_col].values.astype(float)
        j_clean = self._ssa_denoise_series(j_series)
        df['j_clean'] = j_clean

        slope_dk = df[yellow_col].rolling(window=5, min_periods=5).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
        )
        sign_slope = np.where(slope_dk > 0, 1, -1)
        price_dist_ratio = (df['Close'] - df[yellow_col]) / df[yellow_col]
        df['temd'] = sign_slope * price_dist_ratio * (100.0 - j_clean)
        return df

    def compute_pwvc(self, df, vol_ratio_col='vol_ratio_20', window=20):
        df = df.copy()
        if 'Close' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns:
            df['pwvc'] = 0.0
            return df

        high_w = df['High'].rolling(window=window, min_periods=1).max()
        low_w = df['Low'].rolling(window=window, min_periods=1).min()
        range_w = high_w - low_w
        close_position = np.where(
            range_w > 1e-8,
            (df['Close'] - low_w) / range_w,
            0.5
        )
        df['close_position_20'] = close_position

        if vol_ratio_col not in df.columns:
            if 'Vol_MA20' not in df.columns:
                df['Vol_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            df['vol_ratio_20'] = df['Volume'] / df['Vol_MA20']
            vol_ratio_col = 'vol_ratio_20'

        vol_ratio_ssa = self._ssa_denoise_series(df[vol_ratio_col].values.astype(float))
        df['vol_ratio_ssa'] = vol_ratio_ssa

        df['pwvc'] = vol_ratio_ssa * (close_position - 0.5)
        return df

    def build_features(self, df, j_col='j_raw', yellow_col='yellow_line',
                       vol_ratio_col='vol_ratio_20', window=20):
        df = self.compute_temd(df, j_col, yellow_col)
        df = self.compute_pwvc(df, vol_ratio_col, window)

        df['pwvc_distribution'] = 0
        df.loc[df['pwvc'] > 1.5, 'pwvc_distribution'] = -1
        df.loc[df['pwvc'] < -1.5, 'pwvc_distribution'] = 1

        return df

    def summary(self):
        return (
            f"DisagreementFeatureBuilder:\n"
            f"  SSA window: {self.ssa_window}\n"
            f"  SSA signal groups: {self.ssa_signal_groups}\n"
            f"  Features: TEMD, PWVC, close_position_20, j_clean, pwvc_distribution"
        )
