import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "pip_libs"))


class SSADenoiser:

    def __init__(self, window_length=10, n_signal_groups=2):
        self.window_length = window_length
        self.n_signal_groups = n_signal_groups

    def _embed(self, series):
        N = len(series)
        L = self.window_length
        K = N - L + 1
        if K <= 0:
            return None
        trajectory = np.zeros((L, K))
        for i in range(K):
            trajectory[:, i] = series[i:i + L]
        return trajectory

    def _reconstruct(self, U, sigma, Vt, groups):
        L, K = U.shape[0], Vt.shape[1]
        N = L + K - 1
        reconstructed = np.zeros(N)
        counts = np.zeros(N)
        for group in groups:
            X_group = np.zeros((L, K))
            for idx in group:
                s = sigma[idx]
                u = U[:, idx]
                vt = Vt[idx, :]
                X_group += s * np.outer(u, vt)
            for i in range(L):
                for j in range(K):
                    reconstructed[i + j] += X_group[i, j]
                    counts[i + j] += 1
        valid = counts > 0
        reconstructed[valid] /= counts[valid]
        return reconstructed

    def denoise_series(self, series):
        series = np.asarray(series, dtype=float)
        mask = ~np.isnan(series)
        if mask.sum() < self.window_length * 2:
            return series.copy()
        clean = series.copy()
        valid_idx = np.where(mask)[0]
        valid_vals = series[valid_idx]
        trajectory = self._embed(valid_vals)
        if trajectory is None:
            return clean
        try:
            U, sigma, Vt = np.linalg.svd(trajectory, full_matrices=False)
        except np.linalg.LinAlgError:
            return clean
        n_components = len(sigma)
        signal_groups = [list(range(min(self.n_signal_groups, n_components)))]
        reconstructed = self._reconstruct(U, sigma, Vt, signal_groups)
        result = valid_vals.copy()
        min_len = min(len(reconstructed), len(result))
        result[:min_len] = reconstructed[:min_len]
        clean[valid_idx] = result
        return clean

    def denoise_dataframe(self, df, columns):
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].values.astype(float)
            denoised = self.denoise_series(series)
            df[f'{col}_ssa'] = denoised
        return df

    def denoise_features(self, featured_df, continuous_cols=None):
        if continuous_cols is None:
            continuous_cols = [
                'temd', 'yellow_slope_norm', 'vol_ratio_20',
                'vol_ma5_ratio', 'vol_change', 'atr_ratio',
                'white_yellow_dist', 'macd_value', 'dif_value',
                'rsi_value', 'j_raw', 'j_min_5', 'dist_pct',
                'return_1d', 'return_5d', 'price_position_20',
                'index_yellow_slope', 'industry_oversold',
            ]
        existing_cols = [c for c in continuous_cols if c in featured_df.columns]
        if not existing_cols:
            return featured_df
        result = self.denoise_dataframe(featured_df, existing_cols)
        ssa_cols = [f'{c}_ssa' for c in existing_cols]
        for orig, ssa in zip(existing_cols, ssa_cols):
            if ssa in result.columns:
                result[orig] = result[ssa]
                result.drop(columns=[ssa], inplace=True)
        return result
