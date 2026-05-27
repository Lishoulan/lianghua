import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class PathSignatureBuilder:

    def __init__(self, truncation_level=2, path_dims=3, path_length=5,
                 normalize=True, lead_lag=True, lag_steps=1):
        self.truncation_level = truncation_level
        self.path_dims = path_dims
        self.path_length = path_length
        self.normalize = normalize
        self.lead_lag = lead_lag
        self.lag_steps = lag_steps

    def _build_path(self, df, end_idx):
        start_idx = max(0, end_idx - self.path_length + 1)
        window = df.iloc[start_idx:end_idx + 1]

        price_raw = window['Close'].values.astype(float)
        vol_raw = window['vol_ratio_20'].values.astype(float) if 'vol_ratio_20' in window.columns else np.ones(len(window))
        j_raw = window['j_clean'].values.astype(float) if 'j_clean' in window.columns else np.full(len(window), 50.0)

        for i in range(len(vol_raw)):
            if np.isnan(vol_raw[i]) or vol_raw[i] <= 0:
                vol_raw[i] = 1.0
        for i in range(len(j_raw)):
            if np.isnan(j_raw[i]):
                j_raw[i] = 50.0

        price_norm = np.diff(price_raw) / (price_raw[:-1] + 1e-8) if len(price_raw) > 1 else np.array([0.0])
        vol_norm = (vol_raw[1:] - vol_raw[:-1]) / (vol_raw[:-1] + 1e-8) if len(vol_raw) > 1 else np.array([0.0])
        j_norm = (j_raw[1:] - j_raw[:-1]) / 100.0 if len(j_raw) > 1 else np.array([0.0])

        path = np.column_stack([price_norm, vol_norm, j_norm])

        if self.lead_lag:
            path = self._build_lead_lag_path(path)

        if self.normalize and len(path) > 1:
            std = np.std(path, axis=0)
            std[std < 1e-8] = 1.0
            path = path / std

        return path

    def _build_lead_lag_path(self, path):
        d = path.shape[1]
        lag = self.lag_steps
        n = len(path)

        if n <= lag:
            return np.column_stack([path, path])

        lead_part = path[lag:]
        lag_part = path[:-lag]

        min_len = min(len(lead_part), len(lag_part))
        lead_part = lead_part[:min_len]
        lag_part = lag_part[:min_len]

        lead_lag_path = np.column_stack([lead_part, lag_part])

        return lead_lag_path

    @staticmethod
    def _compute_signature(path, truncation_level):
        n_steps, d = path.shape
        sig = []

        level1 = np.sum(path, axis=0)
        sig.extend(level1.tolist())

        if truncation_level >= 2:
            cum_path = np.cumsum(path, axis=0)
            for i in range(d):
                for j in range(d):
                    cross = np.sum(path[:, j] * cum_path[:, i] - path[:, i] * cum_path[:, j]) * 0.5
                    sig.append(float(np.sum(path[:, i]) * np.sum(path[:, j]) * 0.5 + cross))

        if truncation_level >= 3:
            from itertools import product
            for i, j, k in product(range(d), repeat=3):
                val = float(level1[i] * level1[j] * level1[k] / 6.0)
                sig.append(val)

        return np.array(sig)

    def compute_signature_features(self, df):
        df = df.copy()
        n = len(df)

        d = self.path_dims
        if self.lead_lag:
            d = 2 * d

        total_sig_len = sum(d ** l for l in range(1, self.truncation_level + 1))

        sig_array = np.zeros((n, total_sig_len))

        for i in range(self.path_length - 1, n):
            try:
                path = self._build_path(df, i)
                if len(path) < 2:
                    continue
                sig = self._compute_signature(path, self.truncation_level)
                if len(sig) == total_sig_len:
                    sig_array[i] = sig
            except Exception:
                continue

        col_names = self.get_signature_col_names()

        for k, name in enumerate(col_names):
            if k < total_sig_len:
                df[name] = sig_array[:, k]

        return df

    def get_signature_col_names(self):
        d = self.path_dims
        if self.lead_lag:
            d = 2 * d
        col_names = []
        for level in range(1, self.truncation_level + 1):
            n_terms = d ** level
            for j in range(n_terms):
                col_names.append(f'sig_l{level}_{j}')
        return col_names

    def summary(self):
        d = self.path_dims
        if self.lead_lag:
            d = 2 * d
        n_features = sum(d ** l for l in range(1, self.truncation_level + 1))
        return (
            f"PathSignatureBuilder:\n"
            f"  Truncation level: {self.truncation_level}\n"
            f"  Path dimensions: {self.path_dims} (lead-lag: {self.lead_lag} -> effective {d})\n"
            f"  Path length: {self.path_length} days\n"
            f"  Lag steps: {self.lag_steps}\n"
            f"  Normalize: {self.normalize}\n"
            f"  Signature features: {n_features}"
        )
