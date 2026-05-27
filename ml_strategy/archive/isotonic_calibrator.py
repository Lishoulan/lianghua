import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:

    def __init__(self, out_of_bounds='clip', y_min=0.01, y_max=0.99):
        self.out_of_bounds = out_of_bounds
        self.y_min = y_min
        self.y_max = y_max
        self._ir = None
        self._fitted = False

    def fit(self, y_true, y_prob):
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)

        mask = ~(np.isnan(y_true) | np.isnan(y_prob))
        y_true = y_true[mask]
        y_prob = y_prob[mask]

        if len(y_true) < 10:
            self._fitted = False
            return self

        sort_idx = np.argsort(y_prob)
        y_prob_sorted = y_prob[sort_idx]
        y_true_sorted = y_true[sort_idx]

        self._ir = IsotonicRegression(
            out_of_bounds=self.out_of_bounds,
            y_min=self.y_min,
            y_max=self.y_max,
        )
        self._ir.fit(y_prob_sorted, y_true_sorted)
        self._fitted = True
        return self

    def transform(self, y_prob):
        if not self._fitted or self._ir is None:
            return np.asarray(y_prob, dtype=float)

        y_prob = np.asarray(y_prob, dtype=float)
        result = self._ir.transform(y_prob)
        result = np.clip(result, self.y_min, self.y_max)
        return result

    def fit_transform(self, y_true, y_prob):
        self.fit(y_true, y_prob)
        return self.transform(y_prob)

    def calibration_error(self, y_true, y_prob, n_bins=10):
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)

        mask = ~(np.isnan(y_true) | np.isnan(y_prob))
        y_true = y_true[mask]
        y_prob = y_prob[mask]

        if len(y_true) < n_bins:
            return float('nan')

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(y_true)

        for i in range(n_bins):
            mask_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            count = mask_bin.sum()
            if count == 0:
                continue
            avg_prob = y_prob[mask_bin].mean()
            avg_true = y_true[mask_bin].mean()
            ece += count / total * abs(avg_prob - avg_true)

        return ece

    def summary(self):
        status = "Fitted" if self._fitted else "Not fitted"
        return (
            f"IsotonicCalibrator:\n"
            f"  Status: {status}\n"
            f"  Output range: [{self.y_min}, {self.y_max}]\n"
            f"  Out-of-bounds: {self.out_of_bounds}"
        )
