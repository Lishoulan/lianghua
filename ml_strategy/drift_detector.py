import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class ADDMDriftDetector:

    def __init__(self, ar_order=3, ph_threshold=2.0, ph_delta=0.01,
                 window_size=20, min_samples=20, use_vol_filter=True,
                 decay_lambda=0.005, retrain_cooldown_days=10):
        self.ar_order = ar_order
        self.ph_threshold = ph_threshold
        self.ph_delta = ph_delta
        self.window_size = window_size
        self.min_samples = min_samples
        self.use_vol_filter = use_vol_filter
        self.decay_lambda = decay_lambda
        self.retrain_cooldown_days = retrain_cooldown_days
        self.error_history = []
        self.vol_history = []
        self.ph_cumsum = 0.0
        self.ph_min_cumsum = 0.0
        self.drift_detected = False
        self.drift_count = 0
        self.ar_coeffs = None
        self.retrain_needed = False
        self.last_retrain_day = -999

    def _compute_logloss(self, y_true, y_prob):
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

    def _compute_mse(self, y_true, y_prob):
        return np.mean((y_true - y_prob) ** 2)

    def set_market_volatility(self, vol_value):
        self.vol_history.append(vol_value)

    def update(self, y_true, y_prob):
        if len(y_true) == 0:
            return False
        error = self._compute_logloss(np.asarray(y_true), np.asarray(y_prob))
        self.error_history.append(error)
        if len(self.error_history) < self.min_samples:
            return False
        recent_errors = self.error_history[-self.window_size:]
        if len(recent_errors) < self.ar_order + 1:
            return False
        self._fit_ar(recent_errors)
        residual = self._compute_residual(recent_errors)
        if self.use_vol_filter and len(self.vol_history) >= self.min_samples:
            vol_value = self.vol_history[-1]
            if vol_value > 0:
                residual = residual / vol_value
        self.drift_detected = self._page_hinkley_test(residual)
        if self.drift_detected:
            self.drift_count += 1
            self.retrain_needed = True
        return self.drift_detected

    def _fit_ar(self, errors):
        errors = np.array(errors)
        n = len(errors)
        p = self.ar_order
        if n <= p:
            return
        Y = errors[p:]
        X = np.zeros((n - p, p))
        for i in range(p):
            X[:, i] = errors[p - 1 - i:n - 1 - i] if i > 0 else errors[p - 1:n - 1]
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            self.ar_coeffs = coeffs
        except np.linalg.LinAlgError:
            self.ar_coeffs = None

    def _compute_residual(self, errors):
        errors = np.array(errors)
        if self.ar_coeffs is None:
            return errors[-1] - np.mean(errors)
        p = len(self.ar_coeffs)
        if len(errors) <= p:
            return errors[-1] - np.mean(errors)
        predicted = np.dot(self.ar_coeffs, errors[-p:][::-1])
        return errors[-1] - predicted

    def _page_hinkley_test(self, residual):
        self.ph_cumsum += residual - self.ph_delta
        if self.ph_cumsum < self.ph_min_cumsum:
            self.ph_min_cumsum = self.ph_cumsum
        if (self.ph_cumsum - self.ph_min_cumsum) > self.ph_threshold:
            self.ph_cumsum = 0.0
            self.ph_min_cumsum = 0.0
            return True
        return False

    def compute_garch_vol(self, returns_series, p=1, q=1):
        returns = np.asarray(returns_series, dtype=np.float64).flatten()
        n = len(returns)
        if n < max(p, q) + 2:
            return np.array([])
        var_r = np.var(returns)
        alpha_init = 0.1 / p
        beta_init = 0.85 / q
        omega_init = var_r * (1.0 - alpha_init * p - beta_init * q)
        if omega_init <= 0:
            omega_init = var_r * 0.05
            alpha_init = 0.05 / p
            beta_init = 0.9 / q
        omega = omega_init
        alpha = np.full(p, alpha_init)
        beta = np.full(q, beta_init)
        sigma2 = np.full(n, var_r)
        for t in range(max(p, q), n):
            arch_term = 0.0
            for i in range(p):
                arch_term += alpha[i] * returns[t - 1 - i] ** 2
            garch_term = 0.0
            for j in range(q):
                garch_term += beta[j] * sigma2[t - 1 - j]
            sigma2[t] = omega + arch_term + garch_term
            if sigma2[t] <= 0:
                sigma2[t] = 1e-8
        best_ll = -np.inf
        best_params = (omega, alpha.copy(), beta.copy())
        best_sigma2 = sigma2.copy()
        for _ in range(20):
            ll = 0.0
            for t in range(max(p, q), n):
                ll += -0.5 * (np.log(2 * np.pi) + np.log(sigma2[t]) + returns[t] ** 2 / sigma2[t])
            if ll > best_ll:
                best_ll = ll
                best_params = (omega, alpha.copy(), beta.copy())
                best_sigma2 = sigma2.copy()
            omega_grad = 0.0
            alpha_grad = np.zeros(p)
            beta_grad = np.zeros(q)
            for t in range(max(p, q), n):
                d_dsigma2 = 0.5 * (returns[t] ** 2 / sigma2[t] ** 2 - 1.0 / sigma2[t])
                omega_grad += d_dsigma2
                for i in range(p):
                    alpha_grad[i] += d_dsigma2 * returns[t - 1 - i] ** 2
                for j in range(q):
                    beta_grad[j] += d_dsigma2 * sigma2[t - 1 - j]
            lr = 1e-5
            omega_new = omega + lr * omega_grad
            alpha_new = alpha + lr * alpha_grad
            beta_new = beta + lr * beta_grad
            if omega_new <= 0:
                omega_new = omega
            alpha_new = np.clip(alpha_new, 1e-6, None)
            beta_new = np.clip(beta_new, 1e-6, None)
            if np.sum(alpha_new) + np.sum(beta_new) >= 1.0:
                scale = 0.999 / (np.sum(alpha_new) + np.sum(beta_new))
                alpha_new *= scale
                beta_new *= scale
            omega = omega_new
            alpha = alpha_new
            beta = beta_new
            for t in range(max(p, q), n):
                arch_term = 0.0
                for i in range(p):
                    arch_term += alpha[i] * returns[t - 1 - i] ** 2
                garch_term = 0.0
                for j in range(q):
                    garch_term += beta[j] * sigma2[t - 1 - j]
                sigma2[t] = omega + arch_term + garch_term
                if sigma2[t] <= 0:
                    sigma2[t] = 1e-8
        return np.sqrt(best_sigma2)

    def compute_temporal_weights(self, dates, decay_lambda=None):
        if len(dates) == 0:
            return np.ones(1)
        if decay_lambda is None:
            decay_lambda = self.decay_lambda
        dates = pd.to_datetime(dates)
        max_date = dates.max()
        weights = np.exp(-decay_lambda * (max_date - dates).days.values.astype(float))
        weights = weights / weights.sum() * len(weights)
        return weights

    def should_retrain(self, current_day_index):
        if not self.retrain_needed:
            return False
        if (current_day_index - self.last_retrain_day) < self.retrain_cooldown_days:
            return False
        return True

    def mark_retrained(self, current_day_index):
        self.retrain_needed = False
        self.last_retrain_day = current_day_index
        self.ph_cumsum = 0.0
        self.ph_min_cumsum = 0.0

    def reset(self):
        self.error_history = []
        self.vol_history = []
        self.ph_cumsum = 0.0
        self.ph_min_cumsum = 0.0
        self.drift_detected = False
        self.retrain_needed = False

    def summary(self):
        vol_status = "ON" if self.use_vol_filter else "OFF"
        vol_count = len(self.vol_history)
        latest_vol = self.vol_history[-1] if self.vol_history else None
        vol_line = f"  Latest vol value: {latest_vol:.6f}\n" if latest_vol is not None else "  Latest vol value: N/A\n"
        return (
            f"ADDM Drift Detector:\n"
            f"  AR order: {self.ar_order}\n"
            f"  PH threshold: {self.ph_threshold}\n"
            f"  PH delta: {self.ph_delta}\n"
            f"  Vol filter: {vol_status}\n"
            f"  Vol history length: {vol_count}\n"
            f"{vol_line}"
            f"  Error history length: {len(self.error_history)}\n"
            f"  Drift detections: {self.drift_count}\n"
            f"  Current drift: {self.drift_detected}\n"
            f"  Retrain needed: {self.retrain_needed}\n"
            f"  Decay lambda: {self.decay_lambda}\n"
            f"  Retrain cooldown: {self.retrain_cooldown_days} days"
        )
