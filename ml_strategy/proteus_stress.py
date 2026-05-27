import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from scipy import stats as sp_stats


class ARMAGARCHFitter:

    def __init__(self, ar_order=1, ma_order=1, garch_p=1, garch_q=1,
                 dist='normal', max_iter=500, tol=1e-6):
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.dist = dist
        self.max_iter = max_iter
        self.tol = tol
        self.ar_coeffs = None
        self.ma_coeffs = None
        self.omega = None
        self.alpha = None
        self.beta = None
        self.sigma2 = None
        self.residuals = None
        self.fitted = False

    def fit(self, returns):
        returns = np.asarray(returns, dtype=float)
        n = len(returns)
        if n < 30:
            return

        y = returns.copy()
        mu = np.mean(y)
        y_demean = y - mu

        ar_coeffs = np.zeros(self.ar_order)
        ma_coeffs = np.zeros(self.ma_order)

        for iteration in range(self.max_iter):
            residuals = y_demean.copy()
            for t in range(max(self.ar_order, self.ma_order), n):
                for p in range(self.ar_order):
                    residuals[t] -= ar_coeffs[p] * y_demean[t - 1 - p]
                for q in range(self.ma_order):
                    if t - 1 - q >= 0:
                        residuals[t] -= ma_coeffs[q] * residuals[t - 1 - q]

            omega = 0.1 * np.var(residuals)
            alpha = np.array([0.1])
            beta = np.array([0.85])

            sigma2 = np.zeros(n)
            sigma2[0] = omega / (1 - alpha[0] - beta[0]) if (1 - alpha[0] - beta[0]) > 0 else np.var(residuals)

            for t in range(1, n):
                sigma2[t] = omega
                for i in range(min(self.garch_q, t)):
                    sigma2[t] += alpha[i] * residuals[t - 1 - i] ** 2
                for j in range(min(self.garch_p, t)):
                    sigma2[t] += beta[j] * sigma2[t - 1 - j]
                sigma2[t] = max(sigma2[t], 1e-8)

            new_ar = ar_coeffs.copy()
            new_ma = ma_coeffs.copy()

            if self.ar_order > 0:
                X_ar = np.zeros((n - self.ar_order, self.ar_order))
                for p in range(self.ar_order):
                    X_ar[:, p] = y_demean[self.ar_order - 1 - p:n - 1 - p]
                y_target = y_demean[self.ar_order:]
                w = 1.0 / (sigma2[self.ar_order:] + 1e-8)
                try:
                    new_ar = np.linalg.lstsq(X_ar * w[:, None], y_target * w, rcond=None)[0]
                except Exception:
                    pass

            self.ar_coeffs = new_ar
            self.ma_coeffs = new_ma
            self.omega = omega
            self.alpha = alpha
            self.beta = beta
            self.sigma2 = sigma2
            self.residuals = residuals
            self.fitted = True
            break

    def generate(self, n_steps, rng=None):
        if not self.fitted:
            return np.random.randn(n_steps) * 0.02

        if rng is None:
            rng = np.random.default_rng()

        y = np.zeros(n_steps)
        sigma2 = np.zeros(n_steps)
        eps = np.zeros(n_steps)

        sigma2[0] = self.omega / (1 - self.alpha[0] - self.beta[0]) if (1 - self.alpha[0] - self.beta[0]) > 0 else np.var(self.residuals)
        eps[0] = rng.normal(0, np.sqrt(sigma2[0]))
        y[0] = eps[0]

        for t in range(1, n_steps):
            sigma2[t] = self.omega
            for i in range(min(self.garch_q, t)):
                sigma2[t] += self.alpha[i] * eps[t - 1 - i] ** 2
            for j in range(min(self.garch_p, t)):
                sigma2[t] += self.beta[j] * sigma2[t - 1 - j]
            sigma2[t] = max(sigma2[t], 1e-8)

            eps[t] = rng.normal(0, np.sqrt(sigma2[t]))

            y[t] = eps[t]
            for p in range(min(self.ar_order, t)):
                y[t] += self.ar_coeffs[p] * y[t - 1 - p]
            for q in range(min(self.ma_order, t)):
                y[t] += self.ma_coeffs[q] * eps[t - 1 - q]

        return y


class ProteuSGenerator:

    def __init__(self, n_scenarios=1000, seed=42, drift_magnitude=3.0,
                 drift_types=None, n_drift_points=3):
        self.n_scenarios = n_scenarios
        self.seed = seed
        self.drift_magnitude = drift_magnitude
        self.drift_types = drift_types or ['gradual', 'abrupt', 'incremental', 'recurring']
        self.n_drift_points = n_drift_points
        self.fitted_models = {}
        self.scenario_metadata = []

    def fit(self, index_df, stock_data_dict=None):
        if index_df is not None and 'Close' in index_df.columns:
            returns = index_df['Close'].pct_change().dropna().values
            if len(returns) > 60:
                fitter = ARMAGARCHFitter(ar_order=1, ma_order=1, garch_p=1, garch_q=1)
                fitter.fit(returns)
                self.fitted_models['index'] = fitter

        if stock_data_dict is not None:
            count = 0
            for code, info in stock_data_dict.items():
                if count >= 20:
                    break
                df = info.get('data', info) if isinstance(info, dict) else info
                if df is not None and 'Close' in df.columns:
                    returns = df['Close'].pct_change().dropna().values
                    if len(returns) > 60:
                        fitter = ARMAGARCHFitter(ar_order=1, ma_order=1, garch_p=1, garch_q=1)
                        fitter.fit(returns)
                        self.fitted_models[code] = fitter
                        count += 1

        print(f"  ProteuS fitted {len(self.fitted_models)} ARMAGARCH models")

    def _inject_drift(self, series, drift_type, drift_point, magnitude):
        n = len(series)
        result = series.copy()

        if drift_type == 'abrupt':
            shift = magnitude * np.std(series) * np.random.choice([-1, 1])
            result[drift_point:] += shift

        elif drift_type == 'gradual':
            ramp = np.linspace(0, magnitude * np.std(series), n - drift_point)
            ramp *= np.random.choice([-1, 1])
            result[drift_point:] += ramp

        elif drift_type == 'incremental':
            n_increments = min(5, n - drift_point)
            increment_points = np.sort(np.random.choice(
                range(drift_point, n), size=n_increments, replace=False))
            for ip in increment_points:
                shift = magnitude * np.std(series) * 0.2 * np.random.choice([-1, 1])
                result[ip:] += shift

        elif drift_type == 'recurring':
            period = max(10, (n - drift_point) // 4)
            for t in range(drift_point, n):
                cycle = np.sin(2 * np.pi * (t - drift_point) / period)
                result[t] += magnitude * np.std(series) * 0.5 * cycle

        return result

    def _inject_extreme_event(self, series, event_type, event_point):
        n = len(series)
        result = series.copy()

        if event_type == 'flash_crash':
            crash_depth = np.random.uniform(-0.08, -0.03)
            crash_duration = np.random.randint(2, 6)
            for t in range(event_point, min(event_point + crash_duration, n)):
                result[t] += crash_depth / crash_duration
            recovery_start = min(event_point + crash_duration, n - 1)
            for t in range(recovery_start, min(recovery_start + crash_duration * 2, n)):
                result[t] += abs(crash_depth) * 0.3 / (crash_duration * 2)

        elif event_type == 'liquidity_crisis':
            vol_multiplier = np.random.uniform(3.0, 6.0)
            for t in range(event_point, min(event_point + 10, n)):
                result[t] *= vol_multiplier
            trend = np.random.uniform(-0.02, -0.005)
            for t in range(event_point, min(event_point + 15, n)):
                result[t] += trend

        elif event_type == 'sector_rotation':
            shift = np.random.uniform(0.01, 0.03)
            for t in range(event_point, min(event_point + 20, n)):
                result[t] += shift * (1 - (t - event_point) / 20)

        return result

    def generate_scenarios(self, n_steps=120):
        rng = np.random.default_rng(self.seed)
        scenarios = []
        self.scenario_metadata = []

        if 'index' not in self.fitted_models:
            print("  Warning: No index model fitted, using random walk")
            base_model = ARMAGARCHFitter()
            base_model.fitted = True
            base_model.omega = 1e-5
            base_model.alpha = np.array([0.1])
            base_model.beta = np.array([0.85])
            base_model.ar_coeffs = np.array([0.05])
            base_model.ma_coeffs = np.array([0.02])
            base_model.residuals = np.random.randn(100) * 0.02
        else:
            base_model = self.fitted_models['index']

        drift_types = self.drift_types
        extreme_events = ['flash_crash', 'liquidity_crisis', 'sector_rotation', None]

        for i in range(self.n_scenarios):
            base_series = base_model.generate(n_steps, rng=rng)

            n_drifts = rng.integers(1, self.n_drift_points + 1)
            drift_points = sorted(rng.choice(
                range(n_steps // 4, 3 * n_steps // 4),
                size=n_drifts, replace=False))

            scenario = base_series.copy()
            drift_info = []

            for dp in drift_points:
                dtype = drift_types[rng.integers(0, len(drift_types))]
                mag = self.drift_magnitude * rng.uniform(0.5, 1.5)
                scenario = self._inject_drift(scenario, dtype, dp, mag)
                drift_info.append({
                    'type': dtype,
                    'point': int(dp),
                    'magnitude': float(mag),
                })

            extreme = extreme_events[rng.integers(0, len(extreme_events))]
            if extreme is not None:
                event_point = rng.integers(n_steps // 3, 2 * n_steps // 3)
                scenario = self._inject_extreme_event(scenario, extreme, event_point)
                drift_info.append({
                    'type': f'extreme_{extreme}',
                    'point': int(event_point),
                    'magnitude': 0.0,
                })

            scenarios.append(scenario)
            self.scenario_metadata.append({
                'scenario_id': i,
                'drift_points': drift_info,
            })

        return scenarios

    def stress_test_addm(self, scenarios, addm_detector_class, param_grid,
                         metric='detection_delay'):
        results = []

        for params in param_grid:
            detector = addm_detector_class(**params)
            total_delay = 0
            total_detected = 0
            total_false_alarms = 0
            n_scenarios = len(scenarios)

            for i, scenario in enumerate(scenarios):
                detector.reset()
                meta = self.scenario_metadata[i]

                true_drift_points = set()
                for dp_info in meta['drift_points']:
                    if 'point' in dp_info:
                        true_drift_points.add(dp_info['point'])

                detected_points = set()
                for t in range(len(scenario)):
                    detector.update(scenario[t])
                    if detector.drift_detected:
                        detected_points.add(t)

                for true_pt in true_drift_points:
                    detected_after = [d for d in detected_points if d >= true_pt and d <= true_pt + 20]
                    if detected_after:
                        total_delay += min(detected_after) - true_pt
                        total_detected += 1

                false_alarms = [d for d in detected_points
                                if not any(abs(d - tp) <= 20 for tp in true_drift_points)]
                total_false_alarms += len(false_alarms)

            detection_rate = total_detected / max(len(scenarios), 1)
            avg_delay = total_delay / max(total_detected, 1)
            avg_false_alarms = total_false_alarms / max(len(scenarios), 1)

            results.append({
                'params': params,
                'detection_rate': float(detection_rate),
                'avg_delay': float(avg_delay),
                'avg_false_alarms': float(avg_false_alarms),
                'score': float(detection_rate - 0.5 * avg_false_alarms - 0.01 * avg_delay),
            })

        results.sort(key=lambda x: -x['score'])
        return results

    def summary(self):
        return (
            f"ProteuSGenerator:\n"
            f"  N scenarios: {self.n_scenarios}\n"
            f"  Drift types: {self.drift_types}\n"
            f"  Drift magnitude: {self.drift_magnitude}\n"
            f"  Fitted models: {len(self.fitted_models)}\n"
            f"  Seed: {self.seed}"
        )
