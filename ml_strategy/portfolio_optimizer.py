import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "pip_libs"))

from scipy.optimize import minimize


class BootstrappedMVO:

    def __init__(self, n_scenarios=500, block_size=5, lookback_days=200,
                 risk_aversion=0.5, max_weight=0.25, min_weight=0.0,
                 total_max_weight=0.75):
        self.n_scenarios = n_scenarios
        self.block_size = block_size
        self.lookback_days = lookback_days
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.total_max_weight = total_max_weight

    def _find_similar_days(self, oamv_state_df, current_date, n_days=200):
        if current_date not in oamv_state_df.index:
            return None
        current_x = oamv_state_df.loc[current_date, 'oamv_x']
        current_state = oamv_state_df.loc[current_date, 'oamv_state']

        same_state = oamv_state_df[oamv_state_df['oamv_state'] == current_state].copy()
        if len(same_state) == 0:
            return None
        same_state['x_diff'] = (same_state['oamv_x'] - current_x).abs()
        similar = same_state.nsmallest(min(n_days, len(same_state)), 'x_diff')
        return similar.index

    def _block_bootstrap(self, returns_matrix, n_scenarios, block_size):
        n_days, n_assets = returns_matrix.shape
        n_blocks = n_days // block_size + 1
        scenarios = np.zeros((n_scenarios, n_assets))

        for s in range(n_scenarios):
            selected_indices = []
            for _ in range(n_blocks):
                start = np.random.randint(0, n_days - block_size + 1)
                selected_indices.extend(range(start, start + block_size))
            selected_indices = selected_indices[:5]
            if len(selected_indices) == 0:
                continue
            sampled_returns = returns_matrix.iloc[selected_indices].values
            scenarios[s] = sampled_returns.sum(axis=0)

        return scenarios

    def _optimize_portfolio(self, scenarios, expected_returns):
        n_assets = scenarios.shape[1]
        cov_matrix = np.cov(scenarios.T)
        if n_assets == 1:
            cov_matrix = cov_matrix.reshape(1, 1)

        def objective(w):
            port_var = w @ cov_matrix @ w
            port_ret = w @ expected_returns
            return port_var - self.risk_aversion * port_ret

        constraints = [
            {'type': 'ineq', 'fun': lambda w: np.sum(w) - 0.01},
            {'type': 'ineq', 'fun': lambda w: self.total_max_weight - np.sum(w)},
        ]
        bounds = [(self.min_weight, self.max_weight)] * n_assets
        x0 = np.ones(n_assets) * min(self.max_weight, self.total_max_weight / n_assets)
        x0 = x0 / x0.sum() * min(self.total_max_weight, n_assets * self.max_weight)

        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 500})
            if result.success:
                weights = result.x
                weights = np.clip(weights, 0, self.max_weight)
                if weights.sum() > 0:
                    weights = weights / weights.sum() * min(weights.sum(), self.total_max_weight)
                return weights
        except Exception:
            pass

        equal_w = np.ones(n_assets) / n_assets * min(self.total_max_weight, n_assets * self.max_weight)
        return equal_w

    def optimize(self, candidate_codes, all_stock_data, oamv_state_df,
                 current_date, ml_probs):
        if len(candidate_codes) == 0:
            return np.array([]), candidate_codes

        if len(candidate_codes) == 1:
            return np.array([min(self.max_weight, self.total_max_weight)]), candidate_codes

        similar_dates = self._find_similar_days(oamv_state_df, current_date)
        if similar_dates is None or len(similar_dates) < self.block_size * 2:
            equal_w = np.ones(len(candidate_codes)) / len(candidate_codes) * self.total_max_weight
            return equal_w, candidate_codes

        returns_dict = {}
        valid_codes = []
        for code in candidate_codes:
            if code not in all_stock_data:
                continue
            df = all_stock_data[code]['data']
            available = df.index.intersection(similar_dates)
            if len(available) < self.block_size:
                continue
            returns = df.loc[available, 'Close'].pct_change().dropna()
            if len(returns) < self.block_size:
                continue
            returns_dict[code] = returns
            valid_codes.append(code)

        if len(valid_codes) < 2:
            equal_w = np.ones(len(candidate_codes)) / len(candidate_codes) * self.total_max_weight
            return equal_w, candidate_codes

        common_idx = returns_dict[valid_codes[0]].index
        for code in valid_codes[1:]:
            common_idx = common_idx.intersection(returns_dict[code].index)
        if len(common_idx) < self.block_size * 2:
            equal_w = np.ones(len(valid_codes)) / len(valid_codes) * self.total_max_weight
            return equal_w, valid_codes

        returns_matrix = pd.DataFrame(
            {code: returns_dict[code].loc[common_idx] for code in valid_codes}
        ).dropna()

        if len(returns_matrix) < self.block_size * 2:
            equal_w = np.ones(len(valid_codes)) / len(valid_codes) * self.total_max_weight
            return equal_w, valid_codes

        scenarios = self._block_bootstrap(
            returns_matrix, self.n_scenarios, self.block_size
        )

        expected_returns = np.array([ml_probs.get(code, 0.5) for code in valid_codes])

        weights = self._optimize_portfolio(scenarios, expected_returns)
        return weights, valid_codes

    def summary(self):
        return (
            f"Bootstrapped MVO:\n"
            f"  Scenarios: {self.n_scenarios}\n"
            f"  Block size: {self.block_size}\n"
            f"  Risk aversion: {self.risk_aversion}\n"
            f"  Max weight per asset: {self.max_weight}\n"
            f"  Max total weight: {self.total_max_weight}"
        )
