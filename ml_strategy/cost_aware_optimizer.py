import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "pip_libs"))

import cvxpy as cp


class CostAwarePortfolioOptimizer:

    def __init__(self, n_scenarios=500, block_size=5, lookback_days=200,
                 risk_aversion=0.5, cost_aversion=1.0, max_weight=0.25,
                 min_weight=0.0, total_max_weight=0.75,
                 impact_coefficient=0.4, spread_half=0.001,
                 total_capital=10000000):
        self.n_scenarios = n_scenarios
        self.block_size = block_size
        self.lookback_days = lookback_days
        self.risk_aversion = risk_aversion
        self.cost_aversion = cost_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.total_max_weight = total_max_weight
        self.impact_coefficient = impact_coefficient
        self.spread_half = spread_half
        self.total_capital = total_capital

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

    def _compute_daily_volumes(self, candidate_codes, all_stock_data, current_date, window=5):
        volumes = {}
        volatilities = {}
        for code in candidate_codes:
            if code not in all_stock_data:
                volumes[code] = 1e8
                volatilities[code] = 0.02
                continue
            df = all_stock_data[code]['data']
            loc = df.index.get_indexer([current_date], method='ffill')
            if len(loc) == 0 or loc[0] < 0:
                volumes[code] = 1e8
                volatilities[code] = 0.02
                continue
            end_idx = loc[0] + 1
            start_idx = max(0, end_idx - window)
            recent = df.iloc[start_idx:end_idx]
            if 'Volume' in recent.columns and 'Close' in recent.columns:
                daily_amounts = recent['Close'] * recent['Volume'] * 100
                volumes[code] = daily_amounts.mean() if len(daily_amounts) > 0 else 1e8
            else:
                volumes[code] = 1e8
            if 'Close' in recent.columns and len(recent) >= 14:
                ret = recent['Close'].pct_change().dropna()
                volatilities[code] = ret.std() * np.sqrt(252) if len(ret) > 1 else 0.02
            else:
                volatilities[code] = 0.02
        return volumes, volatilities

    def _cost_aware_optimize(self, expected_returns, cov_matrix, daily_volumes,
                              volatilities, candidate_codes):
        n = len(candidate_codes)
        if n == 0:
            return np.array([])

        w = cp.Variable(n, nonneg=True)
        mu = np.array(expected_returns)
        Sigma = np.array(cov_matrix)

        ret_term = mu @ w
        risk_term = cp.quad_form(w, cp.psd_wrap(Sigma))

        cost_term = 0.0
        for i in range(n):
            code = candidate_codes[i]
            vol_i = volatilities.get(code, 0.02)
            daily_vol_i = daily_volumes.get(code, 1e8)
            eta_i = self.impact_coefficient * vol_i
            cost_i = eta_i * w[i] * cp.sqrt(w[i] * self.total_capital / daily_vol_i) + self.spread_half * w[i]
            cost_term += cost_i

        objective = cp.Maximize(
            ret_term - self.risk_aversion * risk_term - self.cost_aversion * cost_term
        )

        constraints = [
            w >= self.min_weight,
            w <= self.max_weight,
            cp.sum(w) <= self.total_max_weight,
        ]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=500)
        except Exception:
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=1000)
            except Exception:
                pass

        if w.value is not None:
            result = np.clip(w.value, 0, self.max_weight)
            if result.sum() > self.total_max_weight:
                result = result / result.sum() * self.total_max_weight
            return result

        equal_w = np.ones(n) / n * min(self.total_max_weight, n * self.max_weight)
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
        cov_matrix = np.cov(scenarios.T)
        if len(valid_codes) == 1:
            cov_matrix = cov_matrix.reshape(1, 1)

        daily_volumes, volatilities = self._compute_daily_volumes(
            valid_codes, all_stock_data, current_date
        )

        weights = self._cost_aware_optimize(
            expected_returns, cov_matrix, daily_volumes, volatilities, valid_codes
        )
        return weights, valid_codes

    def summary(self):
        return (
            f"CostAwarePortfolioOptimizer:\n"
            f"  Scenarios: {self.n_scenarios}\n"
            f"  Block size: {self.block_size}\n"
            f"  Risk aversion (lambda): {self.risk_aversion}\n"
            f"  Cost aversion (gamma): {self.cost_aversion}\n"
            f"  Max weight per asset: {self.max_weight}\n"
            f"  Max total weight: {self.total_max_weight}\n"
            f"  Impact coefficient (Y): {self.impact_coefficient}\n"
            f"  Spread half: {self.spread_half}\n"
            f"  Total capital: {self.total_capital:,.0f}"
        )
