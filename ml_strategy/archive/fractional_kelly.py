import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class FractionalKelly:

    def __init__(self, atr_mult_upper=1.5, atr_mult_lower=0.8,
                 entropy_shrinkage_theta=0.5, kelly_fraction=0.5,
                 min_weight=0.0, max_weight=0.25):
        self.atr_mult_upper = atr_mult_upper
        self.atr_mult_lower = atr_mult_lower
        self.entropy_shrinkage_theta = entropy_shrinkage_theta
        self.kelly_fraction = kelly_fraction
        self.min_weight = min_weight
        self.max_weight = max_weight

    def compute_dynamic_reward_ratio(self, atr_upper, atr_lower):
        if atr_lower <= 0:
            return 1.875
        return atr_upper / atr_lower

    def compute_kelly_weight(self, p_calibrated, entropy_sigma_sq,
                             sigma_sq_max=None, reward_ratio=1.875):
        if p_calibrated <= 0 or p_calibrated >= 1:
            return 0.0

        kelly_raw = (p_calibrated * reward_ratio - (1.0 - p_calibrated)) / reward_ratio

        if kelly_raw <= 0:
            return 0.0

        if sigma_sq_max is not None and sigma_sq_max > 0:
            shrinkage = 1.0 - self.entropy_shrinkage_theta * (entropy_sigma_sq / sigma_sq_max)
        elif entropy_sigma_sq > 0:
            shrinkage = 1.0 - self.entropy_shrinkage_theta * min(entropy_sigma_sq, 1.0)
        else:
            shrinkage = 1.0

        shrinkage = max(shrinkage, 0.0)

        kelly_adjusted = kelly_raw * shrinkage

        kelly_fractional = kelly_adjusted * self.kelly_fraction

        kelly_fractional = np.clip(kelly_fractional, self.min_weight, self.max_weight)

        return float(kelly_fractional)

    def compute_catboost_entropy(self, model, features_df):
        if model is None or not hasattr(model, 'model'):
            return 0.0

        try:
            cb_model = model.model
            if not hasattr(cb_model, 'predict_proba'):
                return 0.0

            leaf_indices = cb_model.calc_leaf_indexes(features_df)
            if leaf_indices is None or len(leaf_indices) == 0:
                return 0.0

            n_trees = leaf_indices.shape[1] if len(leaf_indices.shape) > 1 else 1
            leaf_counts = np.zeros(n_trees)
            for t in range(n_trees):
                if len(leaf_indices.shape) > 1:
                    unique_leaves = len(np.unique(leaf_indices[:, t]))
                else:
                    unique_leaves = len(np.unique(leaf_indices))
                leaf_counts[t] = unique_leaves

            avg_entropy = np.mean(leaf_counts) / max(leaf_counts.max(), 1)
            return float(avg_entropy)
        except Exception:
            return 0.0

    def compute_prediction_variance(self, probs_list):
        if len(probs_list) < 2:
            return 0.0

        probs_array = np.array(probs_list)
        variance = np.var(probs_array, axis=0)
        return float(np.mean(variance))

    def adjust_mvo_weights(self, mvo_weights, candidate_codes, p_calibrated_dict,
                           entropy_dict=None, sigma_sq_max=None,
                           atr_upper=1.5, atr_lower=0.8):
        reward_ratio = self.compute_dynamic_reward_ratio(atr_upper, atr_lower)

        adjusted_weights = np.copy(mvo_weights)

        for i, code in enumerate(candidate_codes):
            p_cal = p_calibrated_dict.get(code, 0.5)
            entropy = 0.0
            if entropy_dict is not None:
                entropy = entropy_dict.get(code, 0.0)

            kelly_w = self.compute_kelly_weight(
                p_calibrated=p_cal,
                entropy_sigma_sq=entropy,
                sigma_sq_max=sigma_sq_max,
                reward_ratio=reward_ratio,
            )

            adjusted_weights[i] = min(adjusted_weights[i], kelly_w)

        total = adjusted_weights.sum()
        if total > 0:
            adjusted_weights = adjusted_weights / total * mvo_weights.sum()

        adjusted_weights = np.clip(adjusted_weights, self.min_weight, self.max_weight)

        return adjusted_weights

    def summary(self):
        return (
            f"FractionalKelly:\n"
            f"  ATR mult upper: {self.atr_mult_upper}\n"
            f"  ATR mult lower: {self.atr_mult_lower}\n"
            f"  Default reward ratio: {self.atr_mult_upper/self.atr_mult_lower:.3f}\n"
            f"  Entropy shrinkage theta: {self.entropy_shrinkage_theta}\n"
            f"  Kelly fraction: {self.kelly_fraction}\n"
            f"  Weight range: [{self.min_weight}, {self.max_weight}]"
        )
