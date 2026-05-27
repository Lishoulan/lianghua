import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class DynamicSpreadPredictor:

    def __init__(self, base_spread_half=0.001, alpha=0.0003, beta=0.5,
                 min_spread_half=0.0005, max_spread_half=0.01):
        self.base_spread_half = base_spread_half
        self.alpha = alpha
        self.beta = beta
        self.min_spread_half = min_spread_half
        self.max_spread_half = max_spread_half

    def predict_spread_half(self, atr_ratio, vol_ratio_20, market_vol=None):
        atr_ratio = np.asarray(atr_ratio, dtype=float)
        vol_ratio_20 = np.asarray(vol_ratio_20, dtype=float)

        vol_ratio_safe = np.where(
            (np.isnan(vol_ratio_20)) | (vol_ratio_20 <= 0),
            1.0,
            vol_ratio_20
        )
        atr_safe = np.where(np.isnan(atr_ratio), 0.02, atr_ratio)

        liquidity_stress = atr_safe / (vol_ratio_safe + 1e-8)

        spread_half = self.alpha + self.beta * liquidity_stress

        if market_vol is not None:
            market_vol = np.asarray(market_vol, dtype=float)
            market_vol_safe = np.where(np.isnan(market_vol), 0.02, market_vol)
            market_stress = np.clip(market_vol_safe / 0.02, 0.5, 3.0)
            spread_half = spread_half * market_stress

        spread_half = np.clip(spread_half, self.min_spread_half, self.max_spread_half)

        return spread_half

    def predict_single(self, atr_ratio, vol_ratio_20, market_vol=None):
        result = self.predict_spread_half(
            np.array([atr_ratio]),
            np.array([vol_ratio_20]),
            np.array([market_vol]) if market_vol is not None else None
        )
        return float(result[0])

    def batch_predict(self, stock_features):
        results = {}
        for code, feat in stock_features.items():
            atr_ratio = feat.get('atr_ratio', 0.02)
            vol_ratio = feat.get('vol_ratio_20', 1.0)
            market_vol = feat.get('market_vol', None)
            results[code] = self.predict_single(atr_ratio, vol_ratio, market_vol)
        return results

    def summary(self):
        return (
            f"DynamicSpreadPredictor:\n"
            f"  Base spread half: {self.base_spread_half:.4f} ({self.base_spread_half*100:.2f}%)\n"
            f"  Alpha (intercept): {self.alpha:.4f}\n"
            f"  Beta (liquidity stress): {self.beta:.2f}\n"
            f"  Range: [{self.min_spread_half*100:.2f}%, {self.max_spread_half*100:.2f}%]\n"
            f"  Formula: spread = alpha + beta * (ATR_ratio / vol_ratio_20) * market_stress"
        )
