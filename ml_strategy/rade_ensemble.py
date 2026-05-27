import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class RADEEnsemble:

    def __init__(self, gamma=0.5, min_catboost_weight=0.2, max_catboost_weight=0.9):
        self.gamma = gamma
        self.min_catboost_weight = min_catboost_weight
        self.max_catboost_weight = max_catboost_weight
        self.catboost_model = None
        self.kan_model = None
        self.last_catboost_weight = 0.5

    def set_models(self, catboost_model, kan_trainer):
        self.catboost_model = catboost_model
        self.kan_model = kan_trainer

    def compute_catboost_weight(self, oamv_x_pct, atr_ratio):
        amv_bias = oamv_x_pct / 100.0
        raw = 1.0 / (1.0 + np.exp(-self.gamma * amv_bias * atr_ratio * 100))
        w = self.min_catboost_weight + raw * (self.max_catboost_weight - self.min_catboost_weight)
        return np.clip(w, self.min_catboost_weight, self.max_catboost_weight)

    def predict(self, features_df, oamv_x_pct=0.0, atr_ratio=0.02):
        if self.catboost_model is None:
            return np.full(len(features_df), 0.5)

        cat_probs = self.catboost_model.predict_proba(features_df)

        if self.kan_model is None:
            return cat_probs

        kan_probs = self.kan_model.predict_proba(features_df)

        w_cb = self.compute_catboost_weight(oamv_x_pct, atr_ratio)
        self.last_catboost_weight = w_cb
        w_kan = 1.0 - w_cb

        ensemble_probs = w_cb * cat_probs + w_kan * kan_probs
        return ensemble_probs

    def get_weight_info(self):
        return {
            'catboost_weight': round(self.last_catboost_weight, 3),
            'kan_weight': round(1.0 - self.last_catboost_weight, 3),
        }

    def summary(self):
        w = self.get_weight_info()
        return (
            f"RADE Ensemble:\n"
            f"  CatBoost weight: {w['catboost_weight']}\n"
            f"  KAN weight: {w['kan_weight']}\n"
            f"  Gamma: {self.gamma}\n"
            f"  CatBoost range: [{self.min_catboost_weight}, {self.max_catboost_weight}]"
        )
