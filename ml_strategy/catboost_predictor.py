import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "pip_libs"))
from catboost import CatBoostClassifier, Pool
from ml_strategy.amse_loss import AMSELossCatBoost


class CatBoostPredictor:

    FEATURE_COLS = [
        'price_zone', 'j_zone', 'k_pattern', 'temd',
        'yellow_slope_sign', 'yellow_slope_norm', 'vol_ratio_20',
        'vol_ma5_ratio', 'vol_change', 'atr_ratio',
        'white_above_yellow', 'white_yellow_dist', 'macd_value',
        'dif_value', 'macd_above_zero', 'rsi_value',
        'j_raw', 'j_min_5', 'j_rising', 'dist_pct',
        'return_1d', 'return_5d', 'price_position_20', 'low_above_yellow',
        'index_yellow_slope', 'index_state2_prob', 'industry_oversold',
        'pwvc', 'close_position_20', 'pwvc_distribution', 'j_clean',
    ]

    CATEGORICAL_FEATURES = ['price_zone', 'j_zone', 'k_pattern', 'pwvc_distribution']

    def __init__(self, buy_threshold=0.42, l2_leaf_reg=8, max_depth=4,
                 use_amse_loss=False, amse_omega=10.0):
        self.buy_threshold = buy_threshold
        self.l2_leaf_reg = l2_leaf_reg
        self.max_depth = max_depth
        self.use_amse_loss = use_amse_loss
        self.amse_omega = amse_omega
        self.model = None
        self.feature_names = list(self.FEATURE_COLS)

    def prepare_features(self, df):
        available = [col for col in self.FEATURE_COLS if col in df.columns]
        features_df = df[available].copy()
        features_df = features_df.dropna(subset=available)
        self.feature_names = available
        return features_df

    def train(self, features_df, labels, eval_features_df=None, eval_labels=None):
        cat_features = [
            features_df.columns.get_loc(col)
            for col in self.CATEGORICAL_FEATURES
            if col in features_df.columns
        ]
        loss_fn = AMSELossCatBoost(omega=self.amse_omega) if self.use_amse_loss else 'Logloss'
        eval_metric = loss_fn if self.use_amse_loss else 'Logloss'
        params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': self.max_depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'loss_function': loss_fn,
            'eval_metric': eval_metric,
            'random_seed': 42,
            'verbose': 100,
            'early_stopping_rounds': 50,
            'cat_features': cat_features,
            'auto_class_weights': 'Balanced',
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 1.0,
        }
        self.model = CatBoostClassifier(**params)
        fit_kwargs = {
            'X': features_df,
            'y': labels,
        }
        if eval_features_df is not None and eval_labels is not None:
            fit_kwargs['eval_set'] = (eval_features_df, eval_labels)
        self.model.fit(**fit_kwargs)
        print(f"Best iteration: {self.model.get_best_iteration()}")
        print(f"Best score: {self.model.get_best_score()}")

    def predict_proba(self, features_df):
        if self.model is None:
            return np.full(len(features_df), 0.5)
        proba = self.model.predict_proba(features_df)
        return proba[:, 1]

    def get_buy_signals(self, features_df, threshold=None):
        if threshold is None:
            threshold = self.buy_threshold
        return self.predict_proba(features_df) >= threshold

    def feature_importance(self):
        if self.model is None:
            return None
        importance = self.model.get_feature_importance()
        return dict(zip(self.feature_names, importance))

    def save_model(self, path):
        if self.model is None:
            return False
        self.model.save_model(path)
        print(f"Model saved to {path}")
        return True

    def load_model(self, path):
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        print(f"Model loaded from {path}")
        return True
