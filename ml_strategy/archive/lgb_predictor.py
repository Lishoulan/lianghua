import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb


class LGBPredictor:

    FEATURE_COLS = [
        'price_zone',
        'j_zone',
        'k_pattern',
        'temd',
        'yellow_slope_sign',
        'yellow_slope_norm',
        'vol_ratio_20',
        'vol_ma5_ratio',
        'vol_change',
        'atr_ratio',
        'white_above_yellow',
        'white_yellow_dist',
        'macd_value',
        'dif_value',
        'macd_above_zero',
        'rsi_value',
        'j_raw',
        'j_min_5',
        'j_rising',
        'dist_pct',
        'return_1d',
        'return_5d',
        'price_position_20',
        'low_above_yellow',
    ]

    CATEGORICAL_FEATURES = ['price_zone', 'j_zone', 'k_pattern']

    def __init__(self, buy_threshold=0.55):
        self.buy_threshold = buy_threshold
        self.model = None
        self.feature_names = list(self.FEATURE_COLS)

    def prepare_features(self, df):
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        result = df[available].copy()
        result = result.dropna(subset=available)
        self.feature_names = available
        return result

    def train(self, features_df, labels, eval_features_df=None, eval_labels=None):
        categorical_features = [c for c in self.CATEGORICAL_FEATURES if c in features_df.columns]
        train_data = lgb.Dataset(
            features_df,
            label=labels,
            categorical_feature=categorical_features,
        )
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 30,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'verbose': -1,
            'n_jobs': 1,
            'seed': 42,
        }
        if eval_features_df is not None and eval_labels is not None:
            eval_data = lgb.Dataset(
                eval_features_df,
                label=eval_labels,
                categorical_feature=categorical_features,
            )
            callbacks = [
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ]
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[eval_data],
                callbacks=callbacks,
            )
        else:
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=300,
            )
        print(f"Best iteration: {self.model.best_iteration}, Best score: {self.model.best_score}")

    def predict_proba(self, features_df):
        if self.model is None:
            return np.full(len(features_df), 0.5)
        data = features_df.copy()
        for col in self.CATEGORICAL_FEATURES:
            if col in data.columns:
                data[col] = data[col].astype(int)
        return self.model.predict(data, num_iteration=self.model.best_iteration)

    def get_buy_signals(self, features_df, threshold=None):
        if threshold is None:
            threshold = self.buy_threshold
        return self.predict_proba(features_df) >= threshold

    def feature_importance(self):
        if self.model is None:
            return None
        importance = self.model.feature_importance(importance_type='split')
        return dict(zip(self.feature_names, importance))
