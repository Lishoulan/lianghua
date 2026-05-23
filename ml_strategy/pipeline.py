import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "pip_libs"))

from .hmm_model import MarketStateHMM
from .feature_engine import FeatureDiscretizer
from .triple_barrier import TripleBarrierLabeler
from .lgb_predictor import LGBPredictor


class InferencePipeline:

    def __init__(self, hmm_model=None, discretizer=None, lgb_model=None,
                 hmm_state2_threshold=0.65, lgb_buy_threshold=0.70):
        self.hmm_model = hmm_model or MarketStateHMM()
        self.discretizer = discretizer or FeatureDiscretizer()
        self.lgb_model = lgb_model or LGBPredictor(buy_threshold=lgb_buy_threshold)
        self.hmm_state2_threshold = hmm_state2_threshold
        self.lgb_buy_threshold = lgb_buy_threshold
        self.barrier_labeler = TripleBarrierLabeler()

    def step1_hmm_scan(self, market_df):
        if not self.hmm_model._fitted:
            print("HMM not fitted, training on market data...")
            self.hmm_model.fit(market_df)
        hmm_result = self.hmm_model.predict_proba(market_df)
        state2_mask = hmm_result['hmm_state2_prob'] >= self.hmm_state2_threshold
        print(f"HMM scan: {state2_mask.sum()}/{len(market_df)} days in State2 (golden pullback)")
        return hmm_result, state2_mask

    def step2_feature_transform(self, stock_df):
        return self.discretizer.transform(stock_df)

    def step3_predict(self, features_df):
        clean_features = self.lgb_model.prepare_features(features_df)
        if len(clean_features) == 0:
            return np.array([]), clean_features.index
        probs = self.lgb_model.predict_proba(clean_features)
        return probs, clean_features.index

    def run_daily(self, stock_df, market_hmm_result, date):
        if date not in stock_df.index:
            return None
        if date not in market_hmm_result.index:
            return None

        state2_prob = market_hmm_result.loc[date, 'hmm_state2_prob']
        if state2_prob < self.hmm_state2_threshold:
            return {'buy': False, 'reason': f'HMM state2 prob {state2_prob:.2f} < {self.hmm_state2_threshold}'}

        featured_df = self.step2_feature_transform(stock_df)
        if date not in featured_df.index:
            return {'buy': False, 'reason': 'Feature transform failed'}

        row_features = featured_df.loc[[date]]
        clean_features = self.lgb_model.prepare_features(row_features)
        if len(clean_features) == 0:
            return {'buy': False, 'reason': 'NaN in features'}

        prob = self.lgb_model.predict_proba(clean_features)[0]
        buy = prob >= self.lgb_buy_threshold

        result = {
            'buy': buy,
            'hmm_state2_prob': float(state2_prob),
            'lgb_prob': float(prob),
            'reason': 'PASS' if buy else f'LGB prob {prob:.2f} < {self.lgb_buy_threshold}'
        }

        if buy and 'ATR14' in stock_df.columns:
            atr_val = stock_df.loc[date, 'ATR14']
            close_val = stock_df.loc[date, 'Close']
            if not pd.isna(atr_val) and not pd.isna(close_val):
                result['dynamic_stop_loss'] = float(close_val - 0.8 * atr_val)
                result['dynamic_take_profit'] = float(close_val + 1.5 * atr_val)

        return result

    def build_training_data(self, stock_dfs, market_hmm_result, eval_start,
                            train_start='2021-01-01'):
        all_samples = []
        labeler = TripleBarrierLabeler()

        for ts_code, stock_df in stock_dfs.items():
            try:
                featured_df = self.step2_feature_transform(stock_df)
                hmm_result = market_hmm_result.reindex(stock_df.index).fillna(0)
                state2_mask = hmm_result['hmm_state2_prob'] >= self.hmm_state2_threshold

                train_mask = (stock_df.index >= pd.Timestamp(train_start)) & \
                             (stock_df.index < eval_start)
                combined_mask = state2_mask & train_mask

                candidate_indices = [i for i in range(len(stock_df)) if combined_mask.iloc[i]]

                if len(candidate_indices) == 0:
                    continue

                atr14 = stock_df['ATR14'] if 'ATR14' in stock_df.columns else pd.Series(np.nan, index=stock_df.index)
                labels = labeler.label_all(stock_df, candidate_indices, atr14)

                for lab in labels:
                    idx = lab['entry_idx']
                    if idx >= len(featured_df):
                        continue
                    row = featured_df.iloc[idx]
                    feature_cols = ['price_zone', 'j_zone', 'k_pattern', 'temd',
                                   'yellow_slope_sign', 'vol_ratio_20', 'atr_ratio',
                                   'white_above_yellow', 'macd_value', 'rsi_value']
                    if any(pd.isna(row.get(c)) for c in feature_cols):
                        continue
                    sample = {c: row[c] for c in feature_cols}
                    sample['label'] = lab['label']
                    sample['ts_code'] = ts_code
                    sample['date'] = stock_df.index[idx]
                    all_samples.append(sample)
            except Exception as e:
                continue

        if len(all_samples) == 0:
            return None, None

        samples_df = pd.DataFrame(all_samples)
        feature_cols = ['price_zone', 'j_zone', 'k_pattern', 'temd',
                        'yellow_slope_sign', 'vol_ratio_20', 'atr_ratio',
                        'white_above_yellow', 'macd_value', 'rsi_value']
        X = samples_df[feature_cols]
        y = samples_df['label'].values

        print(f"Training data: {len(X)} samples, positive rate: {y.mean():.2%}")
        return X, y

    def train_lgb(self, X_train, y_train, X_val=None, y_val=None):
        self.lgb_model.train(X_train, y_train, X_val, y_val)
        importance = self.lgb_model.feature_importance()
        if importance:
            print("\nFeature Importance:")
            for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
                print(f"  {name}: {imp}")
