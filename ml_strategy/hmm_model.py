import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class MarketStateHMM:

    def __init__(self, n_states=3, n_init=5, random_state=42):
        self.n_states = n_states
        self.n_init = n_init
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.state_mapping = {}
        self._fitted = False

    def _build_observations(self, df):
        price_position = (df['Close'] - df['yellow_line']) / df['yellow_line']
        atr_ratio = df['ATR14'] / df['Close']
        vol_ratio = df['Volume'] / df['Vol_MA20']
        obs = pd.DataFrame({
            'price_position': price_position,
            'atr_ratio': atr_ratio,
            'vol_ratio': vol_ratio
        })
        return obs

    def _map_states(self, model, X_scaled):
        means = model.means_
        state_scores = []
        for s in range(self.n_states):
            price_pos_mean = means[s, 0]
            atr_mean = means[s, 1]
            vol_mean = means[s, 2]
            state_scores.append({
                'raw_state': s,
                'price_position': price_pos_mean,
                'atr_ratio': atr_mean,
                'vol_ratio': vol_mean
            })

        pullback_candidates = []
        uptrend_candidates = []
        breakdown_candidates = []

        for sc in state_scores:
            pp = sc['price_position']
            vr = sc['vol_ratio']
            ar = sc['atr_ratio']

            if abs(pp) < 0.5 and vr < 0 and ar < 0:
                pullback_candidates.append(sc)
            elif pp > 0:
                uptrend_candidates.append(sc)
            else:
                breakdown_candidates.append(sc)

        if not pullback_candidates:
            pullback_candidates = sorted(state_scores, key=lambda x: abs(x['price_position']))[:1]

        if not uptrend_candidates:
            uptrend_candidates = sorted(state_scores, key=lambda x: -x['price_position'])[:1]

        if not breakdown_candidates:
            breakdown_candidates = sorted(state_scores, key=lambda x: x['price_position'])[:1]

        pullback_candidates = [c for c in pullback_candidates if c['raw_state'] not in [u['raw_state'] for u in uptrend_candidates]]
        if not pullback_candidates:
            pullback_candidates = sorted(state_scores, key=lambda x: abs(x['price_position']))[:1]

        breakdown_candidates = [c for c in breakdown_candidates if c['raw_state'] not in [u['raw_state'] for u in uptrend_candidates] + [p['raw_state'] for p in pullback_candidates]]
        if not breakdown_candidates:
            remaining = [s for s in state_scores if s['raw_state'] not in [u['raw_state'] for u in uptrend_candidates] + [p['raw_state'] for p in pullback_candidates]]
            breakdown_candidates = remaining if remaining else sorted(state_scores, key=lambda x: x['price_position'])[:1]

        mapping = {}
        for c in uptrend_candidates:
            mapping[c['raw_state']] = 1
        for c in pullback_candidates:
            mapping[c['raw_state']] = 2
        for c in breakdown_candidates:
            mapping[c['raw_state']] = 3

        return mapping

    def fit(self, df):
        obs = self._build_observations(df)
        valid_mask = obs.notna().all(axis=1) & np.isfinite(obs).all(axis=1)
        obs_valid = obs[valid_mask]

        if len(obs_valid) < self.n_states * 10:
            self._fitted = False
            return self

        X = obs_valid.values.astype(np.float64)
        X_scaled = self.scaler.fit_transform(X)

        best_model = None
        best_ll = -np.inf

        rng = np.random.RandomState(self.random_state)
        for i in range(self.n_init):
            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type='full',
                    n_iter=200,
                    random_state=rng.randint(0, 100000),
                    tol=1e-4
                )
                model.fit(X_scaled)
                ll = model.score(X_scaled)
                if ll > best_ll:
                    best_ll = ll
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            self._fitted = False
            return self

        self.model = best_model
        self.state_mapping = self._map_states(best_model, X_scaled)
        self._fitted = True
        return self

    def predict_proba(self, df):
        result = pd.DataFrame(index=df.index)
        result['hmm_state1_prob'] = 0.0
        result['hmm_state2_prob'] = 0.0
        result['hmm_state3_prob'] = 0.0
        result['hmm_current_state'] = 0

        if not self._fitted or self.model is None:
            return result

        obs = self._build_observations(df)
        valid_mask = obs.notna().all(axis=1) & np.isfinite(obs).all(axis=1)

        if valid_mask.sum() < self.n_states:
            return result

        obs_valid = obs[valid_mask]
        X = obs_valid.values.astype(np.float64)
        X_scaled = self.scaler.transform(X)

        try:
            posteriors = self.model.predict_proba(X_scaled)
            raw_states = self.model.predict(X_scaled)
        except Exception:
            return result

        mapped_posteriors = np.zeros((len(X_scaled), 3))

        for raw_s, sem_s in self.state_mapping.items():
            mapped_posteriors[:, sem_s - 1] = posteriors[:, raw_s]

        mapped_states = np.array([self.state_mapping.get(int(rs), 0) for rs in raw_states])

        result.loc[valid_mask, 'hmm_state1_prob'] = mapped_posteriors[:, 0]
        result.loc[valid_mask, 'hmm_state2_prob'] = mapped_posteriors[:, 1]
        result.loc[valid_mask, 'hmm_state3_prob'] = mapped_posteriors[:, 2]
        result.loc[valid_mask, 'hmm_current_state'] = mapped_states

        return result
