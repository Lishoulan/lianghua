import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "pip_libs"))

import torch
import torch.nn as nn


class KANPredictor(nn.Module):

    FEATURE_COLS = [
        'price_zone', 'j_zone', 'k_pattern', 'temd',
        'yellow_slope_sign', 'yellow_slope_norm', 'vol_ratio_20',
        'vol_ma5_ratio', 'vol_change', 'atr_ratio',
        'white_above_yellow', 'white_yellow_dist', 'macd_value',
        'dif_value', 'macd_above_zero', 'rsi_value',
        'j_raw', 'j_min_5', 'j_rising', 'dist_pct',
        'return_1d', 'return_5d', 'price_position_20', 'low_above_yellow',
        'index_yellow_slope', 'index_state2_prob', 'industry_oversold',
    ]

    def __init__(self, input_dim=27, hidden_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return probs


class KANTrainer:

    def __init__(self, input_dim=27, hidden_dim=16, lr=0.005, epochs=200,
                 batch_size=256, weight_decay=1e-4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.model = None
        self.feature_means = None
        self.feature_stds = None

    def _normalize(self, X, fit=False):
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)
        if fit:
            self.feature_means = X.mean(axis=0)
            self.feature_stds = X.std(axis=0) + 1e-8
        X = (X - self.feature_means) / self.feature_stds
        return X

    def train(self, X_train, y_train, X_val=None, y_val=None, sample_weights=None):
        X_train_np = self._normalize(X_train, fit=True)
        y_train_np = np.asarray(y_train, dtype=np.float32)

        X_t = torch.tensor(X_train_np, dtype=torch.float32)
        y_t = torch.tensor(y_train_np, dtype=torch.float32)

        self.model = KANPredictor(
            input_dim=X_train_np.shape[1],
            hidden_dim=self.hidden_dim,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        X_val_t = None
        y_val_t = None
        if X_val is not None:
            X_val_np = self._normalize(X_val, fit=False)
            X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
            y_val_t = torch.tensor(np.asarray(y_val, dtype=np.float32))

        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        best_state = None

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            if X_val_t is not None and (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = nn.functional.binary_cross_entropy_with_logits(
                        val_logits, y_val_t
                    ).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        print(f"  KAN (MLP-SiLU) training done. Best val loss: {best_val_loss:.4f}")
        return self.model

    def predict_proba(self, X):
        if self.model is None:
            return np.full(len(X), 0.5)
        X_np = self._normalize(X, fit=False)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        probs = self.model.predict_proba(X_t)
        return probs.cpu().numpy()
