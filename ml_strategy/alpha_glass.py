import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn


class DifferentiableRankAndMask(nn.Module):

    def __init__(self, n_assets, temperature=1.0, top_k=3, min_weight=0.0, max_weight=0.25):
        super().__init__()
        self.n_assets = n_assets
        self.temperature = temperature
        self.top_k = min(top_k, n_assets)
        self.min_weight = min_weight
        self.max_weight = max_weight

    def forward(self, raw_scores):
        soft_ranks = self._soft_sort(raw_scores, self.temperature)
        mask = self._topk_mask(soft_ranks)
        weights = torch.softmax(raw_scores, dim=-1) * mask
        weights = torch.clamp(weights, self.min_weight, self.max_weight)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return weights

    def _soft_sort(self, scores, temperature):
        n = scores.shape[-1]
        scores_expanded = scores.unsqueeze(-2)
        indices = torch.arange(n, device=scores.device, dtype=torch.float32)
        indices_expanded = indices.unsqueeze(-1)
        diff = scores_expanded - indices_expanded
        soft_indicator = torch.sigmoid(diff / max(temperature, 0.01))
        ranks = soft_indicator.sum(dim=-1)
        return ranks

    def _topk_mask(self, ranks):
        n = ranks.shape[-1]
        topk_threshold = self.top_k + 0.5
        mask = torch.sigmoid((topk_threshold - ranks) * 10.0)
        return mask


class SharpeLoss(nn.Module):

    def __init__(self, risk_free_rate=0.0, annualize_factor=252, downside_only=False,
                 max_drawdown_weight=0.0):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.annualize_factor = annualize_factor
        self.downside_only = downside_only
        self.max_drawdown_weight = max_drawdown_weight

    def forward(self, portfolio_returns):
        if len(portfolio_returns) < 2:
            return torch.tensor(0.0, device=portfolio_returns.device, requires_grad=True)

        excess = portfolio_returns - self.risk_free_rate / self.annualize_factor
        mean_ret = excess.mean()

        if self.downside_only:
            downside = torch.clamp(excess, max=0.0)
            downside_std = torch.sqrt((downside ** 2).mean() + 1e-8)
            sharpe = mean_ret / downside_std
        else:
            std_ret = excess.std() + 1e-8
            sharpe = mean_ret / std_ret

        loss = -sharpe

        if self.max_drawdown_weight > 0:
            cum_returns = torch.cumsum(portfolio_returns, dim=0)
            running_max = torch.cummax(cum_returns, dim=0).values
            drawdowns = running_max - cum_returns
            max_dd = drawdowns.max()
            loss = loss + self.max_drawdown_weight * max_dd

        return loss


class AlphaGlassNet(nn.Module):

    def __init__(self, input_dim, n_assets, hidden_dim=32, top_k=3,
                 temperature=1.0, min_weight=0.0, max_weight=0.25,
                 use_batch_norm=True, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.n_assets = n_assets

        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.score_head = nn.Linear(hidden_dim // 2, n_assets)

        self.rank_and_mask = DifferentiableRankAndMask(
            n_assets=n_assets,
            temperature=temperature,
            top_k=top_k,
            min_weight=min_weight,
            max_weight=max_weight,
        )

    def forward(self, x):
        features = self.feature_net(x)
        raw_scores = self.score_head(features)
        weights = self.rank_and_mask(raw_scores)
        return weights, raw_scores

    def get_feature_importance(self, x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
            features = self.feature_net(x_tensor)
            grad_weights = self.score_head.weight
            importance = torch.abs(grad_weights).mean(dim=0)
            return importance.cpu().numpy()


class AlphaGlassTrainer:

    def __init__(self, input_dim, n_assets, hidden_dim=32, top_k=3,
                 temperature=1.0, lr=0.001, epochs=100, batch_size=64,
                 min_weight=0.0, max_weight=0.25,
                 sharpe_downside_only=False, max_drawdown_weight=0.5,
                 use_batch_norm=True, dropout=0.1):
        self.input_dim = input_dim
        self.n_assets = n_assets
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.temperature = temperature
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.sharpe_downside_only = sharpe_downside_only
        self.max_drawdown_weight = max_drawdown_weight
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.model = None
        self.best_val_loss = float('inf')

    def train(self, X_train, returns_train, X_val=None, returns_val=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        R_t = torch.tensor(returns_train, dtype=torch.float32).to(device)

        has_val = X_val is not None and returns_val is not None
        if has_val:
            X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
            R_v = torch.tensor(returns_val, dtype=torch.float32).to(device)

        self.model = AlphaGlassNet(
            input_dim=self.input_dim,
            n_assets=self.n_assets,
            hidden_dim=self.hidden_dim,
            top_k=self.top_k,
            temperature=self.temperature,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            use_batch_norm=self.use_batch_norm,
            dropout=self.dropout,
        ).to(device)

        sharpe_loss = SharpeLoss(
            downside_only=self.sharpe_downside_only,
            max_drawdown_weight=self.max_drawdown_weight,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        n_samples = len(X_t)
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                idx = perm[start:end]
                x_batch = X_t[idx]
                r_batch = R_t[idx]

                weights, _ = self.model(x_batch)
                portfolio_returns = (weights * r_batch).sum(dim=-1)
                loss = sharpe_loss(portfolio_returns)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if has_val and (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    w_val, _ = self.model(X_v)
                    val_port_ret = (w_val * R_v).sum(dim=-1)
                    val_loss = sharpe_loss(val_port_ret).item()

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(device)

        print(f"  AlphaGlass training done. Best val loss: {self.best_val_loss:.4f}")

    def predict_weights(self, X):
        if self.model is None:
            return np.ones(self.n_assets) / self.n_assets
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32).to(device)
            weights, scores = self.model(x)
            return weights.cpu().numpy(), scores.cpu().numpy()

    def summary(self):
        return (
            f"AlphaGlassTrainer:\n"
            f"  Input dim: {self.input_dim}\n"
            f"  N assets: {self.n_assets}\n"
            f"  Hidden dim: {self.hidden_dim}\n"
            f"  Top-K: {self.top_k}\n"
            f"  Temperature: {self.temperature}\n"
            f"  Sharpe downside-only: {self.sharpe_downside_only}\n"
            f"  Max drawdown weight: {self.max_drawdown_weight}\n"
            f"  Best val loss: {self.best_val_loss:.4f}"
        )
