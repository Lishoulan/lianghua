import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "pip_libs"))


class AMSELossCatBoost:

    def __init__(self, omega=10.0, y_threshold=0.5):
        self.omega = omega
        self.y_threshold = y_threshold

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return False

    def _get_approx(self, approxes):
        approx = np.array(approxes[0])
        if len(approxes) == 2:
            approx = np.array(approxes[1]) - np.array(approxes[0])
        return approx

    def evaluate(self, approxes, target, weight):
        approx = self._get_approx(approxes)
        y = np.array(target)
        y_prob = 1.0 / (1.0 + np.exp(-approx))
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
        error = (y - y_prob) ** 2
        sign_wrong = ((y - self.y_threshold) * (y_prob - self.y_threshold)) < 0
        penalty = 1.0 + self.omega * sign_wrong.astype(float)
        loss = error * penalty
        if weight is not None:
            weight_arr = np.array(weight)
            return float(np.sum(loss * weight_arr)), float(np.sum(weight_arr))
        return float(np.sum(loss)), float(len(loss))

    def calc_ders_range(self, approxes, target, weight):
        approx = self._get_approx(approxes)
        y = np.array(target)
        y_prob = 1.0 / (1.0 + np.exp(-approx))
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
        sign_wrong = ((y - self.y_threshold) * (y_prob - self.y_threshold)) < 0
        penalty = 1.0 + self.omega * sign_wrong.astype(float)
        d_error = -2.0 * (y - y_prob)
        d_sigmoid = y_prob * (1.0 - y_prob)
        grad = d_error * penalty * d_sigmoid
        hess = (2.0 * d_sigmoid ** 2 * penalty +
                np.abs(d_error * penalty * d_sigmoid * (1.0 - 2.0 * y_prob)))
        hess = np.maximum(hess, 1e-6)
        result = []
        for i in range(len(y)):
            w = 1.0 if weight is None else weight[i]
            result.append((float(grad[i]) * w, float(hess[i]) * w))
        return result


try:
    import torch
    import torch.nn as nn

    class AMSELossPyTorch(nn.Module):

        def __init__(self, omega=10.0, y_threshold=0.5):
            super().__init__()
            self.omega = omega
            self.y_threshold = y_threshold

        def forward(self, logits, targets):
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
            error = (targets - probs) ** 2
            sign_wrong = ((targets - self.y_threshold) * (probs - self.y_threshold)) < 0
            penalty = 1.0 + self.omega * sign_wrong.float()
            loss = (error * penalty).mean()
            return loss

except ImportError:
    pass
