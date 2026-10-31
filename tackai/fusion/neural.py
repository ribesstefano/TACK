"""Small molecular/context interaction network on the frozen fusion inputs.

Torch is an optional dependency and this module is not imported by the default
CPU tree inference path. Missing endpoint supervision is masked, never imputed.
All learned normalization belongs to a fitted artifact, not the context cache.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TASK_ORDER = ("dmax", "pdc50", "activity")


class FusionMLP(nn.Module):
    """Project fixed molecule/context vectors and learn their interactions."""

    def __init__(self, n_features, molecular_width, width=128, dropout=.1):
        super().__init__()
        self.molecular_width = molecular_width
        self.molecule = nn.Sequential(nn.Linear(molecular_width, width), nn.LayerNorm(width),
                                      nn.SiLU(), nn.Dropout(dropout))
        self.context = nn.Sequential(nn.Linear(n_features - molecular_width, width),
                                     nn.LayerNorm(width), nn.SiLU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(nn.Linear(width * 3, width), nn.SiLU(), nn.Dropout(dropout),
                                    nn.Linear(width, width), nn.SiLU())
        self.head = nn.Linear(width, 3)

    def forward(self, X):
        mol = self.molecule(X[:, :self.molecular_width])
        ctx = self.context(X[:, self.molecular_width:])
        outputs = self.head(self.fusion(torch.cat((mol, ctx, mol * ctx), dim=1)))
        return torch.stack((outputs[:, 0].sigmoid(), outputs[:, 1], outputs[:, 2]), dim=1)


def masked_loss(prediction, targets, scales, weights=(1., 1., .5), bounds=None):
    """Balanced endpoint loss with Dmax bounded and pDC50 already standardized.

    The activity weight is smaller because its labels derive from the endpoints.
    Missing labels contribute neither loss nor denominator; a wholly unlabelled
    batch gives a differentiable zero rather than fabricated zero-valued targets.
    """
    total = prediction.sum() * 0.
    active_weight = 0.
    for head, weight in enumerate(weights):
        interval = bounds is not None and head < 2
        mask = ((~torch.isnan(bounds[:, head]).any(dim=1)) &
                torch.isfinite(bounds[:, head]).any(dim=1) if interval
                else torch.isfinite(targets[:, head]))
        if not mask.any() or weight == 0:
            continue
        if interval:
            lower, upper = bounds[mask, head, 0], bounds[mask, head, 1]
            distance = F.relu(lower - prediction[mask, head]) + F.relu(prediction[mask, head] - upper)
            value = distance.square().mean()
            if head == 0:
                value = value / scales[0] ** 2
        elif head == 2:
            value = F.binary_cross_entropy_with_logits(prediction[mask, head], targets[mask, head])
        elif head == 0:
            value = F.mse_loss(prediction[mask, head], targets[mask, head]) / scales[0] ** 2
        else:
            value = F.mse_loss(prediction[mask, head], targets[mask, head])
        total = total + weight * value
        active_weight += weight
    return total / max(active_weight, 1e-12)


class NeuralPredictor:
    """Fitted point predictor; uncertainty is supplied by an external ensemble.

    Fit is CPU-capable for small pilot datasets. A CUDA device can be selected
    only within an appropriate Slurm allocation. Validation rows are used only
    for early stopping, never for scalers or target normalization.
    """

    def __init__(self, molecular_width, width=128, dropout=.1, learning_rate=.001,
                 weight_decay=.001, batch_size=256, seed=42, device="cpu",
                 task_weights=(1., 1., .5)):
        self.config = dict(molecular_width=molecular_width, width=width, dropout=dropout,
                           learning_rate=learning_rate, weight_decay=weight_decay,
                           batch_size=batch_size, seed=seed, task_weights=list(task_weights))
        self.device = device
        self.model = None

    def _inputs(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.mean_) or not np.isfinite(X).all():
            raise ValueError("Invalid neural input shape or nonfinite features")
        normalized = np.clip((X - self.mean_) / self.scale_, -10., 10.).astype(np.float32)
        return torch.as_tensor(normalized, device=self.device)

    def _targets(self, y):
        y = np.asarray(y, dtype=np.float32).copy()
        if y.ndim != 2 or y.shape[1] != 3:
            raise ValueError("Neural targets must have three columns in TASK_ORDER")
        y[:, 1] = (y[:, 1] - self.pdc50_mean_) / self.target_scales_[1]
        return torch.as_tensor(y, device=self.device)

    def _bounds(self, bounds):
        if bounds is None:
            return None
        bounds = np.asarray(bounds, dtype=np.float32).copy()
        if bounds.ndim != 3 or bounds.shape[1:] != (2, 2):
            raise ValueError("Regression bounds must have shape (n, 2 endpoints, 2 bounds)")
        bounds[:, 1] = (bounds[:, 1] - self.pdc50_mean_) / self.target_scales_[1]
        return torch.as_tensor(bounds, device=self.device)

    def fit(self, X, y, *, epochs=100, validation=None, patience=12, bounds=None):
        """Fit on explicit training arrays, optionally selecting epochs on validation."""
        X, y = np.asarray(X), np.asarray(y, dtype=float)
        if y.shape != (len(X), 3):
            raise ValueError("Need aligned, partly observed three-task targets")
        keep = np.isfinite(y).any(axis=1)
        if bounds is not None:
            bounds = np.asarray(bounds, float)
            if bounds.shape != (len(X), 2, 2):
                raise ValueError("Bounds must align with rows and two regression endpoints")
            keep |= np.isfinite(bounds).any(axis=(1, 2))
            bounds = bounds[keep]
        if not keep.any():
            raise ValueError("No observed targets or informative bounds")
        X, y = X[keep], y[keep]
        if not np.isfinite(X).all() or epochs < 1:
            raise ValueError("Need finite inputs and at least one epoch")
        torch.manual_seed(self.config["seed"])
        self.mean_ = X.mean(axis=0, dtype=np.float64)
        self.scale_ = X.std(axis=0, dtype=np.float64)
        self.scale_[self.scale_ < 1e-8] = 1.
        self.target_scales_ = np.array([
            max(float(np.std(y[np.isfinite(y[:, h]), h])), .05)
            if np.isfinite(y[:, h]).any() else 1. for h in (0, 1)], dtype=np.float32)
        self.pdc50_mean_ = float(np.nanmean(y[:, 1])) if np.isfinite(y[:, 1]).any() else 0.
        self.model = FusionMLP(X.shape[1], self.config["molecular_width"],
                               self.config["width"], self.config["dropout"]).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"],
                                     weight_decay=self.config["weight_decay"])
        Xt, yt, bt = self._inputs(X), self._targets(y), self._bounds(bounds)
        if validation is not None:
            Xv, yv = self._inputs(validation[0]), self._targets(validation[1])
            bv = self._bounds(validation[2]) if len(validation) == 3 else None
            if not torch.isfinite(yv).any() and (bv is None or not torch.isfinite(bv).any()):
                raise ValueError("Early-stopping partition has no observed targets")
        scales = torch.as_tensor(self.target_scales_, device=self.device)
        best_loss, best_state, stale = np.inf, None, 0
        self.epochs_ = epochs
        self.history_ = []
        generator = torch.Generator(device=self.device).manual_seed(self.config["seed"])
        for epoch in range(1, epochs + 1):
            self.model.train()
            permutation = torch.randperm(len(Xt), generator=generator, device=self.device)
            for batch in permutation.split(self.config["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                loss = masked_loss(self.model(Xt[batch]), yt[batch], scales,
                                   self.config["task_weights"], None if bt is None else bt[batch])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite neural training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                optimizer.step()
            if validation is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = float(masked_loss(self.model(Xv), yv, scales,
                                                self.config["task_weights"], bv).cpu())
                self.history_.append(val_loss)
                if not np.isfinite(val_loss):
                    raise ValueError("Nonfinite neural validation loss")
                if val_loss < best_loss - 1e-5:
                    best_loss, stale, self.epochs_ = val_loss, 0, epoch
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in self.model.state_dict().items()}
                else:
                    stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, X):
        """Return aligned point estimates; no uncalibrated uncertainty claims."""
        if self.model is None:
            raise ValueError("Model is not fitted")
        inputs = self._inputs(X)
        with torch.no_grad():
            chunks = [self.model(batch).cpu() for batch in inputs.split(self.config["batch_size"])]
        if not chunks:
            return {task: np.empty(0) for task in TASK_ORDER}
        out = torch.cat(chunks).numpy()
        out[:, 1] = out[:, 1] * self.target_scales_[1] + self.pdc50_mean_
        out[:, 2] = torch.sigmoid(torch.from_numpy(out[:, 2])).numpy()
        return {task: out[:, i] for i, task in enumerate(TASK_ORDER)}

    def save(self, directory):
        """Save state tensors and explicit scaler/config arrays without estimator pickle."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        torch.save(self.model.state_dict(), directory / "weights.pt")
        np.savez(directory / "scalers.npz", mean=self.mean_, scale=self.scale_,
                 target_scales=self.target_scales_, pdc50_mean=self.pdc50_mean_)
        (directory / "config.json").write_text(json.dumps(
            {"config": self.config, "n_features": len(self.mean_), "epochs": self.epochs_}, indent=2))

    @classmethod
    def load(cls, directory, device="cpu"):
        """Recreate a saved model with restricted tensor-only torch deserialization."""
        directory = Path(directory)
        meta = json.loads((directory / "config.json").read_text())
        obj = cls(**meta["config"], device=device)
        with np.load(directory / "scalers.npz", allow_pickle=False) as data:
            obj.mean_, obj.scale_ = data["mean"], data["scale"]
            obj.target_scales_, obj.pdc50_mean_ = data["target_scales"], float(data["pdc50_mean"])
        obj.model = FusionMLP(meta["n_features"], obj.config["molecular_width"],
                              obj.config["width"], obj.config["dropout"]).to(device)
        obj.model.load_state_dict(torch.load(directory / "weights.pt", map_location=device, weights_only=True))
        obj.epochs_ = meta["epochs"]
        obj.model.eval()
        return obj
