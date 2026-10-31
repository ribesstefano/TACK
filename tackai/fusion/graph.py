"""Directed message-passing predictor on molecules plus frozen fusion context.

This research-only adapter replaces the Morgan fingerprint with a Chemprop
directed-bond graph embedding. RDKit descriptors and all biological context
columns are passed through unchanged. The core package does not import this
module, so existing fusion encoders and artifacts retain their input schema.
"""
from __future__ import annotations

import json
import importlib.metadata
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F

from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import BondMessagePassing, MeanAggregation

TASK_ORDER = ("dmax", "pdc50", "activity")
FEATURIZER_CONTRACT = "Chemprop SimpleMoleculeMolGraphFeaturizer default atom/bond features"


def _mol_graphs(smiles):
    """Convert isomeric SMILES to Chemprop graphs, preserving row order."""
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    graphs = []
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError(f"Invalid molecule SMILES at row {index}")
        graphs.append(featurizer(mol))
    return graphs, featurizer.atom_fdim, featurizer.bond_fdim


class DirectedGraphNetwork(nn.Module):
    """Chemprop D-MPNN followed by context interaction and masked task heads."""

    def __init__(self, atom_dim, bond_dim, side_dim, width=128, depth=3, dropout=.1):
        super().__init__()
        self.message_passing = BondMessagePassing(
            d_v=atom_dim, d_e=bond_dim, d_h=width, depth=depth,
            dropout=dropout, activation="relu",
        )
        self.aggregation = MeanAggregation()
        self.side = nn.Sequential(nn.Linear(side_dim, width), nn.LayerNorm(width),
                                  nn.SiLU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(nn.Linear(width * 3, width), nn.LayerNorm(width),
                                    nn.SiLU(), nn.Dropout(dropout),
                                    nn.Linear(width, width), nn.SiLU())
        self.head = nn.Linear(width, 3)

    def forward(self, graphs, side):
        batch = BatchMolGraph(graphs)
        batch.to(side.device)
        atom = self.message_passing(batch)
        molecule = self.aggregation(atom, batch.batch)
        context = self.side(side)
        output = self.head(self.fusion(torch.cat((molecule, context, molecule * context), dim=1)))
        return torch.stack((output[:, 0].sigmoid(), output[:, 1], output[:, 2]), dim=1)


def masked_graph_loss(prediction, targets, scales, weights=(1., 1., .5),
                      denominators=None, training_size=None):
    """Average only over labeled endpoints, with train-only target scaling.

    ``denominators`` is the full training-partition label count per head. For
    minibatch fitting, ``training_size`` rescales each global-denominator batch
    contribution by dataset size / batch size to keep optimizer-step magnitudes
    near the full masked loss. Validation omits both and uses exact full-set means.
    """
    counts = ([int(torch.isfinite(targets[:, head]).sum()) for head in range(3)]
              if denominators is None else list(denominators))
    if len(counts) != 3 or any(count < 0 for count in counts):
        raise ValueError("Need three nonnegative endpoint denominators")
    active_weight = sum(weight for weight, count in zip(weights, counts) if weight and count)
    total = prediction.sum() * 0.
    batch_scale = (training_size / len(targets)
                   if denominators is not None and training_size is not None else 1.)
    for head, weight in enumerate(weights):
        mask = torch.isfinite(targets[:, head])
        if not mask.any() or weight == 0 or counts[head] == 0:
            continue
        if head == 2:
            value = F.binary_cross_entropy_with_logits(
                prediction[mask, head], targets[mask, head], reduction="sum") / counts[head]
        else:
            value = F.mse_loss(prediction[mask, head], targets[mask, head], reduction="sum") / counts[head]
            if head == 0:
                value = value / scales[0].square()
        total = total + weight * value
    return batch_scale * total / max(active_weight, 1e-12)


class GraphPredictor:
    """Fit and score new isomeric SMILES with frozen non-fingerprint features.

    ``side_features`` is the baseline design matrix with its fingerprint columns
    removed: descriptors followed by the existing E3, cell, POI, assay and time
    vectors. Input and target normalization are fitted on the fit partition only.
    Artifacts contain JSON, NPZ and tensor state dicts; estimator pickle is never used.
    """

    def __init__(self, side_dim=None, width=128, depth=3, dropout=.1,
                 learning_rate=.001, weight_decay=.001, batch_size=64, seed=42,
                 device="cpu", task_weights=(1., 1., .5)):
        self.config = dict(side_dim=side_dim, width=width, depth=depth, dropout=dropout,
                           learning_rate=learning_rate, weight_decay=weight_decay,
                           batch_size=batch_size, seed=seed,
                           task_weights=list(task_weights))
        self.device = device
        self.model = None

    def _side(self, features):
        features = np.asarray(features, dtype=np.float64)
        if (features.ndim != 2 or features.shape[1] != self.config["side_dim"] or
                not np.isfinite(features).all()):
            raise ValueError("Invalid frozen descriptor/context feature shape or values")
        values = np.clip((features - self.side_mean_) / self.side_scale_, -10., 10.)
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _targets(self, targets):
        values = np.asarray(targets, dtype=np.float32).copy()
        if values.ndim != 2 or values.shape[1] != len(TASK_ORDER):
            raise ValueError("Targets must have three columns in TASK_ORDER")
        values[:, 1] = (values[:, 1] - self.pdc50_mean_) / self.target_scales_[1]
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def fit(self, smiles, side_features, targets, *, epochs=80, validation=None,
            patience=10):
        """Fit with optional group-disjoint validation used only for early stopping."""
        side_features = np.asarray(side_features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if side_features.ndim != 2 or targets.shape != (len(side_features), 3):
            raise ValueError("SMILES, side features and targets must have aligned rows")
        if len(smiles) != len(side_features) or not np.isfinite(side_features).all():
            raise ValueError("SMILES and finite side features must align")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        keep = np.isfinite(targets).any(axis=1)
        if not keep.any():
            raise ValueError("No observed task targets")
        smiles = [str(smiles[i]) for i in np.flatnonzero(keep)]
        side_features, targets = side_features[keep], targets[keep]
        graphs, atom_dim, bond_dim = _mol_graphs(smiles)
        self.atom_dim_, self.bond_dim_ = atom_dim, bond_dim
        torch.manual_seed(self.config["seed"])
        self.side_mean_ = side_features.mean(axis=0)
        self.side_scale_ = side_features.std(axis=0)
        self.side_scale_[self.side_scale_ < 1e-8] = 1.
        self.target_scales_ = np.array([
            max(float(np.nanstd(targets[:, head])), .05)
            if np.isfinite(targets[:, head]).any() else 1. for head in (0, 1)
        ], dtype=np.float32)
        self.pdc50_mean_ = float(np.nanmean(targets[:, 1])) if np.isfinite(targets[:, 1]).any() else 0.
        self.config["side_dim"] = side_features.shape[1]
        self.model = DirectedGraphNetwork(atom_dim, bond_dim, self.config["side_dim"],
                                          self.config["width"], self.config["depth"],
                                          self.config["dropout"]).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.config["learning_rate"],
                                      weight_decay=self.config["weight_decay"])
        side = self._side(side_features)
        target = self._targets(targets)
        validation_data = None
        if validation is not None:
            val_smiles, val_side, val_targets = validation[:3]
            val_graphs, _, _ = _mol_graphs(val_smiles)
            val_side = self._side(val_side)
            val_targets = self._targets(val_targets)
            if not torch.isfinite(val_targets).any():
                raise ValueError("Early-stopping partition has no observed labels")
            validation_data = val_graphs, val_side, val_targets
        scale = torch.as_tensor(self.target_scales_, device=self.device)
        denominators = torch.isfinite(target).sum(dim=0).tolist()
        best, best_state, stale = np.inf, None, 0
        self.epochs_ = epochs
        self.history_ = []
        generator = torch.Generator(device=self.device).manual_seed(self.config["seed"])
        for epoch in range(1, epochs + 1):
            self.model.train()
            permutation = torch.randperm(len(graphs), generator=generator, device=self.device)
            for indices in permutation.split(self.config["batch_size"]):
                rows = indices.tolist()
                optimizer.zero_grad(set_to_none=True)
                loss = masked_graph_loss(
                    self.model([graphs[i] for i in rows], side[indices]), target[indices],
                    scale, self.config["task_weights"], denominators,
                    training_size=len(graphs))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite graph training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                optimizer.step()
            if validation_data is not None:
                self.model.eval()
                with torch.no_grad():
                    vg, vs, vy = validation_data
                    outputs = []
                    for start in range(0, len(vg), self.config["batch_size"]):
                        stop = start + self.config["batch_size"]
                        outputs.append(self.model(vg[start:stop], vs[start:stop]))
                    val_prediction = torch.cat(outputs)
                    val_loss = float(masked_graph_loss(
                        val_prediction, vy, scale, self.config["task_weights"]).cpu())
                self.history_.append(val_loss)
                if not np.isfinite(val_loss):
                    raise ValueError("Nonfinite graph validation loss")
                if val_loss < best - 1e-5:
                    best, stale, self.epochs_ = val_loss, 0, epoch
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, smiles, side_features):
        """Score new molecules while preserving SMILES and feature row order."""
        if self.model is None:
            raise ValueError("Model is not fitted")
        if len(smiles) != len(side_features):
            raise ValueError("SMILES and feature rows must align")
        graphs, _, _ = _mol_graphs(smiles)
        side = self._side(side_features)
        chunks = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(graphs), self.config["batch_size"]):
                stop = start + self.config["batch_size"]
                chunks.append(self.model(graphs[start:stop], side[start:stop]).cpu())
        if not chunks:
            return {task: np.empty(0) for task in TASK_ORDER}
        values = torch.cat(chunks).numpy()
        values[:, 1] = values[:, 1] * self.target_scales_[1] + self.pdc50_mean_
        values[:, 2] = torch.sigmoid(torch.from_numpy(values[:, 2])).numpy()
        return {task: values[:, i] for i, task in enumerate(TASK_ORDER)}

    def save(self, directory, *, metadata=None):
        """Persist safe tensor and numeric artifacts without Python pickle."""
        if self.model is None:
            raise ValueError("Model is not fitted")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        torch.save(self.model.state_dict(), directory / "weights.pt")
        np.savez(directory / "scalers.npz", side_mean=self.side_mean_,
                 side_scale=self.side_scale_, target_scales=self.target_scales_,
                 pdc50_mean=self.pdc50_mean_)
        info = {"format_version": 1, "config": self.config, "epochs": self.epochs_,
                "atom_dim": self.atom_dim_, "bond_dim": self.bond_dim_,
                "tasks": list(TASK_ORDER), "featurizer_contract": FEATURIZER_CONTRACT,
                "versions": {name: importlib.metadata.version(name)
                             for name in ("chemprop", "rdkit")},
                "training_loss_weighting":
                    "global per-head mask denominator with batch-size correction",
                "metadata": metadata or {}}
        (directory / "config.json").write_text(json.dumps(info, indent=2, allow_nan=False))

    @classmethod
    def load(cls, directory, device="cpu"):
        """Load a graph model with tensor-only deserialization."""
        directory = Path(directory)
        info = json.loads((directory / "config.json").read_text())
        if info.get("format_version") != 1:
            raise ValueError("Unsupported graph artifact format")
        if tuple(info.get("tasks", ())) != TASK_ORDER:
            raise ValueError("Graph artifact task order is incompatible")
        if info.get("featurizer_contract") != FEATURIZER_CONTRACT:
            raise ValueError("Graph artifact molecule featurizer contract is incompatible")
        installed_versions = {name: importlib.metadata.version(name)
                              for name in ("chemprop", "rdkit")}
        if info.get("versions") != installed_versions:
            raise ValueError("Graph artifact Chemprop/RDKit versions differ from this environment")
        obj = cls(**info["config"], device=device)
        with np.load(directory / "scalers.npz", allow_pickle=False) as data:
            obj.side_mean_, obj.side_scale_ = data["side_mean"], data["side_scale"]
            obj.target_scales_, obj.pdc50_mean_ = data["target_scales"], float(data["pdc50_mean"])
        obj.model = DirectedGraphNetwork(info["atom_dim"], info["bond_dim"],
                                         obj.config["side_dim"], obj.config["width"],
                                         obj.config["depth"], obj.config["dropout"]).to(device)
        state = torch.load(directory / "weights.pt", map_location=device, weights_only=True)
        obj.model.load_state_dict(state)
        obj.atom_dim_, obj.bond_dim_ = info["atom_dim"], info["bond_dim"]
        obj.epochs_ = info["epochs"]
        obj.model.eval()
        return obj
