""" TACK Model with per-label losses and weighted loss aggregation. """
import math
import logging
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, LambdaLR
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
from lightning_uq_box.uq_methods.utils import default_regression_metrics
from lightning_uq_box.uq_methods.loss_functions import DERLoss, NLL
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
)
from torchmetrics import MetricCollection

from tackai.models.multi_emb_model import MultiEmbeddingsRegressionModel
from tackai.models.text_emb_model import BERTModel
from tackai.models.mlp_model import MLPModel
from tackai.models.losses import (
    MSLELoss,
    DERLossWithLogReg,
    NLLWithLogReg,
)

def classification_metrics(prefix: str):
    """ Return a set of default classification metrics."""
    return MetricCollection(
        {
            "Accuracy": BinaryAccuracy(),
            "F1-Score": BinaryF1Score(),
            "Precision": BinaryPrecision(),
            "Recall": BinaryRecall(),
            "ROC-AUC": BinaryAUROC(),
        },
        prefix=prefix,
    )

class TACKModel(pl.LightningModule):

    def __init__(
            self,
            label_names: Optional[List[str]] = None,
            label_weights: Optional[Dict[str, float]] = None,
            der_loss_coeffs: Optional[Dict[str, float]] = None,
            default_der_loss_coeff: float = 0.01,
            learning_rate: float = 1e-3,
            warmup_ratio: float = 0.05,
            num_cycles: int = 1,
            model_type: Literal["custom", "bert", "mlp"] = "custom",
            task_type: Literal["der", "mve", "point", "msle", "bin"] = "der",
            lr_scheduler_type: Literal["linear", "cosine", "reduce_on_plateau"] = "reduce_on_plateau",
            use_log_reg: bool = False,  # Enable log regularization
            log_reg_alpha: float = 5.0,  # Weight for log term
            log_reg_epsilon: float = 1e-9,  # Small constant for log
            **model_kwargs: Dict[str, int],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_ratio = warmup_ratio
        self.num_cycles = num_cycles
        self.task_type = task_type
        self.model_type = model_type

        # Infer n_targets from label_names if not explicitly provided
        if label_names is not None and "n_targets" not in model_kwargs:
            model_kwargs["n_targets"] = len(label_names)

        # Update the regression type in model kwargs
        model_kwargs["task_type"] = task_type

        if model_type == "bert":
            self.model = BERTModel(**model_kwargs)
        elif model_type == "mlp":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = MultiEmbeddingsRegressionModel(**model_kwargs)

        self.label_names = label_names if label_names is not None else [f"label_{i}" for i in range(model_kwargs.get("n_targets", 1))]

        # Create explicit mapping from label names to output indices
        self.label_to_idx = {name: i for i, name in enumerate(self.label_names)}

        # Set up label weights - default to equal weights if not provided
        if label_weights is None:
            self.label_weights = {name: 1.0 for name in self.label_names}
        else:
            # Ensure all labels have weights, default missing ones to 1.0
            self.label_weights = {name: label_weights.get(name, 1.0) for name in self.label_names}
        
        # Normalize weights so they sum to the number of labels (maintains scale)
        total_weight = sum(self.label_weights.values())
        if total_weight > 0:
            self.label_weights = {name: weight * len(self.label_names) / total_weight 
                                 for name, weight in self.label_weights.items()}

        # Create individual loss functions for each label
        self.loss_functions = nn.ModuleDict()
        for label_name in self.label_names:
            if self.task_type == "der":
                if der_loss_coeffs is not None:
                    coeff = der_loss_coeffs.get(label_name, default_der_loss_coeff)
                else:
                    coeff = default_der_loss_coeff
                if use_log_reg:
                    self.loss_functions[label_name] = DERLossWithLogReg(
                        coeff=coeff,
                        alpha=log_reg_alpha,
                        epsilon=log_reg_epsilon
                    )
                else:
                    self.loss_functions[label_name] = DERLoss(coeff)
            elif self.task_type == "mve":
                if use_log_reg:
                    self.loss_functions[label_name] = NLLWithLogReg(
                        alpha=log_reg_alpha,
                        epsilon=log_reg_epsilon
                    )
                else:
                    self.loss_functions[label_name] = NLL()
            elif self.task_type == "msle":
                self.loss_functions[label_name] = MSLELoss(
                    alpha=log_reg_alpha,
                    epsilon=log_reg_epsilon
                )
            elif self.task_type == "point":
                # self.loss_functions[label_name] = nn.L1Loss()
                self.loss_functions[label_name] = nn.MSELoss()
            elif self.task_type == "bin":
                self.loss_functions[label_name] = nn.BCEWithLogitsLoss()
            else:
                raise ValueError(f"Unsupported task_type: {self.task_type}. Choose from 'der', 'mve', 'msle', 'point', 'bin'.")

        # Add metrics for logging
        self.metrics = {}
        for label in self.label_names:
            for stage in ["train", "val", "test"]:
                metric_id = f"{stage}_{label}"
                if self.task_type == "bin":
                    self.metrics[metric_id] = classification_metrics(f"{metric_id}_")
                else:
                    self.metrics[metric_id] = default_regression_metrics(f"{metric_id}_")
        # NOTE: Metrics are nn.Module, so they must be registered accordingly
        self.metrics = nn.ModuleDict(self.metrics)

    def forward(self, batch: Dict[str, torch.Tensor], return_embeddings: bool = False) -> torch.Tensor:
        return self.model(batch, return_embeddings)

    def _prepare_labels_tensor(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Prepare labels tensor in the correct order for the loss function.
        
        Args:
            batch: Input batch containing labels with keys matching self.label_names
            
        Returns:
            torch.Tensor of shape (batch_size, 1, n_targets) with labels in correct order
        """
        # Use self.label_to_idx to ensure correct ordering matches model output
        labels_list = [batch[label_name] for label_name in self.label_names]
        labels = torch.stack(labels_list, dim=-1)  # Shape: (batch_size, n_targets)
        return labels.unsqueeze(1)  # Shape: (batch_size, 1, n_targets)

    def _filter_nan_samples(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple:
        """
        Filter out samples where targets contain NaN values for metrics computation.
        
        Args:
            preds: Predictions tensor of any shape
            targets: Target tensor of same shape as preds
            
        Returns:
            Tuple of (filtered_preds, filtered_targets) with NaN samples removed
        """
        valid_mask = ~torch.isnan(targets)
        if valid_mask.sum() == 0:
            # If all samples are NaN, return empty tensors with correct dtype/device
            return preds[:0], targets[:0]
        
        return preds[valid_mask], targets[valid_mask]

    def _compute_weighted_loss(self, outputs: torch.Tensor, labels: torch.Tensor, stage: str) -> tuple:
        """
        Compute weighted loss for each label and return the weighted average.
        
        Key fixes:
        - Proper per-label NaN filtering before loss computation
        - Correct weight normalization that preserves relative importance
        - Robust handling when all samples are NaN for some labels
        
        Args:
            outputs: Model outputs of shape (batch_size, 4, n_targets)
            labels: Ground truth labels of shape (batch_size, 1, n_targets)
            
        Returns:
            Tuple of (weighted_loss, per_label_losses, valid_samples_per_label)
        """
        per_label_losses = []
        valid_samples_per_label = {}
        invalid_samples_per_label = {}
        
        for label_name in self.label_names:
            idx = self.label_to_idx[label_name]
            
            # Extract outputs and labels for this specific label
            label_outputs = outputs[:, :, idx:idx+1]  # Shape: (batch_size, 4, 1)
            label_targets = labels[:, :, idx:idx+1]   # Shape: (batch_size, 1, 1)
            
            # Flatten targets for NaN filtering
            label_targets_flat = label_targets.squeeze(-1).squeeze(-1)  # Shape: (batch_size,)
            
            # Filter out NaN samples for this specific label
            valid_mask = ~torch.isnan(label_targets_flat)
            n_valid = valid_mask.sum().item()
            valid_samples_per_label[label_name] = n_valid
            invalid_samples_per_label[label_name] = label_targets_flat.shape[0] - n_valid
            
            if n_valid == 0:
                # If all samples are NaN for this label, set loss to 0
                # Use tensor with requires_grad=True to maintain gradient flow
                label_loss = torch.tensor(0.0, device=outputs.device, requires_grad=True)
                self.log(f"{stage}_nan_warning_{label_name}", 1.0, on_step=False, on_epoch=True)
                if stage == "train":
                    logging.warning(f"All samples are NaN for label '{label_name}'. Setting loss to 0.")
            else:
                # Filter valid samples
                # - input: (N, *) containing logits
                # - target: (N, *) containing labels in [0, 1] as floats
                valid_label_outputs = label_outputs[valid_mask]  # Shape: (n_valid, 4, 1)
                valid_label_targets = label_targets[valid_mask]  # Shape: (n_valid, 1, 1)
                
                # Compute loss for this label using its specific DER loss function
                label_loss = self.loss_functions[label_name](valid_label_outputs, valid_label_targets.float())
            
            per_label_losses.append(label_loss)
        
        # Stack losses and apply weights
        per_label_losses = torch.stack(per_label_losses)  # Shape: (n_targets,)
        
        # Apply weights - only to labels that have valid samples
        # This prevents NaN labels from contributing to the weighted average
        weighted_losses = []
        total_weight = 0.0
        
        for i, label_name in enumerate(self.label_names):
            if valid_samples_per_label[label_name] > 0:
                weight = self.label_weights[label_name]
                weighted_losses.append(per_label_losses[i] * weight)
                total_weight += weight
            elif stage == "train":
                logging.info(f"Skipping label '{label_name}' in weighted loss due to no valid samples.")
        
        if len(weighted_losses) == 0:
            # All labels have NaN - return zero loss with gradient
            weighted_loss = torch.tensor(0.0, device=outputs.device, requires_grad=True)
            self.log(f"{stage}_all_labels_nan_warning", 1.0, on_step=False, on_epoch=True)
        else:
            # Compute weighted average - normalize by actual contributing weights
            weighted_loss = torch.stack(weighted_losses).sum() / max(total_weight, 1e-8)
        
        return weighted_loss, per_label_losses, invalid_samples_per_label

    def _process_batch(self, batch: Dict[str, torch.Tensor], stage: str):
        """
        Helper method to process a batch for training, validation, or testing.
        Reduces code duplication and ensures consistent logic across all stages.
        
        Key improvements:
        - Consistent NaN handling across all stages
        - Proper metrics computation with filtered samples
        - Clear separation between training (immediate logging) and val/test (accumulated metrics)
        
        Args:
            batch: Input batch containing features and labels
            stage: One of 'train', 'val', 'test'
            
        Returns:
            Total loss tensor for this batch
        """
        # Forward pass
        outputs = self(batch)  # Shape: (batch_size, 4, n_targets)
        labels = self._prepare_labels_tensor(batch)  # Shape: (batch_size, 1, n_targets)
        
        # Compute weighted loss and per-label losses
        loss, per_label_losses, invalid_samples_count = self._compute_weighted_loss(outputs, labels, stage)
        
        # Log overall loss
        self.log(f"{stage}_loss", loss, on_step=(stage=="train"), on_epoch=True, prog_bar=True)
        
        # Log per-label losses and valid sample counts
        for i, label_name in enumerate(self.label_names):
            self.log(f"{stage}_loss_{label_name}", per_label_losses[i], 
                    on_step=(stage=="train"), on_epoch=True, prog_bar=False)
            self.log(f"{stage}_invalid_samples_{label_name}", invalid_samples_count[label_name], 
                    on_step=False, on_epoch=True, prog_bar=False)

        # Update metrics for each label using explicit mapping (with NaN filtering)
        for label_name, output_idx in self.label_to_idx.items():
            # Extract predictions for this specific label (gamma parameter from DER)
            preds = outputs[:, 0, output_idx]  # Shape: (batch_size,) - gamma predictions
            target = batch[label_name]         # Shape: (batch_size,) - ground truth
            
            # Filter out NaN values for metrics computation
            filtered_preds, filtered_target = self._filter_nan_samples(preds, target)
            
            # NOTE: For classification, we do not neet to apply sigmoid because
            # the metrics handle logits directly.

            if len(filtered_target) > 0:  # Only compute metrics if we have valid samples
                metrics_key = f"{stage}_{label_name}"
                if stage == "train":
                    # For training, compute and log metrics immediately (no accumulation)
                    # This prevents memory buildup during long training
                    scores = self.metrics[metrics_key](filtered_preds, filtered_target)
                    self.log_dict(scores)
                else:
                    # For validation/test, accumulate metrics and compute at epoch end
                    # This gives more stable metrics across the full dataset
                    self.metrics[metrics_key].update(filtered_preds, filtered_target)
        
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with immediate metrics logging to prevent memory buildup."""
        return self._process_batch(batch, "train")

    def on_train_epoch_end(self):
        for name in self.label_names:
            k = f"train_{name}"
            self.log_dict(self.metrics[k].compute())
            self.metrics[k].reset()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with metrics accumulation for epoch-end computation."""
        return self._process_batch(batch, "val")

    def on_validation_epoch_end(self):
        for label_name in self.label_names:
            metrics_key = f"val_{label_name}"
            metrics = self.metrics[metrics_key].compute()
            self.log_dict(metrics)
            self.metrics[metrics_key].reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._process_batch(batch, "test")

    def on_test_epoch_end(self):
        for label_name in self.label_names:
            metrics_key = f"test_{label_name}"
            metrics = self.metrics[metrics_key].compute()
            self.log_dict(metrics)
            self.metrics[metrics_key].reset()

    def predict_step(
        self, X: Dict[str, torch.Tensor], batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Prediction Step Deep Evidential Regression.

        Args:
            X: prediction batch of shape [batch_size x input_dims]
            batch_idx: the index of this batch
            dataloader_idx: the index of the dataloader

        Returns:
            dictionary with predictions and uncertainty measures
        """
        with torch.no_grad():
            pred, embeddings = self.forward(X, return_embeddings=True)  # [batch_size x 4 x othe_dims]

        if self.task_type == "der":
            gamma, nu, alpha, beta = (
                pred[:, 0:1, ...],
                pred[:, 1:2, ...],
                pred[:, 2:3, ...],
                pred[:, 3:4, ...],
            )

            epistemic_uct = self.compute_epistemic_uct(nu)
            aleatoric_uct = self.compute_aleatoric_uct(beta, alpha, nu)
            pred_uct = epistemic_uct + aleatoric_uct

            # Squeeze unnecessary dimensions for easier handling, from
            # [B,1,n_targets] to [B,n_targets]
            gamma = gamma.squeeze(1)
            pred_uct = pred_uct.squeeze(1)
            aleatoric_uct = aleatoric_uct.squeeze(1)
            epistemic_uct = epistemic_uct.squeeze(1)

            results = {
                "pred": gamma,
                "pred_uct": pred_uct,
                "aleatoric_uct": aleatoric_uct,
                "epistemic_uct": epistemic_uct,
                "out": pred,
                "embeddings": embeddings,
            }
        
            for label_name, idx in self.label_to_idx.items():
                results[f"pred_{label_name}"] = gamma[:, idx]
                results[f"pred_uct_{label_name}"] = pred_uct[:, idx]
                results[f"aleatoric_uct_{label_name}"] = aleatoric_uct[:, idx]
                results[f"epistemic_uct_{label_name}"] = epistemic_uct[:, idx]

        elif self.task_type == "mve":
            mu, log_sigma_2 = pred[:, 0:1, ...], pred[:, 1:2, ...]

            # Squeeze unnecessary dimensions for easier handling, from
            # [B,1,n_targets] to [B,n_targets]
            mu = mu.squeeze(1)
            log_sigma_2 = log_sigma_2.squeeze(1)

            # From MVERegression in: https://github.com/lightning-uq-box/lightning-uq-box/blob/main/lightning_uq_box/uq_methods/mean_variance_estimation.py
            eps = torch.ones_like(log_sigma_2) * 1e-6
            std = torch.sqrt(eps + torch.exp(log_sigma_2))

            results = {
                "pred": mu,
                "pred_uct": std,
                "aleatoric_uct": std,
                "epistemic_uct": torch.zeros_like(std), # No epistemic uncertainty in MVE
                "out": pred,
                "embeddings": embeddings,
            }
            for label_name, idx in self.label_to_idx.items():
                results[f"pred_{label_name}"] = mu[:, idx]
                results[f"pred_uct_{label_name}"] = std[:, idx]
                results[f"aleatoric_uct_{label_name}"] = std[:, idx]
                results[f"epistemic_uct_{label_name}"] = torch.zeros_like(std[:, idx])

        elif self.task_type == "point" or self.task_type == "msle":
            mu = pred[:, 0:1, ...]  # [B,1,n_targets]
            # Squeeze unnecessary dimensions for easier handling, from
            # [B,1,n_targets] to [B,n_targets]
            mu = mu.squeeze(1)
            results = {
                "pred": mu,
                "out": pred,
                "embeddings": embeddings,
            }
            for label_name, idx in self.label_to_idx.items():
                results[f"pred_{label_name}"] = mu[:, idx]
                results[f"pred_uct_{label_name}"] = torch.zeros_like(mu[:, idx])
                results[f"aleatoric_uct_{label_name}"] = torch.zeros_like(mu[:, idx])
                results[f"epistemic_uct_{label_name}"] = torch.zeros_like(mu[:, idx])
                
        elif self.task_type == "bin":
            logits = pred[:, 0:1, ...]  # [B,1,n_targets]
            # Squeeze unnecessary dimensions for easier handling, from
            # [B,1,n_targets] to [B,n_targets]
            logits = logits.squeeze(1)
            results = {
                "pred": torch.sigmoid(logits),
                "logits": logits,
                "out": pred,
                "embeddings": embeddings,
            }
            for label_name, idx in self.label_to_idx.items():
                results[f"pred_{label_name}"] = torch.sigmoid(logits[:, idx])
                results[f"logits_{label_name}"] = logits[:, idx]

        return results

    def lr_warmup_config(self):
        train_batches = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(self.warmup_ratio * train_batches))

        def warmup(step):
            """
            This method will be called for ceil(warmup_batches/accum_grad_batches) times,
            warmup_steps has been adjusted accordingly
            """
            
            if warmup_steps <= 0:
                factor = 1
            else:
                factor = min(step / warmup_steps, 1)
            return factor

        opt1 = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        return {
            'frequency': warmup_steps,
            'optimizer': opt1,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(opt1, warmup),
                'interval': 'step',
                'frequency': 1,
                'name': 'lr/warmup'
            },
        }


    def lr_decay_config(self):
        opt2 = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)
        train_batches = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(self.warmup_ratio * train_batches))
        return {
            'frequency': train_batches - warmup_steps,
            'optimizer': opt2,
            'lr_scheduler': {
                # NOTE: threshold=0.01, threshold_mode='rel' means: metric
                # should be improving by at least 1% within patience steps
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt2, 'min', factor=0.1, patience=5,
                    threshold=1e-4, threshold_mode='rel',
                    min_lr=1e-6),
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss',
                'strict': False,
                'name': 'lr/reduce_on_plateau',
            }
        }

    def configure_optimizers(self):
        """ Configure optimizers and learning rate schedulers. """
        opt = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        if self.lr_scheduler_type == "linear":
            sched = LinearLR(
                opt,
                start_factor=0.0,
                end_factor=1.0,
                total_iters=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"},
            }
        elif self.lr_scheduler_type == "reduce_on_plateau":
            training_steps = self.trainer.estimated_stepping_batches
            warmup_steps = max(1, int(self.warmup_ratio * training_steps))
    
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=0.0001,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            red_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=0.1,
                patience=1,
                min_lr=1e-7,
            )
            lr_scheduler = {
                "scheduler": red_plateau,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            }
            return [opt], [lr_scheduler, {'scheduler': warmup}]
            # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     opt,
            #     mode="min",
            #     factor=0.1,
            #     patience=5,
            #     threshold=1e-4,
            #     min_lr=1e-6,
            # )
            # return {
            #     "optimizer": opt,
            #     "lr_scheduler": {
            #         "scheduler": sched,
            #         "monitor": "val_loss",
            #         "interval": "epoch",
            #         "frequency": 1,
            #     },
            # }
        elif self.lr_scheduler_type == "cosine":
            T = self.trainer.estimated_stepping_batches
            W = max(1, int(self.warmup_ratio * T))

            def lr_lambda(s):
                # Linear warmup
                if s < W:
                    return (s + 1) / W
                # Oscillating cosine decay with multiple cycles
                # NOTE: When num_cycles == 1, cosine directly decays to 0
                progress = (s - W) / max(1, T - W)
                cosine_arg = self.num_cycles * math.pi * progress
                return 0.5 * (1 + math.cos(cosine_arg))

            sched = LambdaLR(opt, lr_lambda)
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"},
            }

    def compute_aleatoric_uct(self, beta: torch.Tensor, alpha: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """Compute the aleatoric uncertainty for DER model.

        Equation 10 of the paper

        Args:
            beta: beta output DER model
            alpha: alpha output DER model
            nu: nu output DER model

        Returns:
            Aleatoric Uncertainty
        """
        return torch.sqrt(torch.div(beta * (1 + nu), alpha * nu))

    def compute_epistemic_uct(self, nu: torch.Tensor) -> torch.Tensor:
        """Compute the aleatoric uncertainty for DER model.

        Equation 10: of the paper

        Args:
            nu: nu output DER model
        Returns:
            Epistemic Uncertainty
        """
        return torch.reciprocal(torch.sqrt(nu))
    
    def extract_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract embeddings from the model for the given batch.
        
        Args:
            batch: Input batch containing features
            
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        with torch.no_grad():
            _, embeddings = self.model.eval()(batch, return_embeddings=True)
        return embeddings