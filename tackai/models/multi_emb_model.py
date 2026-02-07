"""
Multi-embedding regression model combining various feature embeddings.
"""
from typing import Literal, Optional, Dict, List, Union

import torch
from torch import nn
import torch.nn.functional as F

from lightning_uq_box.uq_methods.deep_evidential_regression import DERLayer

from tackai.models.regression_head import RegressionHead


class InputProjector(nn.Module):
    """
    Input projecting layer that applies a linear transformation to the input features.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, optional
        Whether to include a bias term in the linear transformation. Default is True.
    norm : bool, optional
        Whether to apply Layer Normalization before the linear transformation. Default is False.
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            norm: Optional[Literal["batch", "layer"]] = None,
    ):
        super().__init__()
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = None
        if norm is not None and norm == "layer":
            self.norm = nn.LayerNorm(out_features)
        elif norm is not None and norm == "batch":
            self.norm = nn.BatchNorm1d(out_features)
        self.alpha = nn.Parameter(torch.tensor(1 / (out_features ** 0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
        # Divide by the square root of the output dimension to maintain variance
        # This is a common practice in Transformer models to stabilize training
        # return F.softmax(x * self.alpha, dim=-1)

class GPTStyleFeedForward(nn.Module):
    """
    Pre-LN feed-forward block used in decoder-only Transformers (Radford et al., 2018/GPT family).

    Structure (for input X ∈ R^{BxTxd_model}):
        Y = X + Dropout( W2 * φ( W1 * LN(X) ) )
    where φ is GELU, and W1,W2 are linear (affine) maps with optional bias.

    Parameters
    ----------
    d_model : int
        Hidden size of the model.
    d_ff : Optional[int]
        Inner width of the MLP. Defaults to 4 * d_model if None.
    dropout : float
        Dropout probability applied after the second linear.
    bias : bool
        Whether to use bias terms in the two linear layers. In GPT-style models this is True. 
        (See HF GPT-2 `Conv1D` layers, which include `bias`.) 
    eps : float
        Epsilon for LayerNorm stability.

    Notes
    -----
    * This follows the *pre-norm* residual layout used in GPT-1/2: LayerNorm → MLP → residual add.
    * Activation is GELU, as in GPT-2.
    * Expected input shape is (batch, seq, d_model); the module is position-wise.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        inner = 4 * d_model if d_ff is None else d_ff

        # Pre-norm
        self.ln = nn.LayerNorm(d_model, eps=eps)

        # Two affine projections (bias enabled by default, matching GPT practice)
        self.fc1 = nn.Linear(d_model, inner, bias=bias)
        self.fc2 = nn.Linear(inner, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args
        ----
        x : Tensor of shape (B, T, d_model)

        Returns
        -------
        Tensor of shape (B, T, d_model)
        """
        # Pre-norm residual block
        h = self.ln(x)
        h = F.gelu(self.fc1(h))       # position-wise nonlinearity
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h                   # residual add

class MultiEmbeddingsBackbone(nn.Module):
    """
    Backbone network for feature embedding and trunk processing.
    """
    def __init__(
            self,
            feature_dims: Dict[str, int],
            hidden_dim: int = 128,
            trunk_type: Literal["gpt", "mlp"] = "mlp",
            trunk_size: int = 1,
            dropout: float = 0.1,
            input_norm: Optional[Literal["batch", "layer"]] = None,
            trunk_norm: Optional[Literal["batch", "layer"]] = None,
    ):
        super().__init__()
        self.trunk_type = trunk_type
        self.trunk_size = trunk_size
        self.hidden_dim = hidden_dim
        self.input_proj = nn.ModuleDict({
            k: InputProjector(d, hidden_dim, norm=input_norm, bias=False) for k, d in feature_dims.items()
        })
        # self.fusion = GatedFusion(d=hidden_dim, K=len(feature_dims))
        # self.fusion = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)
        if trunk_type == "gpt":
            self.trunk = nn.Sequential(
                *[GPTStyleFeedForward(hidden_dim, hidden_dim, dropout, bias=False) for _ in range(trunk_size)]
            )
        elif trunk_type == "mlp":
            norm_layer = None
            if trunk_norm is not None and trunk_norm == "layer":
                norm_layer = nn.LayerNorm(hidden_dim)
            elif trunk_norm is not None and trunk_norm == "batch":
                norm_layer = nn.BatchNorm1d(hidden_dim)
            else:
                norm_layer = nn.Identity()
            self.trunk = []
            for _ in range(trunk_size):
                self.trunk.extend([
                    norm_layer,
                    nn.Linear(hidden_dim, hidden_dim, bias=True),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
            self.trunk = nn.Sequential(*self.trunk)
        else:
            raise ValueError(f"Unsupported trunk_type: {trunk_type}. Choose 'gpt' or 'mlp'.")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedded_features = []
        for k in self.input_proj.keys():
            if k in batch:
                embedded_features.append(self.input_proj[k](batch[k]))
            else:
                raise KeyError(f"Key '{k}' not found in the input batch.")

        # Stack along a new dimension and sum to combine features
        # In our case, we have four features: mol, poi, e3, cell
        x = torch.stack(embedded_features, dim=-1)  # (batch_size, hidden_dim, n_features)
        x = x.sum(dim=-1)  # (batch_size, hidden_dim)
        
        # # Concatenate features and project to hidden_dim
        # embedded_features = torch.cat(embedded_features, dim=-1)  # (batch_size, hidden_dim * n_features)
        # x = self.fusion(embedded_features)  # (batch_size, hidden_dim)

        x = self.trunk(x)
        return x

class MultiEmbeddingsRegressionModel(nn.Module):
    """
    Regression model with backbone, linear head, and DER layer.

    TODO: Merge with BERTModel?
    """
    def __init__(
            self,
            n_targets: int = 1,
            head_type: Literal["single", "multi"] = "single",
            head_depth: Union[int, List[int]] = 1,
            head_dropout: float = 0.0,
            task_type: Literal["der", "mve"] = "der",
            **model_kwargs,
    ):
        super().__init__()
        self.n_targets = n_targets
        self.head_type = head_type
        self.task_type = task_type

        # Multi-embedding backbone
        self.backbone = MultiEmbeddingsBackbone(**model_kwargs)

        # Regression head
        self.regression_head = RegressionHead(
            n_targets=n_targets,
            hidden_dim=self.backbone.hidden_dim,
            head_type=head_type,
            depth=head_depth,
            dropout=head_dropout,
            task_type=task_type,
        )

        # DER layer: some of the returned parameters must be positive, so the
        # layer applies a softplus to ensure positivity
        if task_type == "der":
            self.der_layer = DERLayer()

    def forward(self, batch: Dict[str, torch.Tensor], return_embeddings: bool = False) -> torch.Tensor:
        # Get backbone embeddings
        embedding = self.backbone(batch)

        # Pass through regression head
        x = self.regression_head(embedding)
        
        # Apply DER layer
        if self.task_type == "der":
            x = self.der_layer(x)
        
        if return_embeddings:
            return x, embedding
        return x