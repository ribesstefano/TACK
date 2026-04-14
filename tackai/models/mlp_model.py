""" Simple MLP model for regression tasks. """
from typing import Dict, Literal, Optional, Union, List
import torch
import torch.nn as nn
from lightning_uq_box.uq_methods.deep_evidential_regression import DERLayer

from tackai.models.regression_head import RegressionHead


class MLPModel(nn.Module):
    """
    Simple Multi-Layer Perceptron model for regression.
    
    Takes concatenated input features and processes them through linear layers
    before passing to a regression head.
    
    Parameters
    ----------
    input_dim : int
        Dimension of the input features.
    hidden_dim : int
        Hidden dimension for the MLP layers.
    n_targets : int
        Number of target variables to predict.
    mlp_depth : int, optional
        Number of layers in the MLP backbone. Default is 2.
    dropout : float, optional
        Dropout probability. Default is 0.1.
    task_type : Literal["single", "multi"], optional
        Type of regression head. Default is "single".
    head_depth : Union[int, List[int]], optional
        Depth of the regression head(s). Default is 1.
    head_dropout : float, optional
        Dropout probability for the head. Default is 0.1.
    task_type : Literal["der", "mve", "point", "msle"], optional
        Type of regression output. Default is "der".
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[int, List[int]],
        n_targets: int = 1,
        mlp_depth: int = 2,
        dropout: float = 0.1,
        head_type: Literal["single", "multi"] = "single",
        head_depth: Union[int, List[int]] = 1,
        head_dropout: float = 0.1,
        task_type: Literal["der", "mve", "point", "msle", "bin"] = "der",
        activation: Literal["relu", "gelu", "silu"] = "gelu",
        norm_type: Union[None, Literal["batch", "layer"]] = None,
        categorical_vocab_sizes: Optional[Dict[str, int]] = None,
        categorical_embedding_dim: int = 8,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_targets = n_targets
        self.head_type = head_type
        self.task_type = task_type

        # PyTorch Embedding tables for categorical features.
        # categorical_vocab_sizes maps feature key → vocab size (from
        # datamodule.get_categorical_vocab_sizes()).  When None, no embedding
        # layers are created and behavior is identical to the original MLP.
        self.categorical_vocab_sizes = categorical_vocab_sizes or {}
        self.categorical_embedding_dim = categorical_embedding_dim
        self.embeddings = nn.ModuleDict({
            key: nn.Embedding(vocab_size, categorical_embedding_dim)
            for key, vocab_size in self.categorical_vocab_sizes.items()
        })

        if task_type == "der":
            # DER: 4 values per target (gamma, nu, alpha, beta)
            self.output_dim = 4
        elif task_type == "mve":
            # MVE: 2 values per target (mu, log_sigma_2)
            self.output_dim = 2
        elif task_type in ["point", "msle", "bin"]:
            self.output_dim = 1
        else:
            raise ValueError(f"Invalid task_type: {task_type}. Choose from 'der', 'mve', 'point', 'msle', 'bin'.")
        
        # Setup MLP hidden dimensions according to the arguments
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * mlp_depth
        else:
            hidden_dims = hidden_dim

        # When embedding tables are present, their outputs are concatenated to
        # the continuous features before the first linear layer.
        effective_input_dim = input_dim + len(self.categorical_vocab_sizes) * categorical_embedding_dim

        # Build MLP backbone
        layers = []
        current_dim = effective_input_dim
        for hdim in hidden_dims:
            # Add linear layer
            layers.append(nn.Linear(current_dim, hdim))
            # Add normalization if specified
            if norm_type == "batch":
                layers.append(nn.BatchNorm1d(hdim))
            elif norm_type == "layer":
                layers.append(nn.LayerNorm(hdim))
            # Add activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Invalid activation: {activation}. Choose from 'relu', 'gelu', 'silu'.")
            # Add dropout if specified
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hdim
        
        self.mlp = nn.Sequential(*layers)

        last_dim = hidden_dim if isinstance(hidden_dim, int) else hidden_dim[-1]
        
        if task_type == "bin":
            self.head = nn.Linear(last_dim, n_targets * self.output_dim, bias=False)
        else:            
            # Regression head
            self.head = RegressionHead(
                n_targets=n_targets,
                hidden_dim=last_dim,
                head_type=head_type,
                depth=head_depth,
                dropout=head_dropout,
                task_type=task_type,
            )
        
        # DER layer: some of the returned parameters must be positive, so the
        # layer applies a softplus to ensure positivity
        if task_type == "der":
            self.der_layer = DERLayer()
    
    def forward(
        self, 
        batch: Dict[str, torch.Tensor], 
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass.
        
        Args
        ----
        batch : Dict[str, torch.Tensor]
            Input batch. Should contain a key "features" with shape (batch_size, input_dim)
        return_embeddings : bool, optional
            Whether to return embeddings along with predictions. Default is False.
        
        Returns
        -------
        If return_embeddings is False:
            Tensor of shape (batch_size, output_dim, n_targets)
        If return_embeddings is True:
            Tuple of (predictions, embeddings)
        """        
        # Separate continuous Feature_* keys from categorical ones that go
        # through embedding tables.  Both sets are sorted for consistent ordering.
        cat_keys = set(self.categorical_vocab_sizes.keys())
        continuous_keys = sorted([k for k in batch.keys()
                                  if k.startswith("Feature_") and k not in cat_keys])
        categorical_keys = sorted([k for k in batch.keys()
                                   if k.startswith("Feature_") and k in cat_keys])

        parts = [batch[k] for k in continuous_keys]
        for k in categorical_keys:
            idx = batch[k].long().squeeze(-1)   # (batch_size,)
            parts.append(self.embeddings[k](idx))  # (batch_size, categorical_embedding_dim)

        x = torch.cat(parts, dim=-1)
        
        # Process through MLP
        embeddings = self.mlp(x)  # (batch_size, hidden_dim)
        
        # Get predictions from head
        x = self.head(embeddings)
        
        # Reshape to (batch_size, output_dim, n_targets), if head is not RegressionHead
        if self.task_type == "bin":
            x = x.view(x.size(0), self.output_dim, self.n_targets)
        
        # Apply DER layer
        if self.task_type == "der":
            x = self.der_layer(x)
        
        if return_embeddings:
            return x, embeddings
        return x
