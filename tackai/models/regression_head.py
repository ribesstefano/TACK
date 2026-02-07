from typing import List, Literal, Union

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """
    Head module for regression tasks supporting both single and multi-head architectures.
    
    Parameters
    ----------
    n_targets : int
        Number of target variables to predict.
    hidden_dim : int
        Hidden dimension for the head layers.
    head_type : Literal["single", "multi"]
        Type of head architecture:
        - "single": One shared head for all targets
        - "multi": Separate head for each target
    depth : int or List[int], optional
        Depth (number of layers) for each head. If int, same depth for all heads.
        If List[int] and head_type="multi", different depth per head. Default is 1.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    task_type : Literal["der", "mve"], optional
        Type of regression output:
        - "der": Deep Evidential Regression (outputs 4 values per target)
        - "mve": Mean Variance Estimation (outputs 2 values per target)
        - "point": Point estimate regression (outputs 1 value per target)
        - "msle": Mean Squared Logarithmic Error regression (outputs 1 value per target)
        Default is "der".
    """
    def __init__(
        self,
        n_targets: int,
        hidden_dim: int,
        head_type: Literal["single", "multi"] = "single",
        depth: Union[int, List[int]] = 1,
        dropout: float = 0.0,
        task_type: Literal["der", "mve", "point", "msle"] = "der",
    ):
        super().__init__()
        self.n_targets = n_targets
        self.head_type = head_type
        self.task_type = task_type
        
        if task_type == "der":
            # DER (Deep Evidential Regression) outputs 4 values per target,
            # named: (gamma, nu, alpha, beta)
            self.output_dim = 4
        elif task_type == "mve":
            # MVE (Mean Variance Estimation) outputs 2 values per target,
            # named: (mu, log_sigma_2)
            self.output_dim = 2
        else:  # point
            # Point estimate outputs 1 value per target
            self.output_dim = 1

        if head_type == "multi" and isinstance(depth, list) and len(depth) != n_targets:
            raise ValueError("Length of depth list must match n_targets when head_type is 'multi'.")
        
        if head_type == "single":
            # Single shared head for all targets
            layers = []
            for i in range(depth if isinstance(depth, int) else depth[0]):
                if i == depth - 1 if isinstance(depth, int) else i == depth[0] - 1:
                    # Final layer outputs n_targets
                    layers.append(nn.Linear(hidden_dim, self.output_dim * n_targets, bias=False))
                else:
                    layers.extend([
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout)
                    ])
            self.head = nn.Sequential(*layers)

        elif head_type == "multi":
            # Multiple heads, one for each target
            self.heads = nn.ModuleList()
            depths = depth if isinstance(depth, list) else [depth] * n_targets
            
            for i in range(n_targets):
                layers = []
                head_depth = depths[i] if i < len(depths) else depths[0]
                
                for j in range(head_depth):
                    if j == head_depth - 1:
                        # Final layer outputs self.output_dim elements for this target
                        layers.append(nn.Linear(hidden_dim, self.output_dim, bias=False))
                    else:
                        layers.extend([
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout)
                        ])
                self.heads.append(nn.Sequential(*layers))
        else:
            raise ValueError(f"Unsupported head_type: {head_type}. Choose 'single' or 'multi'.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args
        ----
        x : Tensor of shape (batch_size, hidden_dim)
        
        Returns
        -------
        Tensor of shape (batch_size, output_dim x n_targets)
        """
        if self.head_type == "single":
            x = self.head(x)
        else:
            # Concatenate outputs from all heads
            outputs = [head(x) for head in self.heads]
            x = torch.cat(outputs, dim=1)

        # Reshape to (batch_size, output_dim, n_targets)
        return x.view(-1, self.output_dim, self.n_targets)