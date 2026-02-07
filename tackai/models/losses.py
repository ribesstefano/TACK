""" Custom loss functions for regression tasks."""

import torch
import torch.nn as nn
from lightning_uq_box.uq_methods.loss_functions import DERLoss, NLL


class MSLELoss(nn.Module):
    """
    Mean Squared Log-scaled Error Loss.
    
    This loss combines standard MSE with a log-scaled MSE term to amplify
    gradients for small prediction errors. Particularly useful when predictions
    are in the range [0, 1].
    
    Loss formula:
        L = MSE(y, ŷ) + α * MSE(log(y + ε), log(ŷ + ε))
    
    where:
        - MSE is mean squared error
        - α (alpha) is a weighting hyperparameter for the log term
        - ε (epsilon) is a small constant to prevent log(0)
    
    Parameters
    ----------
    alpha : float, default=5.0
        Weight for the log-scaled term. Higher values amplify small errors more.
        Suggested starting value: 5.0
    epsilon : float, default=1e-9
        Small constant added before taking log to prevent log(0).
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        
    Examples
    --------
    >>> loss_fn = MSLELoss(alpha=5.0)
    >>> predictions = torch.tensor([0.1, 0.5, 0.9])
    >>> targets = torch.tensor([0.15, 0.45, 0.85])
    >>> loss = loss_fn(predictions, targets)
    
    Notes
    -----
    - Works best when targets and predictions are in [0, 1] range
    - The log term amplifies gradients for small values
    - Can be used as a drop-in replacement for MSELoss
    """
    
    def __init__(
        self,
        alpha: float = 5.0,
        epsilon: float = 1e-9,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.reduction = reduction
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the MSLE loss.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted values of shape (batch_size, ...)
        target : torch.Tensor
            Ground truth values of same shape as pred
            
        Returns
        -------
        torch.Tensor
            Scalar loss value (if reduction='mean' or 'sum')
            or tensor of losses (if reduction='none')
        """
        # Standard MSE term
        mse_term = (target - pred) ** 2
        
        # Log-scaled MSE term
        log_target = torch.log(target + self.epsilon)
        log_pred = torch.log(pred + self.epsilon)
        log_mse_term = (log_target - log_pred) ** 2
        
        # Combine terms
        loss = mse_term + self.alpha * log_mse_term
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class DERLossWithLogReg(nn.Module):
    """
    Deep Evidential Regression Loss with log-scaled regularization.
    
    This extends the standard DER loss by adding a log-scaled MSE term between
    predictions (gamma) and targets to amplify gradients for small errors.
    
    Loss formula:
        L = DER_loss(pred, target) + α * MSE(log(γ + ε), log(y + ε))
    
    where γ is the gamma prediction from DER output.
    
    Parameters
    ----------
    coeff : float, default=1.0
        Regularization coefficient for the DER loss
    alpha : float, default=5.0
        Weight for the log-scaled regularization term
    epsilon : float, default=1e-9
        Small constant added before taking log to prevent log(0)
        
    Examples
    --------
    >>> loss_fn = DERLossWithLogReg(coeff=0.01, alpha=5.0)
    >>> # DER outputs: [batch_size, 4, n_targets] = [gamma, nu, alpha, beta]
    >>> pred = torch.randn(32, 4, 2)
    >>> target = torch.randn(32, 1, 2)
    >>> loss = loss_fn(pred, target)
    
    Notes
    -----
    - Combines uncertainty quantification from DER with improved gradient flow
    - The log term helps when predictions are in range [0, 1]
    - Works as a drop-in replacement for standard DERLoss
    """
    
    def __init__(
        self,
        coeff: float = 1.0,
        alpha: float = 5.0,
        epsilon: float = 1e-9
    ):
        super().__init__()
        self.der_loss = DERLoss(coeff=coeff)
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute DER loss with log-scaled regularization.
        
        Parameters
        ----------
        pred : torch.Tensor
            DER predictions of shape (batch_size, 4, n_targets)
            where dim 1 contains [gamma, nu, alpha, beta]
        target : torch.Tensor
            Ground truth of shape (batch_size, 1, n_targets)
            
        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        # Standard DER loss
        der_loss = self.der_loss(pred, target)
        
        # Extract gamma (mean prediction) from DER output
        gamma = pred[:, 0:1, ...]  # Shape: (batch_size, 1, n_targets)
        
        # Log-scaled MSE term between gamma and target
        log_target = torch.log(target + self.epsilon)
        log_gamma = torch.log(gamma + self.epsilon)
        log_mse = ((log_target - log_gamma) ** 2).mean()
        
        # Combine losses
        total_loss = der_loss + self.alpha * log_mse
        
        return total_loss


class NLLWithLogReg(nn.Module):
    """
    Negative Log-Likelihood loss with log-scaled regularization.
    
    This extends the standard NLL loss for Mean-Variance Estimation by adding
    a log-scaled MSE term between predictions (mu) and targets.
    
    Loss formula:
        L = NLL_loss(pred, target) + α * MSE(log(μ + ε), log(y + ε))
    
    where μ is the mean prediction from NLL output.
    
    Parameters
    ----------
    alpha : float, default=5.0
        Weight for the log-scaled regularization term
    epsilon : float, default=1e-9
        Small constant added before taking log to prevent log(0)
        
    Examples
    --------
    >>> loss_fn = NLLWithLogReg(alpha=5.0)
    >>> # MVE outputs: [batch_size, 2, n_targets] = [mu, log_sigma_2]
    >>> pred = torch.randn(32, 2, 2)
    >>> target = torch.randn(32, 1, 2)
    >>> loss = loss_fn(pred, target)
    
    Notes
    -----
    - Combines probabilistic uncertainty from NLL with improved gradient flow
    - The log term helps when predictions are in range [0, 1]
    - Works as a drop-in replacement for standard NLL loss
    """
    
    def __init__(
        self,
        alpha: float = 5.0,
        epsilon: float = 1e-9
    ):
        super().__init__()
        self.nll_loss = NLL()
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute NLL loss with log-scaled regularization.
        
        Parameters
        ----------
        pred : torch.Tensor
            MVE predictions of shape (batch_size, 2, n_targets)
            where dim 1 contains [mu, log_sigma_2]
        target : torch.Tensor
            Ground truth of shape (batch_size, 1, n_targets)
            
        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        # Standard NLL loss
        nll_loss = self.nll_loss(pred, target)
        
        # Extract mu (mean prediction) from MVE output
        mu = pred[:, 0:1, ...]  # Shape: (batch_size, 1, n_targets)
        
        # Log-scaled MSE term between mu and target
        log_target = torch.log(target + self.epsilon)
        log_mu = torch.log(mu + self.epsilon)
        log_mse = ((log_target - log_mu) ** 2).mean()
        
        # Combine losses
        total_loss = nll_loss + self.alpha * log_mse
        
        return total_loss