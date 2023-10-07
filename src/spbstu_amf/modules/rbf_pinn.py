"""
Модель и модуль PINN с использованием RBF
"""

from typing import Callable
import lightning as L
import torch
from torch import nn


BASIS_FUNCTIONS = {
    "gaussian": lambda alpha: torch.exp(-1 * alpha.pow(2)),
    "linear": lambda alpha: alpha,
    "quadratic": lambda alpha: alpha.pow(2),
    "inverse_quadratic": lambda alpha: torch.ones_like(alpha)
    / (torch.ones_like(alpha) + alpha.pow(2)),
}


class RbfLayer(nn.Module):
    """
    Radial Basis Function слой
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        basis_function: Callable,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.basis_function = basis_function
        self.centers = nn.Parameter(
            torch.Tensor(self.input_features, self.output_features)
        )
        self.log_sigmas = nn.Parameter(torch.Tensor(self.output_features))

    def reset_parameters(self) -> None:
        nn.init.normal_(self.centers, 0.0, 1.0)
        nn.init.constant_(self.log_sigmas, 0.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size = X.size(0)
        size = (batch_size, self.output_features, self.input_features)
        X = X.squeeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        distances = (X - c).pow(2).sum(-1).pow(0.5) / torch.exp(
            self.log_sigmas
        ).unsqueeze(0)
        return self.basis_function(distances)


class RbfPinnNet(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_features: int,
        output_features: int,
        hidden_features: int,
    ) -> None:
        super().__init__()


class RbfPinnModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
