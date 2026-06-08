"""
Arquitetura de classificador MLP em PyTorch.

Diferenças centrais em relação ao `MLPRegressor`:
    - Saída de `n_classes` neurônios (logits), um por classe
    - Função de perda esperada: `CrossEntropyLoss` (já inclui softmax)
    - Predição final via `argmax` sobre os logits
"""

from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """
    MLP para tarefas de classificação multiclasse.

    Arquitetura:
        Input → [Linear → ReLU → Dropout] × n_hidden_layers → Linear (n_classes)

    A última camada **não** aplica softmax: a função de perda padrão para
    classificação no PyTorch (`nn.CrossEntropyLoss`) espera logits brutos
    e aplica softmax internamente, de forma numericamente estável.

    Args:
        input_dim: Número de features de entrada.
        hidden_dims: Lista com o número de neurônios em cada camada oculta.
        n_classes: Número de classes-alvo (≥ 2).
        dropout: Probabilidade de dropout entre camadas.

    Example:
        >>> import torch
        >>> model = MLPClassifier(input_dim=4, hidden_dims=[32, 16], n_classes=3)
        >>> x = torch.randn(5, 4)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([5, 3])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        n_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if n_classes < 2:
            raise ValueError(f"n_classes deve ser ≥ 2. Recebido: {n_classes}")
        if not hidden_dims:
            raise ValueError("hidden_dims não pode ser vazio.")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout deve estar em [0, 1). Recebido: {dropout}")

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        # Saída: n_classes logits (sem ativação — CrossEntropyLoss cuida do softmax)
        layers.append(nn.Linear(prev_dim, n_classes))

        self.network = nn.Sequential(*layers)
        self.config = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "n_classes": n_classes,
            "dropout": dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Retorna logits (batch_size, n_classes)."""
        return self.network(x)

    def count_parameters(self) -> int:
        """Conta o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
