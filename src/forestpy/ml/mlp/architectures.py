"""
Arquiteturas de Redes Neurais Densas (MLP) em PyTorch.

Multi-Layer Perceptrons (MLPs) são a arquitetura fundamental para tarefas
tabulares (features em colunas, sem estrutura espacial ou temporal explícita).
Compostas por camadas densas (`Linear`) intercaladas com funções de ativação
não-lineares (ReLU é a escolha-padrão moderna), permitindo aproximar funções
arbitrárias (Teorema da Aproximação Universal de Cybenko, 1989).

Para problemas de regressão florestal (volumetria, hipsometria), MLPs
tipicamente vencem regressão linear quando há **não-linearidades residuais**
não capturadas pela forma funcional clássica.

Referências:
    Goodfellow, I.; Bengio, Y.; Courville, A. (2016). *Deep Learning*. MIT Press.
    Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPRegressor(nn.Module):
    """
    MLP para tarefas de regressão.

    Arquitetura:
        Input → [Linear → ReLU → Dropout] × n_hidden_layers → Linear (1 saída)

    Args:
        input_dim: Número de features de entrada.
        hidden_dims: Lista com o número de neurônios em cada camada oculta.
            Ex.: [64, 32] gera duas camadas ocultas com 64 e 32 neurônios.
        dropout: Probabilidade de dropout (regularização) entre as camadas.
        output_dim: Número de saídas (padrão 1 para regressão univariada).

    Example:
        >>> import torch
        >>> model = MLPRegressor(input_dim=2, hidden_dims=[16, 8])
        >>> x = torch.randn(5, 2)
        >>> y = model(x)
        >>> y.shape
        torch.Size([5, 1])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

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

        # Camada de saída (sem ativação para regressão)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Guarda os hiperparâmetros para serialização
        self.config = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "output_dim": output_dim,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def count_parameters(self) -> int:
        """Conta o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
