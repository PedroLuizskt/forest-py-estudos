"""
Arquiteturas de Redes Neurais Convolucionais (CNN) em PyTorch.

CNNs são a arquitetura padrão para dados com **estrutura espacial** —
imagens, chips raster, mapas. Diferentemente de MLPs, exploram dois
princípios indutivos fundamentais (LeCun et al., 1998):

    - Localidade: cada filtro só vê uma pequena janela (receptive field)
    - Compartilhamento de pesos: o mesmo filtro varre toda a imagem

Esses dois princípios reduzem drasticamente o número de parâmetros e
codificam invariância translacional — um pinheiro detectado no canto
superior é o mesmo padrão de um pinheiro no canto inferior.

Referências:
    LeCun, Y.; Bottou, L.; Bengio, Y.; Haffner, P. (1998). Gradient-based
        learning applied to document recognition. *Proc. IEEE*, 86(11).
    Krizhevsky, A.; Sutskever, I.; Hinton, G. (2012). ImageNet classification
        with deep convolutional neural networks. *NeurIPS*.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """
    CNN compacta para classificação de chips raster multi-banda.

    Arquitetura (3 blocos convolucionais + classificador denso):

        Input (in_channels, H, W)
        → [Conv 3×3 → BN → ReLU → MaxPool 2×2] × 3
        → Flatten → Linear → ReLU → Dropout → Linear (n_classes)

    BatchNorm (Ioffe & Szegedy, 2015) acelera convergência ao normalizar
    ativações intra-batch. MaxPool reduz a dimensão espacial pela metade
    em cada bloco.

    Args:
        in_channels: Número de bandas da entrada (4 para R,G,B,NIR).
        n_classes: Número de classes-alvo.
        base_filters: Número de filtros no primeiro bloco (dobra a cada bloco).
        chip_size: Lado do chip (chip_size × chip_size). Padrão 32.
        dropout: Dropout no classificador denso.

    Example:
        >>> import torch
        >>> model = SimpleCNN(in_channels=4, n_classes=4, chip_size=32)
        >>> x = torch.randn(8, 4, 32, 32)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([8, 4])
    """

    def __init__(
        self,
        in_channels: int = 4,
        n_classes: int = 4,
        base_filters: int = 16,
        chip_size: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if n_classes < 2:
            raise ValueError(f"n_classes deve ser ≥ 2. Recebido: {n_classes}")
        if chip_size < 8:
            raise ValueError(f"chip_size deve ser ≥ 8. Recebido: {chip_size}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout deve estar em [0, 1). Recebido: {dropout}")

        f1, f2, f3 = base_filters, base_filters * 2, base_filters * 4

        # Três blocos convolucionais
        self.features = nn.Sequential(
            # Bloco 1: in_channels → f1, espacial: chip_size → chip_size/2
            nn.Conv2d(in_channels, f1, kernel_size=3, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Bloco 2: f1 → f2, espacial: /4
            nn.Conv2d(f1, f2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Bloco 3: f2 → f3, espacial: /8
            nn.Conv2d(f2, f3, kernel_size=3, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Tamanho espacial após 3 max-pool 2×2: chip_size / 8
        spatial = chip_size // 8
        if spatial < 1:
            raise ValueError(
                f"chip_size={chip_size} muito pequeno para 3 blocos de pooling."
            )

        flatten_dim = f3 * spatial * spatial

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        self.config = {
            "in_channels": in_channels,
            "n_classes": n_classes,
            "base_filters": base_filters,
            "chip_size": chip_size,
            "dropout": dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: retorna logits (batch_size, n_classes)."""
        x = self.features(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Conta o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
