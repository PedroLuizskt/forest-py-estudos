"""
U-Net (Ronneberger et al., 2015) para segmentação semântica de imagens raster.

Arquitetura encoder-decoder com *skip connections* — características aprendidas
em níveis crescentes de abstração no encoder são reincorporadas no decoder
nos níveis correspondentes, preservando detalhes espaciais finos necessários
para predição pixel-a-pixel.

Em aplicações florestais, a U-Net é o padrão moderno para:
    - Segmentação de copas individuais em ortomosaicos drone/satélite
    - Mapeamento de áreas queimadas
    - Detecção de desmatamento

Esta implementação usa **2 níveis de pooling** (compacta o suficiente para
treinar em CPU com chips pequenos) e é totalmente parametrizável para
adaptação a outras tarefas.

Referências:
    Ronneberger, O.; Fischer, P.; Brox, T. (2015). U-Net: Convolutional networks
        for biomedical image segmentation. *MICCAI*, LNCS 9351, 234-241.
    Weinstein, B. G. et al. (2019). Individual tree-crown detection in RGB
        imagery using semi-supervised deep learning. *Remote Sensing*, 11(11).
"""

from __future__ import annotations

import torch
from torch import nn


class _DoubleConv(nn.Module):
    """Bloco padrão da U-Net: [Conv 3×3 → BN → ReLU] × 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net compacta para segmentação binária (ou multi-classe) pixel-a-pixel.

    Estrutura:

        Encoder:
            Input (in_ch, H, W)
            → DoubleConv(in_ch → base)            [enc1]
            → MaxPool 2×2
            → DoubleConv(base → 2·base)           [enc2]
            → MaxPool 2×2
            → DoubleConv(2·base → 4·base)         [bottleneck]

        Decoder (com skip connections):
            → Up 2×2 + concat(enc2) → DoubleConv(6·base → 2·base)
            → Up 2×2 + concat(enc1) → DoubleConv(3·base → base)
            → Conv 1×1 (base → out_ch)

    Args:
        in_channels: Número de bandas de entrada (4 para R,G,B,NIR).
        out_channels: Número de classes de saída (1 para segmentação binária).
        base_filters: Filtros do primeiro nível (dobra a cada nível abaixo).
        chip_size: Lado do chip; deve ser divisível por 4 (2 níveis de pool).

    Example:
        >>> import torch
        >>> model = UNet(in_channels=4, out_channels=1, base_filters=16)
        >>> x = torch.randn(2, 4, 64, 64)
        >>> y = model(x)
        >>> y.shape
        torch.Size([2, 1, 64, 64])
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_filters: int = 16,
        chip_size: int = 64,
    ) -> None:
        super().__init__()

        if in_channels < 1:
            raise ValueError(f"in_channels deve ser ≥ 1. Recebido: {in_channels}")
        if out_channels < 1:
            raise ValueError(f"out_channels deve ser ≥ 1. Recebido: {out_channels}")
        if chip_size % 4 != 0:
            raise ValueError(
                f"chip_size deve ser divisível por 4 (2 níveis de pooling). "
                f"Recebido: {chip_size}"
            )

        f1, f2, f3 = base_filters, base_filters * 2, base_filters * 4

        # ── Encoder ──
        self.enc1 = _DoubleConv(in_channels, f1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _DoubleConv(f1, f2)
        self.pool2 = nn.MaxPool2d(2)

        # ── Bottleneck ──
        self.bottleneck = _DoubleConv(f2, f3)

        # ── Decoder ──
        self.up2 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(f3, f2)  # f2 (up) + f2 (skip enc2) = f3 in
        self.up1 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(f2, f1)  # f1 (up) + f1 (skip enc1) = f2 in

        # ── Saída ──
        self.head = nn.Conv2d(f1, out_channels, kernel_size=1)

        self.config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_filters": base_filters,
            "chip_size": chip_size,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            Logits (batch, out_channels, H, W). Para segmentação binária,
            aplicar sigmoid externamente para obter probabilidades.
        """
        # Encoder
        e1 = self.enc1(x)                     # (B, f1, H,   W)
        e2 = self.enc2(self.pool1(e1))        # (B, f2, H/2, W/2)
        b = self.bottleneck(self.pool2(e2))   # (B, f3, H/4, W/4)

        # Decoder com skip connections
        d2 = self.up2(b)                                   # (B, f2, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))         # concat ao longo do canal
        d1 = self.up1(d2)                                  # (B, f1, H, W)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)                               # (B, out_channels, H, W)

    def count_parameters(self) -> int:
        """Conta o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
