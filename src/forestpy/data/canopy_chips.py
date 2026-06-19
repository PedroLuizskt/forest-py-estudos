"""
Gerador de chips raster sintéticos com máscaras de segmentação de copas.

Simula chips de imagem aérea/satélite onde copas individuais aparecem como
elipses orientadas aleatoriamente sobre um fundo de gramínea/solo. Cada chip
vem com uma **máscara binária** indicando os pixels pertencentes a copas
(1 = copa, 0 = fundo).

Este dataset apoia a Sessão 11 (U-Net para segmentação semântica),
permitindo treinar modelos densos pixel-a-pixel com avaliação rigorosa
via métricas IoU (Jaccard) e coeficiente de Dice.

Referências:
    Ronneberger, O.; Fischer, P.; Brox, T. (2015). U-Net: Convolutional
        networks for biomedical image segmentation. *MICCAI*.
    Weinstein, B. G. et al. (2019). Individual tree-crown detection in
        RGB imagery using semi-supervised deep learning. *Remote Sensing*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Perfis espectrais (R, G, B, NIR) — copa de árvore vs fundo gramíneo
# Sobreposição moderada: ambos são vegetação, mas copa tem NIR mais alto
# e cor um pouco mais escura. Tarefa ainda difícil para threshold simples.
_CANOPY_PROFILE = np.array([0.09, 0.17, 0.07, 0.46])
_BACKGROUND_PROFILE = np.array([0.13, 0.24, 0.10, 0.38])

# Variações intra-chip — bastante variabilidade entre chips
_CANOPY_STD = np.array([0.030, 0.040, 0.025, 0.045])
_BACKGROUND_STD = np.array([0.035, 0.045, 0.030, 0.050])

# Ruído pixel-a-pixel
_TEXTURE_NOISE = 0.05


@dataclass
class SegmentationDataset:
    """Dataset de chips para segmentação semântica binária.

    Attributes:
        X: Array (n, 4, h, w) — chips multi-banda.
        Y: Array (n, h, w) — máscaras binárias (0=fundo, 1=copa).
        n_trees_per_chip: Lista com o número de copas em cada chip.
    """

    X: np.ndarray
    Y: np.ndarray
    n_trees_per_chip: list[int]


def _draw_ellipse_mask(
    canvas: np.ndarray,
    cy: float,
    cx: float,
    ry: float,
    rx: float,
    angle_rad: float,
) -> None:
    """Desenha uma elipse rotacionada na máscara, modificando in-place.

    Usa a equação canônica da elipse rotacionada:
        ((dx*cos+dy*sin)/rx)² + ((-dx*sin+dy*cos)/ry)² <= 1
    """
    h, w = canvas.shape
    yy, xx = np.ogrid[:h, :w]
    dx = xx - cx
    dy = yy - cy
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_rot = dx * cos_a + dy * sin_a
    y_rot = -dx * sin_a + dy * cos_a
    mask = (x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1.0
    canvas[mask] = 1.0


def _generate_segmentation_chip(
    rng: np.random.Generator,
    size: int,
    n_trees: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera um par (chip, máscara) com `n_trees` copas elípticas.

    A máscara é construída primeiro como união das elipses; depois o chip
    raster é composto: pixels de copa recebem o perfil espectral de copa,
    pixels de fundo recebem o perfil de fundo, ambos com variação interna.

    Returns:
        chip: array (4, size, size) float32 em [0, 1]
        mask: array (size, size) float32 em {0, 1}
    """
    # ── 1. Construção da máscara: desenha N elipses ──
    mask = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_trees):
        cy = rng.uniform(2, size - 2)
        cx = rng.uniform(2, size - 2)
        ry = rng.uniform(2.5, 5.0)
        rx = rng.uniform(2.5, 5.0)
        angle = rng.uniform(0, np.pi)
        _draw_ellipse_mask(mask, cy, cx, ry, rx, angle)

    # ── 2. Construção do chip: pixels de copa vs fundo ──
    chip = np.zeros((4, size, size), dtype=np.float32)
    canopy_pixels = mask > 0.5
    bg_pixels = ~canopy_pixels

    for b in range(4):
        chip_band = np.zeros((size, size), dtype=np.float32)
        # Pixels de copa
        n_canopy = int(canopy_pixels.sum())
        if n_canopy > 0:
            chip_band[canopy_pixels] = (
                _CANOPY_PROFILE[b]
                + rng.normal(0, _CANOPY_STD[b], size=n_canopy)
            )
        # Pixels de fundo
        n_bg = int(bg_pixels.sum())
        if n_bg > 0:
            chip_band[bg_pixels] = (
                _BACKGROUND_PROFILE[b]
                + rng.normal(0, _BACKGROUND_STD[b], size=n_bg)
            )
        chip[b] = chip_band

    # ── 3. Ruído pixel-a-pixel (textura fina, mesma em todas as bandas) ──
    from scipy.ndimage import gaussian_filter
    noise = rng.normal(0, _TEXTURE_NOISE, size=(4, size, size))
    noise = gaussian_filter(noise, sigma=(0, 0.6, 0.6))
    chip = chip + noise.astype(np.float32)

    return np.clip(chip, 0.0, 1.0), mask


def generate_segmentation_chips(
    n_chips: int = 120,
    chip_size: int = 64,
    trees_range: tuple[int, int] = (3, 10),
    seed: int = 42,
) -> SegmentationDataset:
    """
    Gera dataset de chips raster com máscaras de copas individuais.

    Args:
        n_chips: Número total de chips a gerar.
        chip_size: Lado do chip (chip_size × chip_size pixels).
        trees_range: Intervalo (inclusivo) do número de árvores por chip.
        seed: Semente para reprodutibilidade.

    Returns:
        SegmentationDataset com `X` (n, 4, chip_size, chip_size),
        `Y` (n, chip_size, chip_size) e `n_trees_per_chip`.

    Example:
        >>> ds = generate_segmentation_chips(n_chips=10, chip_size=32, seed=0)
        >>> ds.X.shape
        (10, 4, 32, 32)
        >>> ds.Y.shape
        (10, 32, 32)
    """
    rng = np.random.default_rng(seed)
    X_list, Y_list, n_trees_list = [], [], []

    for _ in range(n_chips):
        n_trees = rng.integers(trees_range[0], trees_range[1] + 1)
        chip, mask = _generate_segmentation_chip(rng, chip_size, n_trees)
        X_list.append(chip)
        Y_list.append(mask)
        n_trees_list.append(int(n_trees))

    return SegmentationDataset(
        X=np.stack(X_list).astype(np.float32),
        Y=np.stack(Y_list).astype(np.float32),
        n_trees_per_chip=n_trees_list,
    )
