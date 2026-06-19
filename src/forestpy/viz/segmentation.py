"""Visualizações para segmentação semântica binária."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_segmentation_triptych(
    chip: np.ndarray,
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    title: str = "Segmentação de Copas",
    probs: np.ndarray | None = None,
) -> plt.Figure:
    """
    Visualiza um chip junto com máscara real e predita lado a lado.

    Layout:
        [Chip RGB falsa-cor] [Máscara observada] [Máscara predita ou probs]

    Args:
        chip: Array (C, H, W) — bandas R, G, B, NIR.
        mask_true: Máscara observada (H, W) binária.
        mask_pred: Máscara predita (H, W) binária.
        title: Título da figura.
        probs: Opcional, mapa de probabilidades contínuas (H, W). Se fornecido,
            substitui a máscara predita binária no terceiro painel.

    Returns:
        Figura matplotlib.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # ── 1. Chip em falsa-cor NIR-R-G ──
    if chip.ndim == 3 and chip.shape[0] >= 4:
        rgb_false = np.stack([chip[3], chip[0], chip[1]], axis=-1)
    else:
        rgb_false = np.stack([chip[0], chip[1], chip[2]], axis=-1)
    # Normaliza para visualização
    rgb_false = (rgb_false - rgb_false.min()) / (rgb_false.max() - rgb_false.min() + 1e-8)
    axes[0].imshow(rgb_false)
    axes[0].set_title("Chip (NIR-R-G)", fontweight="bold")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # ── 2. Máscara real ──
    axes[1].imshow(mask_true, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("Máscara observada", fontweight="bold")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # ── 3. Máscara predita ou probabilidades ──
    if probs is not None:
        im = axes[2].imshow(probs, cmap="Greens", vmin=0, vmax=1)
        axes[2].set_title("Probabilidade predita", fontweight="bold")
        plt.colorbar(im, ax=axes[2], shrink=0.7)
    else:
        axes[2].imshow(mask_pred, cmap="Greens", vmin=0, vmax=1)
        axes[2].set_title("Máscara predita", fontweight="bold")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_segmentation_panel(
    chips: np.ndarray,
    masks_true: np.ndarray,
    masks_pred: np.ndarray,
    n_samples: int = 4,
    title: str = "Amostras de Segmentação",
) -> plt.Figure:
    """
    Painel n_samples × 3 com chips, máscaras observadas e máscaras preditas.

    Args:
        chips: Array (n, C, H, W).
        masks_true: Array (n, H, W) binárias.
        masks_pred: Array (n, H, W) binárias.
        n_samples: Número de amostras a exibir (linhas).
        title: Título global.

    Returns:
        Figura matplotlib.
    """
    n = min(n_samples, len(chips))
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n))

    if n == 1:
        axes = axes[None, :]

    for row in range(n):
        chip = chips[row]
        # Chip em falsa-cor NIR-R-G
        if chip.shape[0] >= 4:
            rgb = np.stack([chip[3], chip[0], chip[1]], axis=-1)
        else:
            rgb = np.stack([chip[0], chip[1], chip[2]], axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(masks_true[row], cmap="Greens", vmin=0, vmax=1)
        axes[row, 2].imshow(masks_pred[row], cmap="Greens", vmin=0, vmax=1)

        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

        if row == 0:
            axes[row, 0].set_title("Chip (NIR-R-G)", fontweight="bold")
            axes[row, 1].set_title("Observado", fontweight="bold")
            axes[row, 2].set_title("Predito", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
