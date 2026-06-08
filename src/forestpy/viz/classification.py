"""Visualizações específicas para problemas de classificação."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from forestpy.ml.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: list | None = None,
    normalize: bool = False,
    title: str = "Matriz de Confusão",
    cmap: str = "Greens",
) -> plt.Figure:
    """
    Heatmap da matriz de confusão.

    Args:
        y_true: Rótulos observados.
        y_pred: Rótulos preditos.
        labels: Ordem das classes. Se None, infere automaticamente.
        normalize: Se True, normaliza por linha (recall) para taxa, em %.
        title: Título.
        cmap: Colormap matplotlib.

    Returns:
        Figura matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))

    if normalize:
        soma = cm.sum(axis=1, keepdims=True)
        cm_show = np.where(soma > 0, cm / np.maximum(soma, 1) * 100, 0)
        fmt = ".1f"
        unit = " (%)"
    else:
        cm_show = cm
        fmt = "d"
        unit = ""

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_show, cmap=cmap, vmin=0)

    # Tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Anotações em cada célula
    threshold = cm_show.max() / 2 if cm_show.max() > 0 else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            valor = cm_show[i, j]
            cor = "white" if valor > threshold else "black"
            ax.text(j, i, format(valor, fmt), ha="center", va="center",
                    color=cor, fontsize=12, fontweight="bold")

    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe observada")
    ax.set_title(title + unit, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_class_metrics(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    labels: list,
    title: str = "Métricas por Classe",
) -> plt.Figure:
    """
    Gráfico de barras agrupadas com precision, recall e F1 por classe.

    Args:
        precision: Array com precision por classe.
        recall: Array com recall por classe.
        f1: Array com F1 por classe.
        labels: Nomes das classes.
        title: Título.

    Returns:
        Figura matplotlib.
    """
    x = np.arange(len(labels))
    largura = 0.27

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - largura, precision, largura, label="Precision", color="#2d5016")
    ax.bar(x, recall, largura, label="Recall", color="#7a9a3f")
    ax.bar(x + largura, f1, largura, label="F1", color="#c9a227")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    fig.tight_layout()
    return fig
