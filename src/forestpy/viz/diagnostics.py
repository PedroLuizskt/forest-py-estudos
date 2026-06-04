"""Diagnósticos visuais para modelos de regressão e classificação."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def plot_learning_curve(
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    best_epoch: int | None = None,
    title: str = "Curva de Aprendizado",
    log_scale: bool = True,
) -> plt.Figure:
    """
    Curvas de loss de treino e validação por época.

    Diagnóstico essencial para detectar overfitting: quando a loss de treino
    continua caindo mas a de validação sobe, o modelo está memorizando os
    dados em vez de generalizar.

    Args:
        train_loss: Loss de treino por época.
        val_loss: Loss de validação por época.
        best_epoch: Época do melhor modelo (marca verticalmente).
        title: Título.
        log_scale: Se True, eixo Y em escala log (típico para losses).

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    epocas = range(1, len(train_loss) + 1)
    ax.plot(epocas, train_loss, label="Treino", lw=2, color="#2d5016")
    ax.plot(epocas, val_loss, label="Validação", lw=2, color="#a04a2c")

    if best_epoch is not None:
        ax.axvline(best_epoch, ls="--", color="gray", alpha=0.7,
                   label=f"Melhor época ({best_epoch})")

    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title, fontweight="bold")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_predicted_vs_observed(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Predito vs. Observado",
    unit: str = "m³",
) -> plt.Figure:
    """
    Scatter de y_pred contra y_true com linha de identidade (y=x).

    Identifica vieses sistemáticos, heterocedasticidade e outliers.

    Args:
        y_true: Valores observados.
        y_pred: Valores preditos pelo modelo.
        title: Título.
        unit: Unidade dos valores (apenas legenda).

    Returns:
        Figura matplotlib.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolor="white", linewidth=0.5)

    # Linha y = x
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")

    ax.set_xlabel(f"Observado ({unit})")
    ax.set_ylabel(f"Predito ({unit})")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_residuals(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Análise de Resíduos",
) -> plt.Figure:
    """
    Painel 2×2 de diagnóstico de resíduos:

        (1) Resíduos vs. predito  — heterocedasticidade
        (2) QQ-plot dos resíduos  — normalidade
        (3) Histograma de resíduos
        (4) Resíduos vs. índice   — autocorrelação

    Args:
        y_true: Valores observados.
        y_pred: Valores preditos.
        title: Título do painel.

    Returns:
        Figura matplotlib.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    res = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (1) Resíduos vs predito
    axes[0, 0].scatter(y_pred, res, alpha=0.5, edgecolor="white", linewidth=0.5)
    axes[0, 0].axhline(0, color="k", lw=1, ls="--")
    axes[0, 0].set_xlabel("Predito")
    axes[0, 0].set_ylabel("Resíduo")
    axes[0, 0].set_title("Resíduos vs. Predito", fontweight="bold")

    # (2) QQ-plot
    stats.probplot(res, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("QQ-Plot dos Resíduos", fontweight="bold")
    axes[0, 1].get_lines()[0].set_markerfacecolor("#2d5016")
    axes[0, 1].get_lines()[0].set_markeredgecolor("white")

    # (3) Histograma
    axes[1, 0].hist(res, bins=30, edgecolor="white", linewidth=0.5)
    axes[1, 0].axvline(0, color="k", lw=1, ls="--")
    axes[1, 0].set_xlabel("Resíduo")
    axes[1, 0].set_ylabel("Frequência")
    axes[1, 0].set_title("Distribuição dos Resíduos", fontweight="bold")

    # (4) Resíduos vs índice
    axes[1, 1].plot(res, marker="o", ms=3, alpha=0.5, lw=0.5)
    axes[1, 1].axhline(0, color="k", lw=1, ls="--")
    axes[1, 1].set_xlabel("Índice da observação")
    axes[1, 1].set_ylabel("Resíduo")
    axes[1, 1].set_title("Resíduos vs. Índice", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    return fig
