"""Funções de visualização para Análise Exploratória de Dados (EDA)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_distribution_grid(
    df: pd.DataFrame,
    cols: list[str],
    ncols: int = 3,
    title: str | None = None,
) -> plt.Figure:
    """
    Grid de histogramas + KDE para variáveis numéricas.

    Args:
        df: DataFrame.
        cols: Colunas numéricas a plotar.
        ncols: Colunas do grid.
        title: Título geral (suptitle).

    Returns:
        Figura matplotlib.
    """
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for ax, col in zip(axes, cols, strict=False):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, edgecolor="white", linewidth=0.5)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")

    # Limpa eixos sobrando
    for ax in axes[len(cols) :]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    title: str = "Matriz de Correlação",
) -> plt.Figure:
    """
    Heatmap de correlação para variáveis numéricas.

    Args:
        df: DataFrame.
        method: 'pearson', 'spearman' ou 'kendall'.
        title: Título da figura.

    Returns:
        Figura matplotlib.
    """
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_boxplot_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    title: str | None = None,
) -> plt.Figure:
    """Boxplot de uma variável numérica agrupada por categoria."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x=group_col, y=value_col, ax=ax)
    ax.set_title(title or f"{value_col} por {group_col}", fontweight="bold")
    fig.tight_layout()
    return fig
