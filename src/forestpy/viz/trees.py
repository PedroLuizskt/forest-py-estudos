"""Visualizações específicas de mensuração: curvas hipsométricas, perfis de árvore."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_hypsometric_curves(
    df: pd.DataFrame,
    models: dict[str, Callable[[np.ndarray], np.ndarray]],
    dap_col: str = "dap",
    h_col: str = "h",
    title: str = "Curvas Hipsométricas — Observado vs. Modelos",
) -> plt.Figure:
    """
    Sobrepõe a nuvem de pontos (DAP × H) observada com curvas de múltiplos modelos.

    Args:
        df: DataFrame com colunas de DAP e altura observada.
        models: Dicionário {nome: função(dap)->altura}. Cada função recebe um
            array de DAP e retorna alturas estimadas.
        dap_col: Nome da coluna de DAP.
        h_col: Nome da coluna de altura observada.
        title: Título da figura.

    Returns:
        Figura matplotlib.

    Example:
        >>> from forestpy.dendrometria.hipsometria import curtis, stoffels
        >>> models = {"Curtis": curtis, "Stoffels": stoffels}
        >>> # fig = plot_hypsometric_curves(df, models)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Nuvem observada
    ax.scatter(
        df[dap_col],
        df[h_col],
        alpha=0.3,
        s=20,
        color="#5b7b7a",
        edgecolor="white",
        linewidth=0.3,
        label="Observado",
        zorder=1,
    )

    # Curvas dos modelos sobre grade suave de DAP
    dap_grid = np.linspace(df[dap_col].min(), df[dap_col].max(), 200)
    for nome, func in models.items():
        h_pred = func(dap_grid)
        ax.plot(dap_grid, h_pred, lw=2, label=nome, zorder=2)

    ax.set_xlabel("DAP (cm)")
    ax.set_ylabel("Altura (m)")
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=True, fancybox=True)
    fig.tight_layout()
    return fig


def plot_scatter_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """
    Scatter de duas variáveis, opcionalmente colorido por categoria.

    Útil para visualizar relações DAP×H, DAP×Volume, H×Volume etc.

    Args:
        df: DataFrame.
        x_col: Variável do eixo X.
        y_col: Variável do eixo Y.
        hue_col: Coluna categórica para colorir (ex.: 'classe'). Opcional.
        title: Título.

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    if hue_col:
        for categoria, grupo in df.groupby(hue_col):
            ax.scatter(
                grupo[x_col], grupo[y_col],
                alpha=0.5, s=25, edgecolor="white", linewidth=0.3,
                label=f"{hue_col}={categoria}",
            )
        ax.legend(frameon=True)
    else:
        ax.scatter(
            df[x_col], df[y_col],
            alpha=0.5, s=25, edgecolor="white", linewidth=0.3,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{y_col} vs. {x_col}", fontweight="bold")
    fig.tight_layout()
    return fig
