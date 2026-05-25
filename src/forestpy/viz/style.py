"""
Tema visual 'forest' para matplotlib/seaborn.

Aplica uma paleta de cores e configurações tipográficas consistentes em
todas as figuras do projeto, dando identidade visual aos relatórios.
"""

import matplotlib.pyplot as plt

# Paleta inspirada em tons florestais brasileiros
FOREST_PALETTE = [
    "#2d5016",  # verde mata fechada
    "#7a9a3f",  # verde campo
    "#c9a227",  # amarelo savana
    "#a04a2c",  # marrom terra
    "#5b7b7a",  # cinza azulado
    "#3e2723",  # marrom escuro
]


def apply_forest_style() -> None:
    """
    Ativa o tema 'forest' globalmente.

    Configura cores, fontes e grid para todas as figuras subsequentes.
    Recomenda-se chamar uma vez no início do notebook.
    """
    plt.rcParams.update(
        {
            # ── Tipografia ──
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            # ── Cores ──
            "axes.prop_cycle": plt.cycler(color=FOREST_PALETTE),
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "text.color": "#333333",
            # ── Grid ──
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            # ── Layout ──
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
