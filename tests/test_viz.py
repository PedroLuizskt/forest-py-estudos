"""
Testes das funções de visualização.

Estratégia: validamos que cada função retorna um `matplotlib.Figure`
com a estrutura esperada (número de eixos, labels), sem renderizar
em tela (backend Agg).
"""

import matplotlib

matplotlib.use("Agg")  # Backend não-interativo para CI

import matplotlib.pyplot as plt
import pytest

from forestpy.viz.diagnostics import plot_predicted_vs_observed, plot_residuals
from forestpy.viz.eda import (
    plot_boxplot_by_group,
    plot_correlation_heatmap,
    plot_distribution_grid,
)
from forestpy.viz.style import FOREST_PALETTE, apply_forest_style


class TestStyle:
    def test_palette_tem_6_cores(self):
        assert len(FOREST_PALETTE) == 6

    def test_apply_style_nao_levanta(self):
        apply_forest_style()
        assert plt.rcParams["axes.grid"] is True


class TestEDA:
    def test_distribution_grid_retorna_figure(self, synthetic_pef):
        fig = plot_distribution_grid(synthetic_pef, cols=["dap", "h", "idade"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correlation_heatmap_retorna_figure(self, synthetic_pef):
        fig = plot_correlation_heatmap(synthetic_pef)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_boxplot_retorna_figure(self, synthetic_pef):
        fig = plot_boxplot_by_group(synthetic_pef, value_col="dap", group_col="classe")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestDiagnostics:
    def test_predicted_vs_observed(self, regression_results):
        y_true, y_pred = regression_results
        fig = plot_predicted_vs_observed(y_true, y_pred)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_residuals_panel(self, regression_results):
        y_true, y_pred = regression_results
        fig = plot_residuals(y_true, y_pred)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # painel 2x2
        plt.close(fig)
