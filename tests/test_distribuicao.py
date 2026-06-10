"""Testes do módulo de distribuição diamétrica e ajuste Weibull."""

import numpy as np
import pytest

from forestpy.inventario.distribuicao import (
    WeibullFit,
    diametric_distribution,
    fit_weibull,
)


# ──────────────────────────────────────────────────────────
# fit_weibull
# ──────────────────────────────────────────────────────────
class TestFitWeibull:
    @pytest.fixture
    def weibull_sample(self):
        """Amostra Weibull com parâmetros conhecidos: shape=2.5, scale=15."""
        rng = np.random.default_rng(42)
        return rng.weibull(2.5, size=500) * 15

    def test_retorna_weibullfit(self, weibull_sample):
        fit = fit_weibull(weibull_sample)
        assert isinstance(fit, WeibullFit)

    def test_recupera_shape_aproximado(self, weibull_sample):
        """Ajuste em amostra grande deve recuperar shape ~ 2.5."""
        fit = fit_weibull(weibull_sample)
        assert 2.0 < fit.shape < 3.0

    def test_recupera_scale_aproximado(self, weibull_sample):
        """Ajuste deve recuperar scale ~ 15."""
        fit = fit_weibull(weibull_sample)
        assert 13.0 < fit.scale < 17.0

    def test_n_observations_correto(self, weibull_sample):
        fit = fit_weibull(weibull_sample)
        assert fit.n_observations == len(weibull_sample)

    def test_amostra_pequena_levanta(self):
        with pytest.raises(ValueError, match="5 observações"):
            fit_weibull([10.0, 12.0])

    def test_dap_nao_positivo_levanta(self):
        with pytest.raises(ValueError, match="positivo"):
            fit_weibull([10.0, 0.0, 15.0, 20.0, 25.0])


# ──────────────────────────────────────────────────────────
# WeibullFit.pdf, cdf, expected_counts
# ──────────────────────────────────────────────────────────
class TestWeibullFit:
    @pytest.fixture
    def fit(self):
        return WeibullFit(shape=2.5, scale=15.0, loc=0.0, n_observations=100)

    def test_pdf_positiva(self, fit):
        x = np.array([5.0, 10.0, 15.0, 20.0])
        assert np.all(fit.pdf(x) > 0)

    def test_cdf_monotona_crescente(self, fit):
        x = np.array([5.0, 10.0, 15.0, 20.0, 30.0])
        cdf = fit.cdf(x)
        assert np.all(np.diff(cdf) > 0)

    def test_cdf_entre_0_e_1(self, fit):
        x = np.array([0.1, 5.0, 50.0])
        cdf = fit.cdf(x)
        assert np.all((cdf >= 0) & (cdf <= 1))

    def test_expected_counts_soma_aproxima_total(self, fit):
        """Sobre um intervalo amplo, soma das contagens esperadas ≈ total."""
        edges = np.linspace(0.001, 80.0, 30)
        n_total = 100
        counts = fit.expected_counts(edges, n_total)
        # Pode perder um pouco nas caudas, mas ≥ 90% do total
        assert counts.sum() > 0.9 * n_total


# ──────────────────────────────────────────────────────────
# diametric_distribution
# ──────────────────────────────────────────────────────────
class TestDiametricDistribution:
    def test_retorna_dataframe(self):
        daps = np.array([8.0, 12.0, 17.0, 22.0])
        hist = diametric_distribution(daps, bin_width=5.0)
        import pandas as pd
        assert isinstance(hist, pd.DataFrame)

    def test_colunas_esperadas(self):
        daps = np.array([8.0, 12.0, 17.0, 22.0])
        hist = diametric_distribution(daps, bin_width=5.0)
        assert set(hist.columns) == {"bin_lower", "bin_upper", "bin_center", "count"}

    def test_soma_contagens_iguala_n(self):
        daps = np.array([8.0, 12.0, 14.0, 17.0, 22.0])
        hist = diametric_distribution(daps, bin_width=5.0)
        assert hist["count"].sum() == len(daps)

    def test_bin_width_5_padrao(self):
        daps = np.array([7.0, 12.0, 17.0, 22.0])
        hist = diametric_distribution(daps)  # padrão 5cm
        assert (hist["bin_upper"] - hist["bin_lower"]).iloc[0] == 5.0

    def test_centros_corretos(self):
        daps = np.array([7.0, 12.0])
        hist = diametric_distribution(daps, bin_width=5.0, bin_min=5.0, bin_max=15.0)
        # bins [5, 10) e [10, 15) → centros 7.5 e 12.5
        assert hist["bin_center"].tolist() == [7.5, 12.5]
