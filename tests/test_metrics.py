"""Testes das métricas de avaliação."""

import numpy as np
import pytest

from forestpy.ml.metrics import (
    accuracy,
    bias,
    mae,
    mape,
    r2,
    regression_report,
    rmse,
)


class TestRMSE:
    def test_ajuste_perfeito_zero(self):
        assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_valor_conhecido(self):
        assert pytest.approx(rmse([1.0, 2.0, 3.0], [1.1, 2.1, 2.9]), rel=1e-3) == 0.1

    def test_shapes_incompativeis_levanta(self):
        with pytest.raises(ValueError, match="incompatíveis"):
            rmse([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_array_vazio_levanta(self):
        with pytest.raises(ValueError, match="vazios"):
            rmse([], [])


class TestMAE:
    def test_ajuste_perfeito_zero(self):
        assert mae([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_valor_conhecido(self):
        assert pytest.approx(mae([1.0, 2.0, 3.0], [1.1, 2.1, 2.9]), rel=1e-3) == 0.1


class TestMAPE:
    def test_valor_conhecido(self):
        assert pytest.approx(mape([100.0, 200.0], [110.0, 190.0]), rel=1e-3) == 7.5

    def test_ignora_zeros(self):
        # y_true com zero: o elemento é ignorado, não quebra
        resultado = mape([0.0, 100.0], [5.0, 110.0])
        assert pytest.approx(resultado, rel=1e-3) == 10.0

    def test_todos_zeros_retorna_nan(self):
        assert np.isnan(mape([0.0, 0.0], [1.0, 2.0]))


class TestR2:
    def test_ajuste_perfeito_um(self):
        assert r2([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == 1.0

    def test_predicao_media_zero(self):
        # Prever sempre a média → R² = 0
        y = [1.0, 2.0, 3.0]
        media = [2.0, 2.0, 2.0]
        assert pytest.approx(r2(y, media), abs=1e-9) == 0.0

    def test_variancia_zero_retorna_nan(self):
        assert np.isnan(r2([5.0, 5.0, 5.0], [4.0, 5.0, 6.0]))


class TestBias:
    def test_superestimativa_positiva(self):
        assert pytest.approx(bias([1.0, 2.0, 3.0], [1.1, 2.1, 3.1]), rel=1e-3) == 0.1

    def test_subestimativa_negativa(self):
        assert bias([1.0, 2.0, 3.0], [0.9, 1.9, 2.9]) < 0

    def test_sem_vies_zero(self):
        assert pytest.approx(bias([1.0, 3.0], [0.0, 4.0]), abs=1e-9) == 0.0


class TestAccuracy:
    def test_valor_conhecido(self):
        assert accuracy([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75

    def test_perfeito(self):
        assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0


class TestRegressionReport:
    def test_contem_todas_metricas(self):
        report = regression_report([1.0, 2.0, 3.0], [1.1, 2.0, 2.9])
        assert set(report.keys()) == {"rmse", "mae", "mape", "r2", "bias"}

    def test_valores_sao_float(self):
        report = regression_report([1.0, 2.0], [1.1, 1.9])
        assert all(isinstance(v, float) for v in report.values())
