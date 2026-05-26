"""Testes do módulo de ajuste de modelos dendrométricos."""

import numpy as np
import pytest

from forestpy.dendrometria.fitting import FitResult, compare_models, fit_model


class TestFitModel:
    def test_schumacher_hall_alto_r2(self, synthetic_pef):
        """Ajuste de Schumacher-Hall a dados gerados por ela deve ter R² alto."""
        res = fit_model("schumacher_hall", synthetic_pef["volume"],
                        synthetic_pef["dap"], synthetic_pef["h"])
        assert res.metrics["r2"] > 0.95

    def test_recupera_coeficientes_geradores(self, synthetic_pef):
        """Coeficientes ajustados devem aproximar os usados na geração (-9.5, 1.8, 1.1)."""
        res = fit_model("schumacher_hall", synthetic_pef["volume"],
                        synthetic_pef["dap"], synthetic_pef["h"])
        assert res.coefficients["b1"] == pytest.approx(1.8, abs=0.2)
        assert res.coefficients["b2"] == pytest.approx(1.1, abs=0.2)

    def test_retorna_fitresult(self, synthetic_pef):
        res = fit_model("spurr", synthetic_pef["volume"],
                        synthetic_pef["dap"], synthetic_pef["h"])
        assert isinstance(res, FitResult)

    def test_ypred_mesmo_tamanho(self, synthetic_pef):
        res = fit_model("curtis", synthetic_pef["h"], synthetic_pef["dap"])
        assert len(res.y_pred) == len(synthetic_pef)

    def test_metrics_completas(self, synthetic_pef):
        res = fit_model("stoffels", synthetic_pef["h"], synthetic_pef["dap"])
        assert set(res.metrics.keys()) == {"rmse", "mae", "mape", "r2", "bias"}

    def test_modelo_desconhecido_levanta(self, synthetic_pef):
        with pytest.raises(ValueError, match="desconhecido"):
            fit_model("modelo_inexistente", synthetic_pef["volume"], synthetic_pef["dap"])

    def test_volumetrico_sem_altura_levanta(self, synthetic_pef):
        with pytest.raises(ValueError, match="altura"):
            fit_model("schumacher_hall", synthetic_pef["volume"], synthetic_pef["dap"])

    def test_summary_retorna_string(self, synthetic_pef):
        res = fit_model("curtis", synthetic_pef["h"], synthetic_pef["dap"])
        s = res.summary()
        assert isinstance(s, str)
        assert "curtis" in s


class TestCompareModels:
    def test_retorna_lista_ordenada_por_rmse(self, synthetic_pef):
        resultados = compare_models(
            ["schumacher_hall", "spurr"],
            synthetic_pef["volume"], synthetic_pef["dap"], synthetic_pef["h"],
        )
        rmses = [r.metrics["rmse"] for r in resultados]
        assert rmses == sorted(rmses), "Resultados devem estar ordenados por RMSE crescente"

    def test_numero_de_resultados(self, synthetic_pef):
        resultados = compare_models(
            ["curtis", "stoffels"],
            synthetic_pef["h"], synthetic_pef["dap"],
        )
        assert len(resultados) == 2
