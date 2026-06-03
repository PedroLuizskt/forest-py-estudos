"""Testes do módulo de avaliação (k-fold cross-validation e bootstrap)."""

import numpy as np
import pandas as pd
import pytest

from forestpy.ml.evaluation import CVResult, bootstrap_metric, kfold_cv
from forestpy.ml.metrics import rmse


# ──────────────────────────────────────────────────────────
# K-Fold Cross-Validation
# ──────────────────────────────────────────────────────────
class TestKFoldCV:
    @pytest.fixture
    def dataset(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.uniform(0, 10, size=(n, 2))
        # y linear: y = 2*x0 + 3*x1 + ruído
        y = 2 * X[:, 0] + 3 * X[:, 1] + rng.normal(0, 0.5, n)
        return X, y

    def _simple_linear(self, X_train, y_train, X_test):
        """Regressão linear simples via numpy.linalg."""
        X_with_bias = np.hstack([np.ones((len(X_train), 1)), X_train])
        coef, *_ = np.linalg.lstsq(X_with_bias, y_train, rcond=None)
        X_test_b = np.hstack([np.ones((len(X_test), 1)), X_test])
        return None, X_test_b @ coef

    def test_retorna_cvresult(self, dataset):
        X, y = dataset
        cv = kfold_cv(self._simple_linear, X, y, n_splits=5)
        assert isinstance(cv, CVResult)

    def test_n_splits_correto(self, dataset):
        X, y = dataset
        cv = kfold_cv(self._simple_linear, X, y, n_splits=5)
        assert cv.n_splits == 5
        assert len(cv.fold_metrics) == 5

    def test_mean_metrics_completas(self, dataset):
        X, y = dataset
        cv = kfold_cv(self._simple_linear, X, y, n_splits=5)
        esperadas = {"rmse", "mae", "mape", "r2", "bias"}
        assert set(cv.mean_metrics.keys()) == esperadas
        assert set(cv.std_metrics.keys()) == esperadas

    def test_to_dataframe(self, dataset):
        X, y = dataset
        cv = kfold_cv(self._simple_linear, X, y, n_splits=5)
        df = cv.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "fold" in df.columns

    def test_summary_retorna_string(self, dataset):
        X, y = dataset
        cv = kfold_cv(self._simple_linear, X, y, n_splits=5, model_name="teste")
        s = cv.summary()
        assert isinstance(s, str)
        assert "teste" in s

    def test_modelo_perfeito_r2_alto(self, dataset):
        """Regressão linear em dados quase-lineares deve dar R² alto."""
        X, y = dataset
        cv = kfold_cv(self._simple_linear, X, y, n_splits=5)
        assert cv.mean_metrics["r2"] > 0.95

    def test_aceita_dataframe_e_series(self, dataset):
        X, y = dataset
        df_X = pd.DataFrame(X, columns=["x1", "x2"])
        s_y = pd.Series(y)
        cv = kfold_cv(self._simple_linear, df_X, s_y, n_splits=5)
        assert isinstance(cv, CVResult)

    def test_reprodutivel_com_mesmo_seed(self, dataset):
        X, y = dataset
        cv1 = kfold_cv(self._simple_linear, X, y, n_splits=5, random_state=42)
        cv2 = kfold_cv(self._simple_linear, X, y, n_splits=5, random_state=42)
        assert cv1.mean_metrics["rmse"] == cv2.mean_metrics["rmse"]


# ──────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────
class TestBootstrap:
    @pytest.fixture
    def regression_data(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(0.5, 2.0, size=100)
        y_pred = y_true + rng.normal(0, 0.1, size=100)
        return y_true, y_pred

    def test_retorna_dict_correto(self, regression_data):
        y_t, y_p = regression_data
        res = bootstrap_metric(y_t, y_p, rmse, n_bootstrap=500)
        assert set(res.keys()) == {"mean", "std", "ci_lower", "ci_upper"}

    def test_ic_contem_mean(self, regression_data):
        y_t, y_p = regression_data
        res = bootstrap_metric(y_t, y_p, rmse, n_bootstrap=500)
        assert res["ci_lower"] <= res["mean"] <= res["ci_upper"]

    def test_nivel_confianca_maior_amplia_ic(self, regression_data):
        y_t, y_p = regression_data
        res_90 = bootstrap_metric(y_t, y_p, rmse, n_bootstrap=500, confidence_level=0.90)
        res_99 = bootstrap_metric(y_t, y_p, rmse, n_bootstrap=500, confidence_level=0.99)
        amp_90 = res_90["ci_upper"] - res_90["ci_lower"]
        amp_99 = res_99["ci_upper"] - res_99["ci_lower"]
        assert amp_99 > amp_90

    def test_reprodutivel(self, regression_data):
        y_t, y_p = regression_data
        r1 = bootstrap_metric(y_t, y_p, rmse, n_bootstrap=300, random_state=42)
        r2 = bootstrap_metric(y_t, y_p, rmse, n_bootstrap=300, random_state=42)
        assert r1["mean"] == r2["mean"]
