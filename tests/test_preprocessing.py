"""Testes do módulo de pré-processamento."""

import numpy as np
import pandas as pd
import pytest

from forestpy.ml.preprocessing import StandardScalerForest


class TestStandardScalerForest:
    def test_fit_transform_zera_media(self):
        X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
        scaler = StandardScalerForest()
        X_scaled = scaler.fit_transform(X)
        # Média deve ser ~0 após normalização
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-9)

    def test_fit_transform_unitariza_desvio(self):
        rng = np.random.default_rng(42)
        X = rng.normal(50, 15, size=(100, 3))
        scaler = StandardScalerForest()
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-9)

    def test_inverse_transform_recupera_original(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 100, size=(50, 4))
        scaler = StandardScalerForest()
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)
        assert np.allclose(X_recovered, X, atol=1e-9)

    def test_transform_sem_fit_levanta(self):
        scaler = StandardScalerForest()
        with pytest.raises(RuntimeError, match="não foi ajustado"):
            scaler.transform(np.array([[1.0, 2.0]]))

    def test_aceita_dataframe_preserva_nomes(self):
        df = pd.DataFrame({"dap": [10.0, 20.0, 30.0], "h": [5.0, 10.0, 15.0]})
        scaler = StandardScalerForest()
        scaler.fit(df)
        assert scaler.feature_names_ == ["dap", "h"]

    def test_feature_constante_nao_quebra(self):
        """Feature com desvio zero não pode causar divisão por zero."""
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        scaler = StandardScalerForest()
        X_scaled = scaler.fit_transform(X)
        # A coluna constante vira zero (sem NaN/Inf)
        assert np.all(np.isfinite(X_scaled))

    def test_transform_treino_e_teste_consistente(self):
        """Parâmetros do treino devem ser aplicados ao teste sem refazer fit."""
        rng = np.random.default_rng(0)
        X_train = rng.normal(50, 10, size=(80, 2))
        X_test = rng.normal(50, 10, size=(20, 2))

        scaler = StandardScalerForest()
        scaler.fit(X_train)
        mean_treino = scaler.mean_.copy()

        # transform no teste não deve alterar parâmetros do scaler
        _ = scaler.transform(X_test)
        assert np.array_equal(scaler.mean_, mean_treino)
