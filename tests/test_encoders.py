"""Testes do módulo de codificação categórica."""

import numpy as np
import pandas as pd
import pytest

from forestpy.ml.encoders import OneHotEncoderForest


class TestOneHotEncoderForest:
    @pytest.fixture
    def df_sample(self):
        return pd.DataFrame({
            "classe": ["I", "II", "III", "I", "II"],
            "especie": ["A", "B", "A", "B", "A"],
            "idade": [3, 5, 7, 10, 5],
        })

    def test_categorias_aprendidas(self, df_sample):
        enc = OneHotEncoderForest()
        enc.fit(df_sample, cols=["classe"])
        assert enc.categories_["classe"] == ["I", "II", "III"]

    def test_shape_da_saida(self, df_sample):
        enc = OneHotEncoderForest()
        encoded = enc.fit_transform(df_sample, cols=["classe"])
        assert encoded.shape == (5, 3)

    def test_um_quente_por_linha(self, df_sample):
        """Cada linha deve ter exatamente um '1' (one-hot puro)."""
        enc = OneHotEncoderForest()
        encoded = enc.fit_transform(df_sample, cols=["classe"])
        assert np.all(encoded.sum(axis=1) == 1)

    def test_valores_binarios(self, df_sample):
        enc = OneHotEncoderForest()
        encoded = enc.fit_transform(df_sample, cols=["classe"])
        valores_unicos = set(np.unique(encoded))
        assert valores_unicos.issubset({0.0, 1.0})

    def test_multiplas_colunas(self, df_sample):
        enc = OneHotEncoderForest()
        encoded = enc.fit_transform(df_sample, cols=["classe", "especie"])
        # 3 categorias em classe + 2 em especie = 5 colunas
        assert encoded.shape == (5, 5)

    def test_column_names_corretos(self, df_sample):
        enc = OneHotEncoderForest()
        enc.fit(df_sample, cols=["classe"])
        assert enc.column_names_ == ["classe_I", "classe_II", "classe_III"]

    def test_transform_sem_fit_levanta(self):
        enc = OneHotEncoderForest()
        with pytest.raises(RuntimeError, match="não foi ajustado"):
            enc.transform(pd.DataFrame({"classe": ["I"]}))

    def test_categoria_nao_vista_vira_zeros(self):
        """Categoria nova no transform deve gerar zeros, sem erro."""
        df_train = pd.DataFrame({"classe": ["I", "II"]})
        df_test = pd.DataFrame({"classe": ["I", "IV"]})  # IV é nova

        enc = OneHotEncoderForest()
        enc.fit(df_train, cols=["classe"])
        encoded = enc.transform(df_test)

        # Primeira linha: classe I → [1, 0]
        # Segunda linha: classe IV → [0, 0]
        assert encoded[0].tolist() == [1.0, 0.0]
        assert encoded[1].tolist() == [0.0, 0.0]
