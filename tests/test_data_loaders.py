"""Testes do carregador de dados PEF Vinhedo."""

import pandas as pd
import pytest

from forestpy.data.loaders import load_pef_vinhedo


class TestLoadPefVinhedo:
    def test_synthetic_retorna_dataframe(self):
        df = load_pef_vinhedo(synthetic_fallback=True, n_synthetic=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_synthetic_tem_colunas_esperadas(self):
        df = load_pef_vinhedo(synthetic_fallback=True, n_synthetic=50)
        esperadas = {"parcela", "arvore", "especie", "dap", "h", "h_com", "idade", "classe"}
        assert esperadas.issubset(df.columns)

    def test_synthetic_valores_positivos(self):
        df = load_pef_vinhedo(synthetic_fallback=True, n_synthetic=200)
        assert (df["dap"] > 0).all()
        assert (df["h"] > 0).all()
        assert (df["h_com"] > 0).all()

    def test_synthetic_h_com_menor_que_h(self):
        df = load_pef_vinhedo(synthetic_fallback=True, n_synthetic=200)
        assert (df["h_com"] <= df["h"]).all()

    def test_synthetic_classes_validas(self):
        df = load_pef_vinhedo(synthetic_fallback=True, n_synthetic=200)
        assert set(df["classe"].unique()).issubset({"I", "II", "III"})

    def test_synthetic_reprodutivel(self):
        df1 = load_pef_vinhedo(synthetic_fallback=True, n_synthetic=50, seed=42)
        df2 = load_pef_vinhedo(synthetic_fallback=True, n_synthetic=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_arquivo_inexistente_levanta(self):
        with pytest.raises(FileNotFoundError):
            load_pef_vinhedo(path="/caminho/inexistente.csv", synthetic_fallback=False)
