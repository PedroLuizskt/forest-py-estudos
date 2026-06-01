"""Testes do módulo de amostragem em inventário florestal."""

import numpy as np
import pandas as pd
import pytest

from forestpy.inventario.amostragem import (
    SamplingResult,
    aas,
    estratificada,
    tamanho_amostra,
)


# ──────────────────────────────────────────────────────────
# Amostragem Aleatória Simples
# ──────────────────────────────────────────────────────────
class TestAAS:
    def test_retorna_sampling_result(self):
        res = aas(np.array([100.0, 110.0, 95.0, 105.0, 102.0]))
        assert isinstance(res, SamplingResult)

    def test_metodo_correto(self):
        res = aas(np.array([1.0, 2.0, 3.0, 4.0]))
        assert res.method == "AAS"

    def test_media_estimada(self):
        valores = np.array([100.0, 110.0, 90.0, 105.0, 95.0])
        res = aas(valores)
        assert pytest.approx(res.mean, rel=1e-9) == 100.0

    def test_n_correto(self):
        res = aas(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert res.n == 5

    def test_ic_contem_media(self):
        valores = np.random.normal(100, 15, 50)
        res = aas(valores)
        assert res.ci_lower < res.mean < res.ci_upper

    def test_fpc_reduz_variancia(self):
        valores = np.array([100.0, 110.0, 90.0, 105.0, 95.0, 102.0, 98.0, 103.0])
        # Sem FPC (população infinita)
        res_inf = aas(valores)
        # Com FPC (n=8 de N=20)
        res_fpc = aas(valores, population_size=20)
        # FPC deve reduzir a variância
        assert res_fpc.variance < res_inf.variance

    def test_nivel_confianca_maior_amplia_ic(self):
        valores = np.random.normal(100, 10, 30)
        res_90 = aas(valores, confidence_level=0.90)
        res_99 = aas(valores, confidence_level=0.99)
        margem_90 = res_90.ci_upper - res_90.mean
        margem_99 = res_99.ci_upper - res_99.mean
        assert margem_99 > margem_90

    def test_amostra_pequena_levanta(self):
        with pytest.raises(ValueError, match="2 observações"):
            aas(np.array([100.0]))

    def test_aceita_pandas_series(self):
        s = pd.Series([100.0, 105.0, 95.0, 110.0])
        res = aas(s)
        assert res.n == 4


# ──────────────────────────────────────────────────────────
# Amostragem Estratificada
# ──────────────────────────────────────────────────────────
class TestEstratificada:
    @pytest.fixture
    def df_estratificado(self):
        """DataFrame com 2 estratos contrastantes."""
        np.random.seed(42)
        return pd.DataFrame({
            "volume": np.concatenate([
                np.random.normal(200, 15, 20),
                np.random.normal(120, 18, 20),
            ]),
            "classe": ["I"] * 20 + ["III"] * 20,
        })

    def test_retorna_sampling_result(self, df_estratificado):
        res = estratificada(df_estratificado, "volume", "classe")
        assert isinstance(res, SamplingResult)

    def test_metodo_correto(self, df_estratificado):
        res = estratificada(df_estratificado, "volume", "classe")
        assert res.method == "AE"

    def test_media_entre_estratos(self, df_estratificado):
        """A média estratificada deve estar entre as médias dos estratos."""
        res = estratificada(df_estratificado, "volume", "classe")
        m_I = df_estratificado.loc[df_estratificado["classe"] == "I", "volume"].mean()
        m_III = df_estratificado.loc[df_estratificado["classe"] == "III", "volume"].mean()
        assert min(m_I, m_III) <= res.mean <= max(m_I, m_III)

    def test_reduz_variancia_vs_aas(self, df_estratificado):
        """Quando os estratos diferem, AE deve ter erro-padrão MENOR que AAS."""
        res_ae = estratificada(df_estratificado, "volume", "classe")
        res_aas = aas(df_estratificado["volume"])
        assert res_ae.std_error < res_aas.std_error

    def test_estrato_pequeno_levanta(self):
        df = pd.DataFrame({
            "volume": [100.0, 110.0, 95.0, 200.0],
            "classe": ["I", "I", "I", "III"],  # classe III só tem 1 obs
        })
        with pytest.raises(ValueError, match="apenas 1"):
            estratificada(df, "volume", "classe")

    def test_pesos_customizados_alteram_media(self, df_estratificado):
        """Pesos diferentes devem produzir médias diferentes."""
        res_padrao = estratificada(df_estratificado, "volume", "classe")
        # Peso enviesado: 90% classe I, 10% classe III
        res_custom = estratificada(
            df_estratificado, "volume", "classe",
            pesos={"I": 0.9, "III": 0.1},
        )
        assert res_custom.mean != res_padrao.mean
        # Deve estar mais próxima da média da classe I (peso 0.9)
        m_I = df_estratificado.loc[df_estratificado["classe"] == "I", "volume"].mean()
        assert abs(res_custom.mean - m_I) < abs(res_padrao.mean - m_I)


# ──────────────────────────────────────────────────────────
# Tamanho de amostra
# ──────────────────────────────────────────────────────────
class TestTamanhoAmostra:
    def test_retorna_inteiro(self):
        n = tamanho_amostra(cv_percent=20.0, erro_admissivel_percent=10.0)
        assert isinstance(n, int)

    def test_erro_menor_exige_amostra_maior(self):
        n_relax = tamanho_amostra(cv_percent=20.0, erro_admissivel_percent=15.0)
        n_rigor = tamanho_amostra(cv_percent=20.0, erro_admissivel_percent=5.0)
        assert n_rigor > n_relax

    def test_cv_maior_exige_amostra_maior(self):
        n_baixo_cv = tamanho_amostra(cv_percent=10.0, erro_admissivel_percent=10.0)
        n_alto_cv = tamanho_amostra(cv_percent=30.0, erro_admissivel_percent=10.0)
        assert n_alto_cv > n_baixo_cv

    def test_populacao_finita_reduz_amostra(self):
        n_inf = tamanho_amostra(cv_percent=25.0, erro_admissivel_percent=10.0)
        n_fin = tamanho_amostra(
            cv_percent=25.0, erro_admissivel_percent=10.0, population_size=30
        )
        assert n_fin < n_inf
