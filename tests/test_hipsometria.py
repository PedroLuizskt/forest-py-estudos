"""
Testes das equações hipsométricas (relação altura-diâmetro).

Segue o mesmo padrão de test_dendrometria.py:
    1. Valor válido → output esperado
    2. Edge cases (zero, negativo)
    3. Input vetorial → preserva shape
    4. Comportamento monotônico esperado
"""

import numpy as np
import pytest

from forestpy.dendrometria.hipsometria import curtis, henricksen, stoffels, trorey


# ──────────────────────────────────────────────────────────
# Curtis
# ──────────────────────────────────────────────────────────
class TestCurtis:
    def test_retorna_valor_positivo(self):
        assert curtis(20.0) > 0

    def test_valor_conhecido(self):
        # exp(3.2 - 8/20) = exp(2.8) ≈ 16.44
        assert pytest.approx(float(curtis(20.0)), rel=1e-2) == 16.44

    def test_altura_cresce_com_dap(self):
        dap = np.array([10.0, 20.0, 30.0, 40.0])
        h = curtis(dap)
        assert np.all(np.diff(h) > 0), "Altura deve crescer com DAP"

    def test_aceita_array(self):
        dap = np.array([15.0, 20.0, 25.0])
        h = curtis(dap)
        assert h.shape == dap.shape

    def test_dap_zero_levanta(self):
        with pytest.raises(ValueError, match="DAP"):
            curtis(0)

    def test_dap_negativo_levanta(self):
        with pytest.raises(ValueError, match="DAP"):
            curtis(-10.0)


# ──────────────────────────────────────────────────────────
# Stoffels
# ──────────────────────────────────────────────────────────
class TestStoffels:
    def test_retorna_valor_positivo(self):
        assert stoffels(20.0) > 0

    def test_altura_cresce_com_dap(self):
        dap = np.array([10.0, 20.0, 30.0])
        h = stoffels(dap)
        assert np.all(np.diff(h) > 0)

    def test_aceita_array(self):
        dap = np.array([15.0, 20.0, 25.0])
        assert stoffels(dap).shape == dap.shape

    def test_dap_negativo_levanta(self):
        with pytest.raises(ValueError):
            stoffels(-1.0)


# ──────────────────────────────────────────────────────────
# Henricksen
# ──────────────────────────────────────────────────────────
class TestHenricksen:
    def test_retorna_valor_positivo(self):
        assert henricksen(20.0) > 0

    def test_altura_cresce_com_dap(self):
        dap = np.array([10.0, 20.0, 30.0])
        h = henricksen(dap)
        assert np.all(np.diff(h) > 0)

    def test_dap_zero_levanta(self):
        with pytest.raises(ValueError):
            henricksen(0)


# ──────────────────────────────────────────────────────────
# Trorey
# ──────────────────────────────────────────────────────────
class TestTrorey:
    def test_retorna_valor_positivo(self):
        assert trorey(20.0) > 0

    def test_aceita_array(self):
        dap = np.array([15.0, 20.0, 25.0])
        assert trorey(dap).shape == dap.shape

    def test_concavidade_negativa(self):
        """Com b2 < 0, a parábola tem concavidade para baixo."""
        dap = np.array([10.0, 30.0, 50.0])
        h = trorey(dap, b0=1.3, b1=1.0, b2=-0.015)
        # Segunda diferença deve ser negativa (côncava)
        assert np.diff(h, 2)[0] < 0

    def test_dap_negativo_levanta(self):
        with pytest.raises(ValueError):
            trorey(-5.0)


# ──────────────────────────────────────────────────────────
# Comparação entre modelos
# ──────────────────────────────────────────────────────────
class TestConsistencia:
    def test_alturas_plausiveis(self):
        """Todos os modelos devem prever alturas plausíveis (5–40 m) para DAP=20."""
        for modelo in (curtis, stoffels, henricksen, trorey):
            h = float(modelo(20.0))
            assert 5.0 < h < 40.0, f"{modelo.__name__} previu altura implausível: {h}"
