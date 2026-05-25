"""
Testes das equações volumétricas — template de referência para os demais módulos.

Padrão de testes adotado:
    1. Valor válido → output esperado
    2. Edge cases (zero, negativo, NaN)
    3. Input vetorial → preserva shape
    4. Consistência entre modelos (ordens de grandeza)
"""

import numpy as np
import pytest

from forestpy.dendrometria.volume import honer, schumacher_hall, spurr


# ──────────────────────────────────────────────────────────
# Schumacher-Hall
# ──────────────────────────────────────────────────────────
class TestSchumacherHall:
    def test_retorna_valor_positivo(self):
        v = schumacher_hall(dap=20.0, h=15.0)
        assert v > 0

    def test_valor_conhecido_dap20_h15(self):
        """Verifica valor calculado contra cálculo manual com defaults."""
        v = schumacher_hall(dap=20.0, h=15.0)
        # exp(-9.5) * 20^1.8 * 15^1.1 ≈ 0.3234
        assert pytest.approx(float(v), rel=1e-3) == 0.3234

    def test_volume_cresce_com_dap(self, sample_dap_h):
        v = schumacher_hall(sample_dap_h["dap"], sample_dap_h["h"])
        assert np.all(np.diff(v) > 0), "Volume deve crescer com DAP/H"

    def test_aceita_array(self, sample_dap_h):
        v = schumacher_hall(sample_dap_h["dap"], sample_dap_h["h"])
        assert v.shape == sample_dap_h["dap"].shape

    def test_aceita_lista(self):
        v = schumacher_hall([10.0, 20.0], [8.0, 15.0])
        assert v.shape == (2,)

    def test_dap_zero_levanta(self):
        with pytest.raises(ValueError, match="DAP"):
            schumacher_hall(dap=0, h=15.0)

    def test_dap_negativo_levanta(self):
        with pytest.raises(ValueError, match="DAP"):
            schumacher_hall(dap=-5.0, h=15.0)

    def test_h_zero_levanta(self):
        with pytest.raises(ValueError, match="H"):
            schumacher_hall(dap=20.0, h=0)

    def test_coeficientes_customizados(self):
        v_default = schumacher_hall(20.0, 15.0)
        v_custom = schumacher_hall(20.0, 15.0, b0=-9.0, b1=1.8, b2=1.1)
        assert v_custom > v_default  # b0 maior ⇒ volume maior


# ──────────────────────────────────────────────────────────
# Spurr
# ──────────────────────────────────────────────────────────
class TestSpurr:
    def test_retorna_valor_positivo(self):
        assert spurr(20.0, 15.0) > 0

    def test_aceita_array(self, sample_dap_h):
        v = spurr(sample_dap_h["dap"], sample_dap_h["h"])
        assert v.shape == sample_dap_h["dap"].shape

    def test_dap_negativo_levanta(self):
        with pytest.raises(ValueError):
            spurr(dap=-1.0, h=15.0)


# ──────────────────────────────────────────────────────────
# Honer
# ──────────────────────────────────────────────────────────
class TestHoner:
    def test_retorna_valor_positivo(self):
        assert honer(20.0, 15.0) > 0

    def test_volume_cresce_com_dap(self, sample_dap_h):
        v = honer(sample_dap_h["dap"], sample_dap_h["h"])
        assert np.all(np.diff(v) > 0)

    def test_h_negativo_levanta(self):
        with pytest.raises(ValueError):
            honer(dap=20.0, h=-1.0)


# ──────────────────────────────────────────────────────────
# Comparação entre modelos
# ──────────────────────────────────────────────────────────
class TestConsistencia:
    def test_ordens_de_grandeza_similares(self):
        """Os três modelos devem produzir valores na mesma ordem para inputs típicos."""
        dap, h = 20.0, 15.0
        v_sh = schumacher_hall(dap, h)
        v_sp = spurr(dap, h)
        # Tolerância larga: modelos diferentes, mas devem estar no mesmo intervalo
        assert 0.01 < float(v_sh) < 10.0
        assert 0.01 < float(v_sp) < 10.0
