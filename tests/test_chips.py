"""Testes do gerador de chips raster sintéticos."""

import numpy as np
import pytest

from forestpy.data.chips import (
    CLASS_NAMES,
    ChipsDataset,
    compute_ndvi,
    generate_chips,
)


class TestGenerateChips:
    def test_retorna_chipsdataset(self):
        ds = generate_chips(n_per_class=5, chip_size=16, seed=0)
        assert isinstance(ds, ChipsDataset)

    def test_shape_correto(self):
        ds = generate_chips(n_per_class=10, chip_size=32, seed=0)
        assert ds.X.shape == (40, 4, 32, 32)
        assert ds.y.shape == (40,)

    def test_balanceamento(self):
        ds = generate_chips(n_per_class=15, chip_size=16, seed=0)
        counts = np.bincount(ds.y)
        assert counts.tolist() == [15, 15, 15, 15]

    def test_4_classes(self):
        ds = generate_chips(n_per_class=5, chip_size=16, seed=0)
        assert set(ds.y.tolist()) == {0, 1, 2, 3}
        assert len(ds.class_names) == 4

    def test_valores_em_0_1(self):
        ds = generate_chips(n_per_class=5, chip_size=16, seed=0)
        assert ds.X.min() >= 0.0
        assert ds.X.max() <= 1.0

    def test_reprodutivel(self):
        ds1 = generate_chips(n_per_class=8, chip_size=16, seed=42)
        ds2 = generate_chips(n_per_class=8, chip_size=16, seed=42)
        assert np.array_equal(ds1.X, ds2.X)
        assert np.array_equal(ds1.y, ds2.y)

    def test_seed_diferente_produz_dados_diferentes(self):
        ds1 = generate_chips(n_per_class=8, chip_size=16, seed=0)
        ds2 = generate_chips(n_per_class=8, chip_size=16, seed=1)
        assert not np.array_equal(ds1.X, ds2.X)

    def test_classes_tem_perfis_espectrais_distintos(self):
        """Médias espectrais por classe devem ser distinguíveis."""
        ds = generate_chips(n_per_class=30, chip_size=32, seed=0)
        # Média da banda NIR por classe
        nir_means = [ds.X[ds.y == c, 3].mean() for c in range(4)]
        # Esperamos pelo menos 0.05 de diferença entre máximo e mínimo
        assert max(nir_means) - min(nir_means) > 0.05


class TestComputeNDVI:
    def test_chip_unico(self):
        ds = generate_chips(n_per_class=5, chip_size=16, seed=0)
        ndvi = compute_ndvi(ds.X[0])
        assert ndvi.shape == (16, 16)

    def test_batch(self):
        ds = generate_chips(n_per_class=5, chip_size=16, seed=0)
        ndvi = compute_ndvi(ds.X)
        assert ndvi.shape == (20, 16, 16)

    def test_intervalo_valido(self):
        """NDVI deve estar em [-1, 1]."""
        ds = generate_chips(n_per_class=10, chip_size=16, seed=0)
        ndvi = compute_ndvi(ds.X)
        assert ndvi.min() >= -1.001
        assert ndvi.max() <= 1.001

    def test_dimensoes_invalidas_levanta(self):
        x = np.zeros((4, 16))  # 2D
        with pytest.raises(ValueError, match="3 ou 4 dimensões"):
            compute_ndvi(x)


def test_class_names_exportadas():
    assert len(CLASS_NAMES) == 4
    assert "Eucalipto" in CLASS_NAMES
