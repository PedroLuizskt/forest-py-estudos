"""Testes para segmentação: gerador, arquitetura U-Net, métricas, trainer."""

import numpy as np
import pytest

from forestpy.data.canopy_chips import (
    SegmentationDataset,
    generate_segmentation_chips,
)
from forestpy.ml.metrics import (
    dice_score,
    iou_score,
    pixel_accuracy,
    segmentation_report,
)


# ──────────────────────────────────────────────────────────
# Gerador de chips de copas
# ──────────────────────────────────────────────────────────
class TestGenerateSegmentationChips:
    def test_retorna_segmentation_dataset(self):
        ds = generate_segmentation_chips(n_chips=5, chip_size=16, seed=0)
        assert isinstance(ds, SegmentationDataset)

    def test_shapes_corretos(self):
        ds = generate_segmentation_chips(n_chips=10, chip_size=32, seed=0)
        assert ds.X.shape == (10, 4, 32, 32)
        assert ds.Y.shape == (10, 32, 32)

    def test_mascara_binaria(self):
        ds = generate_segmentation_chips(n_chips=5, chip_size=16, seed=0)
        valores = set(np.unique(ds.Y).tolist())
        assert valores.issubset({0.0, 1.0})

    def test_chips_em_0_1(self):
        ds = generate_segmentation_chips(n_chips=5, chip_size=16, seed=0)
        assert ds.X.min() >= 0.0
        assert ds.X.max() <= 1.0

    def test_n_trees_intervalo(self):
        ds = generate_segmentation_chips(
            n_chips=20, chip_size=32, trees_range=(2, 5), seed=0,
        )
        assert all(2 <= n <= 5 for n in ds.n_trees_per_chip)

    def test_reprodutivel(self):
        ds1 = generate_segmentation_chips(n_chips=8, chip_size=16, seed=42)
        ds2 = generate_segmentation_chips(n_chips=8, chip_size=16, seed=42)
        assert np.array_equal(ds1.X, ds2.X)
        assert np.array_equal(ds1.Y, ds2.Y)

    def test_copa_ocupa_fracao_da_imagem(self):
        """Copas devem ocupar fração razoável (não zero, não tudo)."""
        ds = generate_segmentation_chips(n_chips=30, chip_size=64, seed=0)
        fracao_copa = ds.Y.mean()
        assert 0.01 < fracao_copa < 0.5


# ──────────────────────────────────────────────────────────
# Métricas de segmentação
# ──────────────────────────────────────────────────────────
class TestSegmentationMetrics:
    def test_iou_perfeito_um(self):
        y = np.array([[1, 1, 0], [0, 1, 0]])
        assert iou_score(y, y) == pytest.approx(1.0, abs=1e-5)

    def test_iou_sem_sobreposicao_zero(self):
        y_t = np.array([[1, 1], [0, 0]])
        y_p = np.array([[0, 0], [1, 1]])
        assert iou_score(y_t, y_p) < 0.01

    def test_iou_valor_conhecido(self):
        # 1 pixel em comum, 3 na união (TP=1, FP=1, FN=1) → IoU = 1/3
        y_t = np.array([[1, 1, 0]])
        y_p = np.array([[1, 0, 1]])
        assert iou_score(y_t, y_p) == pytest.approx(1/3, abs=1e-3)

    def test_iou_shapes_incompativeis_levanta(self):
        with pytest.raises(ValueError, match="incompatíveis"):
            iou_score(np.zeros(5), np.zeros(3))

    def test_dice_perfeito_um(self):
        y = np.array([[1, 1, 0], [0, 1, 0]])
        assert dice_score(y, y) == pytest.approx(1.0, abs=1e-5)

    def test_dice_maior_ou_igual_iou(self):
        """Para qualquer predição válida, Dice ≥ IoU."""
        rng = np.random.default_rng(0)
        y_t = (rng.random((10, 10)) > 0.5).astype(int)
        y_p = (rng.random((10, 10)) > 0.5).astype(int)
        assert dice_score(y_t, y_p) >= iou_score(y_t, y_p) - 1e-9

    def test_pixel_accuracy_perfeita(self):
        y = np.array([[1, 0], [0, 1]])
        assert pixel_accuracy(y, y) == 1.0

    def test_segmentation_report_chaves(self):
        y = np.array([[1, 0], [1, 1]])
        report = segmentation_report(y, y)
        assert set(report.keys()) == {"iou", "dice", "pixel_acc"}


# ──────────────────────────────────────────────────────────
# U-Net (PyTorch)
# ──────────────────────────────────────────────────────────
torch = pytest.importorskip("torch")

from forestpy.ml.cnn import UNet, UNetTrainer
from forestpy.utils import set_seed


@pytest.fixture(autouse=True)
def seed_fixture():
    set_seed(42)


class TestUNet:
    def test_forward_shape_preservado(self):
        model = UNet(in_channels=4, out_channels=1, base_filters=8, chip_size=32)
        x = torch.randn(2, 4, 32, 32)
        y = model(x)
        assert y.shape == (2, 1, 32, 32)

    def test_multiplos_chip_sizes(self):
        for size in [16, 32, 64]:
            model = UNet(in_channels=4, out_channels=1, base_filters=8,
                         chip_size=size)
            x = torch.randn(1, 4, size, size)
            assert model(x).shape == (1, 1, size, size)

    def test_chip_size_nao_divisivel_por_4_levanta(self):
        with pytest.raises(ValueError, match="divisível por 4"):
            UNet(in_channels=4, out_channels=1, chip_size=30)

    def test_in_channels_invalido_levanta(self):
        with pytest.raises(ValueError, match="in_channels"):
            UNet(in_channels=0, out_channels=1)

    def test_count_parameters_positivo(self):
        model = UNet(in_channels=4, out_channels=1, base_filters=8)
        assert model.count_parameters() > 0

    def test_config_salvo(self):
        model = UNet(in_channels=3, out_channels=2, base_filters=8, chip_size=32)
        assert model.config["in_channels"] == 3
        assert model.config["out_channels"] == 2
        assert model.config["base_filters"] == 8


@pytest.fixture
def small_seg_dataset():
    """Dataset pequeno para testes do trainer."""
    ds = generate_segmentation_chips(n_chips=20, chip_size=16, seed=42)
    return ds.X, ds.Y


class TestUNetTrainer:
    def test_fit_reduz_loss(self, small_seg_dataset):
        X, Y = small_seg_dataset
        X_tr, X_va = X[:15], X[15:]
        Y_tr, Y_va = Y[:15], Y[15:]
        model = UNet(in_channels=4, out_channels=1, base_filters=8, chip_size=16)
        trainer = UNetTrainer(model, learning_rate=1e-2)
        hist = trainer.fit(X_tr, Y_tr, X_va, Y_va, epochs=15,
                           batch_size=4, verbose=False)
        # Loss final < loss inicial
        assert hist.val_loss[-1] < hist.val_loss[0]

    def test_predict_proba_shape_e_intervalo(self, small_seg_dataset):
        X, Y = small_seg_dataset
        model = UNet(in_channels=4, out_channels=1, base_filters=8, chip_size=16)
        trainer = UNetTrainer(model)
        trainer.fit(X[:15], Y[:15], X[15:], Y[15:], epochs=3,
                    batch_size=4, verbose=False)
        probs = trainer.predict_proba(X[15:])
        assert probs.shape == (5, 16, 16)
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_predict_retorna_binario(self, small_seg_dataset):
        X, Y = small_seg_dataset
        model = UNet(in_channels=4, out_channels=1, base_filters=8, chip_size=16)
        trainer = UNetTrainer(model)
        trainer.fit(X[:15], Y[:15], X[15:], Y[15:], epochs=3,
                    batch_size=4, verbose=False)
        pred = trainer.predict(X[15:], threshold=0.5)
        assert set(np.unique(pred).tolist()).issubset({0, 1})

    def test_threshold_alto_predicoes_mais_esparsas(self, small_seg_dataset):
        X, Y = small_seg_dataset
        model = UNet(in_channels=4, out_channels=1, base_filters=8, chip_size=16)
        trainer = UNetTrainer(model)
        trainer.fit(X[:15], Y[:15], X[15:], Y[15:], epochs=3,
                    batch_size=4, verbose=False)
        pred_low = trainer.predict(X[15:], threshold=0.3)
        pred_high = trainer.predict(X[15:], threshold=0.7)
        # Threshold mais alto → menos pixels positivos
        assert pred_high.sum() <= pred_low.sum()

    def test_entrada_X_2D_levanta(self):
        model = UNet(in_channels=4, out_channels=1, base_filters=8, chip_size=16)
        trainer = UNetTrainer(model)
        X = np.zeros((4, 10), dtype=np.float32)
        Y = np.zeros((4, 16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="4D"):
            trainer.fit(X, Y, X, Y, epochs=1, verbose=False)
