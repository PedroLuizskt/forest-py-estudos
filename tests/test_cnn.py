"""Testes da arquitetura e do trainer da CNN."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from forestpy.ml.cnn import CNNTrainer, SimpleCNN
from forestpy.ml.metrics import accuracy
from forestpy.utils import set_seed


@pytest.fixture(autouse=True)
def fixed_torch_seed():
    set_seed(42)


@pytest.fixture
def synthetic_chips():
    """Dataset pequeno e sintético de chips."""
    rng = np.random.default_rng(42)
    # 3 classes, 30 chips cada, 4 bandas, 16×16
    n_per_class = 30
    X_list, y_list = [], []
    for cls in range(3):
        # Cada classe tem média espectral distinta
        base = np.array([0.1 * (cls + 1), 0.2, 0.15, 0.4 + 0.1 * cls])
        for _ in range(n_per_class):
            chip = np.tile(base[:, None, None], (1, 16, 16))
            chip = chip + rng.normal(0, 0.05, size=chip.shape)
            X_list.append(chip)
            y_list.append(cls)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ──────────────────────────────────────────────────────────
# SimpleCNN
# ──────────────────────────────────────────────────────────
class TestSimpleCNN:
    def test_forward_shape(self):
        model = SimpleCNN(in_channels=4, n_classes=4, chip_size=32)
        x = torch.randn(8, 4, 32, 32)
        logits = model(x)
        assert logits.shape == (8, 4)

    def test_diferentes_chip_sizes(self):
        for size in [16, 24, 32, 48]:
            model = SimpleCNN(in_channels=4, n_classes=3, chip_size=size)
            x = torch.randn(2, 4, size, size)
            assert model(x).shape == (2, 3)

    def test_diferentes_in_channels(self):
        # 1 banda (pancromático), 3 (RGB), 4 (RGB+NIR), 10 (multiespectral)
        for c in [1, 3, 4, 10]:
            model = SimpleCNN(in_channels=c, n_classes=2, chip_size=32)
            x = torch.randn(2, c, 32, 32)
            assert model(x).shape == (2, 2)

    def test_n_classes_invalido_levanta(self):
        with pytest.raises(ValueError, match="n_classes"):
            SimpleCNN(in_channels=4, n_classes=1)

    def test_chip_size_pequeno_demais_levanta(self):
        with pytest.raises(ValueError, match="chip_size"):
            SimpleCNN(in_channels=4, n_classes=2, chip_size=4)

    def test_dropout_invalido_levanta(self):
        with pytest.raises(ValueError, match="dropout"):
            SimpleCNN(in_channels=4, n_classes=2, dropout=1.5)

    def test_count_parameters_positivo(self):
        model = SimpleCNN(in_channels=4, n_classes=4, chip_size=32)
        assert model.count_parameters() > 0

    def test_config_salvo(self):
        model = SimpleCNN(
            in_channels=3, n_classes=5, base_filters=8,
            chip_size=24, dropout=0.2,
        )
        assert model.config["in_channels"] == 3
        assert model.config["n_classes"] == 5
        assert model.config["base_filters"] == 8


# ──────────────────────────────────────────────────────────
# CNNTrainer
# ──────────────────────────────────────────────────────────
class TestCNNTrainer:
    def test_fit_melhora_acuracia(self, synthetic_chips):
        X, y = synthetic_chips
        n_tr = 60
        X_tr, X_va = X[:n_tr], X[n_tr:]
        y_tr, y_va = y[:n_tr], y[n_tr:]

        model = SimpleCNN(in_channels=4, n_classes=3, chip_size=16,
                          base_filters=8)
        trainer = CNNTrainer(model, learning_rate=1e-2)

        acc_inicial = accuracy(y_va, trainer.predict(X_va))
        trainer.fit(X_tr, y_tr, X_va, y_va, epochs=20,
                    batch_size=8, verbose=False)
        acc_final = accuracy(y_va, trainer.predict(X_va))

        # Em dados linearmente separáveis, deve melhorar substancialmente
        assert acc_final > acc_inicial + 0.15

    def test_predict_retorna_inteiros(self, synthetic_chips):
        X, y = synthetic_chips
        model = SimpleCNN(in_channels=4, n_classes=3, chip_size=16,
                          base_filters=8)
        trainer = CNNTrainer(model)
        trainer.fit(X[:60], y[:60], X[60:], y[60:], epochs=3,
                    batch_size=8, verbose=False)
        preds = trainer.predict(X[60:])
        assert preds.dtype in (np.int32, np.int64)

    def test_predict_proba_soma_um(self, synthetic_chips):
        X, y = synthetic_chips
        model = SimpleCNN(in_channels=4, n_classes=3, chip_size=16,
                          base_filters=8)
        trainer = CNNTrainer(model)
        trainer.fit(X[:60], y[:60], X[60:], y[60:], epochs=3,
                    batch_size=8, verbose=False)
        probs = trainer.predict_proba(X[60:])
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_entrada_2d_levanta(self):
        """CNNTrainer rejeita entrada 2D (deve ser MLPClassifierTrainer)."""
        model = SimpleCNN(in_channels=4, n_classes=3, chip_size=16,
                          base_filters=8)
        trainer = CNNTrainer(model)
        X_2d = np.zeros((10, 16), dtype=np.float32)
        y = np.zeros(10, dtype=np.int64)
        with pytest.raises(ValueError, match="4D"):
            trainer.fit(X_2d, y, X_2d, y, epochs=1, verbose=False)
