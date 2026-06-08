"""Testes das métricas e do classificador MLP."""

import numpy as np
import pytest

from forestpy.ml.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    f1_per_class,
    precision_per_class,
    recall_per_class,
)


# ──────────────────────────────────────────────────────────
# Métricas de classificação
# ──────────────────────────────────────────────────────────
class TestConfusionMatrix:
    def test_perfeito_diagonal(self):
        cm = confusion_matrix([0, 1, 2, 0], [0, 1, 2, 0], labels=[0, 1, 2])
        assert cm[0, 0] == 2  # duas classes 0 corretas
        assert cm[1, 1] == 1
        assert cm[2, 2] == 1
        # Fora da diagonal: zero
        assert cm.sum() - np.trace(cm) == 0

    def test_shape_correto(self):
        cm = confusion_matrix([0, 1, 2], [0, 1, 2], labels=[0, 1, 2])
        assert cm.shape == (3, 3)

    def test_labels_inferidos(self):
        cm = confusion_matrix([0, 1, 2], [0, 1, 1])
        assert cm.shape == (3, 3)

    def test_erro_fora_diagonal(self):
        # 0 previsto como 1
        cm = confusion_matrix([0, 0], [0, 1], labels=[0, 1])
        assert cm[0, 0] == 1  # acerto
        assert cm[0, 1] == 1  # erro: classe 0 prevista como 1


class TestPrecisionRecallF1:
    def test_modelo_perfeito(self):
        y_t = [0, 1, 2, 0, 1, 2]
        y_p = [0, 1, 2, 0, 1, 2]
        assert np.allclose(precision_per_class(y_t, y_p, labels=[0, 1, 2]), 1.0)
        assert np.allclose(recall_per_class(y_t, y_p, labels=[0, 1, 2]), 1.0)
        assert np.allclose(f1_per_class(y_t, y_p, labels=[0, 1, 2]), 1.0)

    def test_classe_nunca_predita(self):
        """Se classe 2 nunca é predita, precision dela = 0."""
        y_t = [0, 0, 1, 1, 2, 2]
        y_p = [0, 0, 1, 1, 1, 0]
        p = precision_per_class(y_t, y_p, labels=[0, 1, 2])
        assert p[2] == 0.0

    def test_classe_sempre_predita(self):
        """Modelo prevendo sempre 0: precision[0]=acerto, demais=0."""
        y_t = [0, 1, 2, 0]
        y_p = [0, 0, 0, 0]
        r = recall_per_class(y_t, y_p, labels=[0, 1, 2])
        # Classe 0: 2/2 acertos = recall 1.0
        assert r[0] == 1.0
        # Classes 1 e 2: nunca preditas → recall = 0
        assert r[1] == 0.0
        assert r[2] == 0.0


class TestClassificationReport:
    def test_chaves_completas(self):
        report = classification_report([0, 1, 1, 0], [0, 1, 0, 0])
        assert set(report.keys()) == {
            "accuracy", "precision_macro", "recall_macro", "f1_macro"
        }

    def test_modelo_perfeito_todos_um(self):
        report = classification_report([0, 1, 2], [0, 1, 2])
        for v in report.values():
            assert v == 1.0


# ──────────────────────────────────────────────────────────
# MLPClassifier (PyTorch)
# ──────────────────────────────────────────────────────────
torch = pytest.importorskip("torch")
from forestpy.ml.mlp import MLPClassifier, MLPClassifierTrainer
from forestpy.utils import set_seed


@pytest.fixture(autouse=True)
def fixed_torch_seed():
    set_seed(42)


@pytest.fixture
def classification_data():
    """Dataset multiclasse simples: 3 classes, 4 features."""
    rng = np.random.default_rng(42)
    n_per_class = 80
    X_list = []
    y_list = []
    for cls in range(3):
        centro = np.array([cls * 3.0, cls * 2.0, -cls, 1.0])
        X_list.append(rng.normal(centro, 1.0, size=(n_per_class, 4)))
        y_list.append(np.full(n_per_class, cls))
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list)
    # Embaralha
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


class TestMLPClassifier:
    def test_forward_shape(self):
        model = MLPClassifier(input_dim=4, hidden_dims=[16, 8], n_classes=3)
        x = torch.randn(5, 4)
        logits = model(x)
        assert logits.shape == (5, 3)

    def test_count_parameters_positivo(self):
        model = MLPClassifier(input_dim=4, hidden_dims=[16], n_classes=3)
        assert model.count_parameters() > 0

    def test_n_classes_invalido_levanta(self):
        with pytest.raises(ValueError, match="n_classes"):
            MLPClassifier(input_dim=4, hidden_dims=[8], n_classes=1)

    def test_config_salvo(self):
        model = MLPClassifier(input_dim=4, hidden_dims=[16], n_classes=3, dropout=0.2)
        assert model.config["n_classes"] == 3
        assert model.config["dropout"] == 0.2


class TestMLPClassifierTrainer:
    def test_fit_melhora_acuracia(self, classification_data):
        X, y = classification_data
        n_tr = 200
        X_tr, X_va = X[:n_tr], X[n_tr:]
        y_tr, y_va = y[:n_tr], y[n_tr:]

        model = MLPClassifier(input_dim=4, hidden_dims=[16, 8], n_classes=3)
        trainer = MLPClassifierTrainer(model, learning_rate=1e-2)

        # Acurácia antes do treino (chute aleatório)
        acc_inicial = accuracy(y_va, trainer.predict(X_va))

        trainer.fit(X_tr, y_tr, X_va, y_va, epochs=30, verbose=False)
        acc_final = accuracy(y_va, trainer.predict(X_va))

        # Após treinar em dados separáveis, deve melhorar substancialmente
        assert acc_final > acc_inicial + 0.2

    def test_predict_retorna_inteiros(self, classification_data):
        X, y = classification_data
        model = MLPClassifier(input_dim=4, hidden_dims=[8], n_classes=3)
        trainer = MLPClassifierTrainer(model, learning_rate=1e-2)
        trainer.fit(X[:200], y[:200], X[200:], y[200:], epochs=5, verbose=False)
        preds = trainer.predict(X[200:])
        assert preds.dtype in (np.int32, np.int64)

    def test_predict_proba_soma_um(self, classification_data):
        X, y = classification_data
        model = MLPClassifier(input_dim=4, hidden_dims=[8], n_classes=3)
        trainer = MLPClassifierTrainer(model)
        trainer.fit(X[:200], y[:200], X[200:], y[200:], epochs=5, verbose=False)
        probs = trainer.predict_proba(X[200:])
        # Cada linha (uma instância) tem probabilidades somando 1
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_shape(self, classification_data):
        X, y = classification_data
        model = MLPClassifier(input_dim=4, hidden_dims=[8], n_classes=3)
        trainer = MLPClassifierTrainer(model)
        trainer.fit(X[:200], y[:200], X[200:], y[200:], epochs=5, verbose=False)
        probs = trainer.predict_proba(X[200:])
        assert probs.shape == (X[200:].shape[0], 3)
