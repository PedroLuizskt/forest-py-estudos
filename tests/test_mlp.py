"""
Testes do módulo MLP (PyTorch).

Estratégia: testes leves (épocas curtas, redes pequenas) que validam a
estrutura e o comportamento esperado, sem rodar treinamentos longos.
"""

import numpy as np
import pytest

# Pula todos os testes se PyTorch não estiver disponível
torch = pytest.importorskip("torch")

from forestpy.ml.mlp import EarlyStopping, MLPRegressor, MLPTrainer
from forestpy.utils import set_seed


@pytest.fixture(autouse=True)
def fixed_torch_seed():
    set_seed(42)


@pytest.fixture
def regression_data():
    """Dataset linear simples: y = 2*x1 + 3*x2 + ruído."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.uniform(0, 10, size=(n, 2)).astype(np.float32)
    y = (2 * X[:, 0] + 3 * X[:, 1] + rng.normal(0, 0.5, n)).astype(np.float32)
    return X, y


# ──────────────────────────────────────────────────────────
# MLPRegressor
# ──────────────────────────────────────────────────────────
class TestMLPRegressor:
    def test_forward_shape_correto(self):
        model = MLPRegressor(input_dim=2, hidden_dims=[16, 8])
        x = torch.randn(5, 2)
        y = model(x)
        assert y.shape == (5, 1)

    def test_count_parameters_positivo(self):
        model = MLPRegressor(input_dim=2, hidden_dims=[16, 8])
        assert model.count_parameters() > 0

    def test_config_salvo(self):
        model = MLPRegressor(input_dim=3, hidden_dims=[32], dropout=0.2)
        assert model.config["input_dim"] == 3
        assert model.config["hidden_dims"] == [32]
        assert model.config["dropout"] == 0.2

    def test_dropout_invalido_levanta(self):
        with pytest.raises(ValueError, match="dropout"):
            MLPRegressor(input_dim=2, hidden_dims=[8], dropout=1.0)

    def test_hidden_dims_vazio_levanta(self):
        with pytest.raises(ValueError, match="hidden_dims"):
            MLPRegressor(input_dim=2, hidden_dims=[])

    def test_output_dim_customizado(self):
        model = MLPRegressor(input_dim=2, hidden_dims=[8], output_dim=3)
        x = torch.randn(4, 2)
        assert model(x).shape == (4, 3)


# ──────────────────────────────────────────────────────────
# EarlyStopping
# ──────────────────────────────────────────────────────────
class TestEarlyStopping:
    def test_nao_para_com_melhoria_continua(self):
        es = EarlyStopping(patience=3)
        for loss in [1.0, 0.8, 0.6, 0.4, 0.2]:
            assert not es(loss)
        assert not es.should_stop

    def test_para_apos_patience_sem_melhoria(self):
        es = EarlyStopping(patience=2)
        assert not es(1.0)   # primeira: best
        assert not es(1.1)   # piora (counter=1)
        assert es(1.2)       # piora (counter=2 == patience)
        assert es.should_stop


# ──────────────────────────────────────────────────────────
# MLPTrainer
# ──────────────────────────────────────────────────────────
class TestMLPTrainer:
    def test_fit_reduz_loss(self, regression_data):
        X, y = regression_data
        X_tr, X_va = X[:160], X[160:]
        y_tr, y_va = y[:160], y[160:]

        model = MLPRegressor(input_dim=2, hidden_dims=[16, 8])
        trainer = MLPTrainer(model, learning_rate=1e-2)
        hist = trainer.fit(X_tr, y_tr, X_va, y_va, epochs=50, verbose=False)

        # Loss final deve ser MUITO menor que inicial
        assert hist.val_loss[-1] < hist.val_loss[0]

    def test_predict_shape(self, regression_data):
        X, y = regression_data
        model = MLPRegressor(input_dim=2, hidden_dims=[8])
        trainer = MLPTrainer(model, learning_rate=1e-2)
        trainer.fit(X[:160], y[:160], X[160:], y[160:],
                    epochs=10, verbose=False)
        y_pred = trainer.predict(X[160:])
        assert y_pred.shape == (40,)

    def test_history_registra_todas_epocas(self, regression_data):
        X, y = regression_data
        model = MLPRegressor(input_dim=2, hidden_dims=[8])
        trainer = MLPTrainer(model)
        hist = trainer.fit(X[:160], y[:160], X[160:], y[160:],
                           epochs=15, patience=100, verbose=False)
        assert len(hist.train_loss) == 15
        assert len(hist.val_loss) == 15

    def test_best_epoch_registrado(self, regression_data):
        X, y = regression_data
        model = MLPRegressor(input_dim=2, hidden_dims=[8])
        trainer = MLPTrainer(model, learning_rate=1e-2)
        hist = trainer.fit(X[:160], y[:160], X[160:], y[160:],
                           epochs=30, verbose=False)
        assert 1 <= hist.best_epoch <= len(hist.train_loss)

    def test_early_stopping_funciona(self, regression_data):
        """Com patience=2 e learning_rate=0, treino deve parar cedo."""
        X, y = regression_data
        model = MLPRegressor(input_dim=2, hidden_dims=[8])
        trainer = MLPTrainer(model, learning_rate=0.0)  # não aprende
        hist = trainer.fit(X[:160], y[:160], X[160:], y[160:],
                           epochs=100, patience=2, verbose=False)
        # Deve parar bem antes das 100 épocas
        assert len(hist.train_loss) < 50
