"""
Treinador para classificadores MLP em PyTorch.

Diferenças centrais em relação ao `MLPTrainer` (regressão):
    - Loss: `CrossEntropyLoss` em vez de `MSELoss`
    - Labels: `torch.long` (índices de classe) em vez de `torch.float32`
    - `predict()` retorna índices de classe via `argmax`
    - `predict_proba()` retorna probabilidades softmax por classe
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from forestpy.ml.mlp.trainer import EarlyStopping, TrainHistory


class MLPClassifierTrainer:
    """
    Treinador para classificadores MLP multiclasse.

    Args:
        model: Instância de `MLPClassifier`.
        learning_rate: Taxa de aprendizado do otimizador Adam.
        weight_decay: Regularização L2.
        device: 'cpu' ou 'cuda'. Se None, detecta automaticamente.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.history = TrainHistory()

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """Cria DataLoader. Labels em torch.long para CrossEntropyLoss."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)

        return total_loss / n_samples

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)

        return total_loss / n_samples

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        patience: int = 20,
        verbose: bool = True,
    ) -> TrainHistory:
        """
        Treina o classificador com early stopping baseado em validação.

        Args:
            X_train, y_train: Conjunto de treino. `y_train` deve conter
                índices inteiros das classes (0, 1, 2, ...).
            X_val, y_val: Conjunto de validação.
            epochs: Número máximo de épocas.
            batch_size: Tamanho dos mini-batches.
            patience: Épocas sem melhoria para parar.
            verbose: Imprime progresso a cada 10 épocas.

        Returns:
            Histórico de treino (`TrainHistory`).
        """
        train_loader = self._make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, batch_size, shuffle=False)

        early = EarlyStopping(patience=patience)
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._eval_epoch(val_loader)

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)

            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"  Época {epoch:>3d} — "
                    f"train_loss = {train_loss:.5f} | val_loss = {val_loss:.5f}"
                )

            if early(val_loss):
                if verbose:
                    print(f"  Early stopping na época {epoch} "
                          f"(melhor: {self.history.best_epoch})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna índices de classe preditos (argmax dos logits)."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        logits = self.model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna probabilidades por classe (softmax dos logits).

        Útil para análises de confiança da predição, calibração, e
        contagem de empates entre classes.

        Returns:
            Array (n_samples, n_classes) com probabilidades somando 1.
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        logits = self.model(X_t)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()
