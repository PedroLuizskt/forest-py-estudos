"""
Treinador de Redes Neurais Densas (MLP) em PyTorch.

Implementa o loop manual de treinamento, padrão didático para mostrar
explicitamente cada etapa (forward, loss, backward, step), em contraste
com APIs de alto nível como `model.fit()` do Keras.

Componentes:
    - `MLPTrainer`: classe principal que encapsula treino + validação
    - `EarlyStopping`: critério de parada antecipada baseado em validação
    - `TrainHistory`: registro histórico de losses por época

Estratégias incorporadas:
    - **Early stopping** com paciência (Prechelt, 1998): evita overfitting
    - **Mini-batches**: gradiente estocástico para melhor generalização
    - **Validação a cada época**: monitora desempenho fora do treino
    - **Restauração do melhor modelo**: ao final, recupera os pesos ótimos
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainHistory:
    """Histórico de treinamento para diagnóstico e plotagem."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


class EarlyStopping:
    """
    Critério de parada antecipada.

    Interrompe o treinamento quando a loss de validação não melhora por
    `patience` épocas consecutivas.

    Args:
        patience: Número de épocas sem melhoria antes de parar.
        min_delta: Diferença mínima na loss para considerar como melhoria.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Atualiza o estado e retorna True se deve parar."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class MLPTrainer:
    """
    Treinador para MLPs de regressão.

    Args:
        model: Instância de `nn.Module` (geralmente MLPRegressor).
        learning_rate: Taxa de aprendizado do otimizador Adam.
        weight_decay: Regularização L2 (0 desativa).
        device: 'cpu' ou 'cuda'. Se None, detecta automaticamente.

    Example:
        >>> from forestpy.ml.mlp.architectures import MLPRegressor
        >>> model = MLPRegressor(input_dim=2, hidden_dims=[16, 8])
        >>> trainer = MLPTrainer(model)
        >>> # trainer.fit(X_train, y_train, X_val, y_val)
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
        self.criterion = nn.MSELoss()
        self.history = TrainHistory()

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """Cria um DataLoader a partir de arrays NumPy."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _train_epoch(self, loader: DataLoader) -> float:
        """Executa uma época de treinamento e retorna a loss média."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)

            # Backward + step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)

        return total_loss / n_samples

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        """Avalia o modelo sem atualizar pesos."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
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
        Treina o modelo com early stopping baseado em validação.

        Args:
            X_train, y_train: Conjunto de treinamento.
            X_val, y_val: Conjunto de validação (para early stopping).
            epochs: Número máximo de épocas.
            batch_size: Tamanho dos mini-batches.
            patience: Épocas sem melhoria para parar.
            verbose: Imprime progresso a cada 10 épocas.

        Returns:
            Histórico de treinamento (`TrainHistory`).
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

            # Restaura o melhor estado quando há melhoria
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
                    print(f"  ⏸  Early stopping na época {epoch} "
                          f"(melhor: {self.history.best_epoch})")
                break

        # Restaura os melhores pesos
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Gera predições para um conjunto X. Retorna array 1D."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_pred = self.model(X_t).cpu().numpy().ravel()
        return y_pred
