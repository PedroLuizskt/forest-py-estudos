"""
Treinador para U-Net (segmentação semântica binária).

Implementa o loop manual com função de perda combinada BCE + Dice:

    L = α · BCE + (1 - α) · DiceLoss

A componente BCE provê gradientes estáveis pixel-a-pixel; a componente Dice
foca explicitamente na sobreposição de regiões, contornando o problema de
**desbalanço de classes** comum em segmentação (copas ocupam tipicamente
5-20% dos pixels em ortomosaicos florestais).

Referência:
    Milletari, F.; Navab, N.; Ahmadi, S.-A. (2016). V-Net: Fully convolutional
        neural networks for volumetric medical image segmentation. *3DV*.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from forestpy.ml.mlp.trainer import EarlyStopping, TrainHistory


class _DiceLoss(nn.Module):
    """Dice Loss = 1 - Dice (em logits via sigmoid)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten por amostra para Dice por imagem, depois média no batch
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.eps) / (denom + self.eps)
        return 1 - dice.mean()


class _BCEDiceLoss(nn.Module):
    """Combinação BCE + DiceLoss com pesos configuráveis."""

    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        if not 0 <= bce_weight <= 1:
            raise ValueError(
                f"bce_weight deve estar em [0, 1]. Recebido: {bce_weight}"
            )
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = _DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


class UNetTrainer:
    """
    Treinador para U-Net de segmentação semântica binária.

    Args:
        model: Instância de `UNet` (ou outra arquitetura encoder-decoder).
        learning_rate: Taxa de aprendizado do Adam.
        weight_decay: Regularização L2.
        bce_weight: Peso da componente BCE na perda combinada (0=só Dice, 1=só BCE).
        device: 'cpu' ou 'cuda'. Se None, detecta automaticamente.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        bce_weight: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = _BCEDiceLoss(bce_weight=bce_weight)
        self.history = TrainHistory()

    def _make_loader(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """
        Cria DataLoader para segmentação.

        X: (n, C, H, W) — chips multi-banda
        Y: (n, H, W) — máscaras binárias; convertidas para (n, 1, H, W)
        """
        if X.ndim != 4:
            raise ValueError(
                f"UNetTrainer espera X 4D (n, C, H, W). Recebido: {X.shape}"
            )
        if Y.ndim == 3:
            Y = Y[:, None, :, :]  # adiciona canal
        if Y.ndim != 4:
            raise ValueError(
                f"UNetTrainer espera Y 3D (n, H, W) ou 4D (n, 1, H, W). "
                f"Recebido: {Y.shape}"
            )
        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)
        return DataLoader(
            TensorDataset(X_t, Y_t),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        for X_b, Y_b in loader:
            X_b = X_b.to(self.device)
            Y_b = Y_b.to(self.device)
            logits = self.model(X_b)
            loss = self.criterion(logits, Y_b)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * X_b.size(0)
            n_samples += X_b.size(0)
        return total_loss / n_samples

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        for X_b, Y_b in loader:
            X_b = X_b.to(self.device)
            Y_b = Y_b.to(self.device)
            logits = self.model(X_b)
            loss = self.criterion(logits, Y_b)
            total_loss += loss.item() * X_b.size(0)
            n_samples += X_b.size(0)
        return total_loss / n_samples

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        epochs: int = 80,
        batch_size: int = 8,
        patience: int = 15,
        verbose: bool = True,
    ) -> TrainHistory:
        """
        Treina a U-Net com early stopping.

        Args:
            X_train, Y_train: Conjunto de treino — chips e máscaras.
            X_val, Y_val: Conjunto de validação.
            epochs: Número máximo de épocas.
            batch_size: Tamanho dos mini-batches.
            patience: Épocas sem melhoria para parar.
            verbose: Imprime progresso a cada 5 épocas.

        Returns:
            Histórico de treino.
        """
        train_loader = self._make_loader(X_train, Y_train, batch_size, shuffle=True)
        val_loader = self._make_loader(X_val, Y_val, batch_size, shuffle=False)

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

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  Época {epoch:>3d} — "
                      f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

            if early(val_loss):
                if verbose:
                    print(f"  Early stopping na época {epoch} "
                          f"(melhor: {self.history.best_epoch})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna probabilidades de copa por pixel (após sigmoid).

        Args:
            X: Array (n, C, H, W).

        Returns:
            Array (n, H, W) com probabilidades em [0, 1].
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        logits = self.model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        # Remove a dimensão de canal se for binário (out_channels=1)
        if probs.shape[1] == 1:
            probs = probs[:, 0]
        return probs

    @torch.no_grad()
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Retorna máscaras binárias (0/1) aplicando threshold às probabilidades.

        Args:
            X: Array (n, C, H, W).
            threshold: Limiar de decisão (padrão 0.5).

        Returns:
            Array (n, H, W) com máscaras binárias.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.uint8)
