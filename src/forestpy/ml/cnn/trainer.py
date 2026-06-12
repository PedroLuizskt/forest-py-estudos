"""
Treinador para Redes Neurais Convolucionais (CNN).

A lógica de treinamento é idêntica ao `MLPClassifierTrainer`:
CrossEntropyLoss, otimizador Adam, early stopping. A diferença está
apenas no formato esperado das features de entrada — tensors 4D
(batch, channels, height, width) em vez de 2D (batch, features).

Por essa razão, `CNNTrainer` herda diretamente de `MLPClassifierTrainer`,
reaproveitando todo o loop de treinamento, validação e predição.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from forestpy.ml.mlp.classifier_trainer import MLPClassifierTrainer


class CNNTrainer(MLPClassifierTrainer):
    """
    Treinador para CNNs de classificação multiclasse.

    Comportamento idêntico ao `MLPClassifierTrainer`, exceto pelo
    formato dos tensors de entrada (X é 4D em vez de 2D).

    Args:
        model: Instância de uma arquitetura CNN (ex.: `SimpleCNN`).
        learning_rate: Taxa de aprendizado do Adam.
        weight_decay: Regularização L2.
        device: 'cpu' ou 'cuda'. Se None, detecta automaticamente.
    """

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """
        Cria DataLoader esperando X com 4 dimensões (batch, C, H, W).

        Sobrescreve o método da classe-pai para preservar a estrutura
        espacial da entrada (não achata em vetor 2D como faria o MLP).
        """
        if X.ndim != 4:
            raise ValueError(
                f"CNNTrainer espera entrada 4D (batch, C, H, W). "
                f"Recebido shape: {X.shape}"
            )
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size,
            shuffle=shuffle,
        )
