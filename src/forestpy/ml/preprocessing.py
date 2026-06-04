"""
Pré-processamento de dados para modelos de Machine Learning.

Fornece transformações reproduzíveis para preparar features tabulares
florestais antes do treinamento de modelos neurais.

A normalização é **crítica para redes neurais**: features em escalas
diferentes (DAP em cm vs. idade em anos vs. altura em m) levam o gradiente
descendente a oscilar e dificultam a convergência.

Princípio fundamental: os parâmetros de normalização (média, desvio) são
estimados **apenas no conjunto de treino** e aplicados ao teste. Caso
contrário, há vazamento de informação (data leakage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


@dataclass
class StandardScalerForest:
    """
    Escalonamento padrão (z-score) com persistência dos parâmetros ajustados.

    Reimplementação minimalista do StandardScaler do scikit-learn, com
    interface explícita e suporte a DataFrames pandas preservando colunas.

    Fórmula:
        z = (x - μ) / σ

    onde μ e σ são estimados no conjunto de treino.

    Attributes:
        mean_: Médias por feature (definidas após fit).
        std_: Desvios-padrão por feature (definidos após fit).
        feature_names_: Nomes das colunas, se entrada for DataFrame.
    """

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    feature_names_: list[str] | None = None

    def fit(self, X: ArrayLike) -> "StandardScalerForest":
        """
        Estima média e desvio-padrão por feature.

        Args:
            X: Matriz de features (DataFrame ou array 2D).

        Returns:
            self (para chamadas encadeadas).
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X, dtype=float)

        self.mean_ = X_arr.mean(axis=0)
        self.std_ = X_arr.std(axis=0, ddof=0)

        # Evita divisão por zero para features constantes
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Aplica a transformação z = (x - μ) / σ.

        Args:
            X: Matriz de features no mesmo formato/ordem do fit.

        Returns:
            Array 2D normalizado.

        Raises:
            RuntimeError: Se chamado antes de fit().
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler não foi ajustado. Chame fit() primeiro.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """Atalho: ajusta e transforma na mesma chamada."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: ArrayLike) -> np.ndarray:
        """
        Reverte a transformação: x = z * σ + μ.

        Útil para inspecionar valores na escala original após manipulação
        no espaço normalizado.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler não foi ajustado.")
        X_arr = np.asarray(X_scaled, dtype=float)
        return X_arr * self.std_ + self.mean_
