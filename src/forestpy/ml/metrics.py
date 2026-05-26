"""
Métricas de avaliação para regressão e classificação.

Centraliza o cálculo de métricas usadas em todo o projeto — desde os ajustes
clássicos (dendrometria/fitting) até as redes neurais (ml/mlp, ml/cnn) —
garantindo consistência metodológica nas comparações.

Convenção: todas as funções de regressão recebem (y_true, y_pred) como arrays
e retornam um float.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Converte entradas em arrays float e valida shapes compatíveis."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Shapes incompatíveis: y_true{yt.shape} vs y_pred{yp.shape}")
    if yt.size == 0:
        raise ValueError("Arrays vazios não são permitidos.")
    return yt, yp


# ──────────────────────────────────────────────────────────────
# Métricas de regressão
# ──────────────────────────────────────────────────────────────
def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error (raiz do erro quadrático médio).

    Penaliza erros grandes mais fortemente. Mesma unidade da variável resposta.

    Example:
        >>> round(rmse([1.0, 2.0, 3.0], [1.1, 2.1, 2.9]), 4)
        0.1
    """
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Absolute Error (erro absoluto médio).

    Menos sensível a outliers que o RMSE. Mesma unidade da resposta.

    Example:
        >>> round(mae([1.0, 2.0, 3.0], [1.1, 2.1, 2.9]), 4)
        0.1
    """
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Absolute Percentage Error (erro percentual absoluto médio), em %.

    Adimensional, facilita comparação entre variáveis de escalas diferentes.
    Atenção: indefinido quando y_true contém zeros (são ignorados).

    Example:
        >>> round(mape([100.0, 200.0], [110.0, 190.0]), 2)
        7.5
    """
    yt, yp = _as_arrays(y_true, y_pred)
    mask = yt != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Coeficiente de determinação R².

    Proporção da variância explicada pelo modelo. 1.0 = ajuste perfeito;
    0.0 = não melhor que a média; negativo = pior que a média.

    Example:
        >>> r2([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        1.0
    """
    yt, yp = _as_arrays(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Viés médio (tendência sistemática) = média(y_pred - y_true).

    Positivo = modelo superestima; negativo = subestima; ~0 = sem viés.

    Example:
        >>> round(bias([1.0, 2.0, 3.0], [1.1, 2.1, 3.1]), 4)
        0.1
    """
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(yp - yt))


# ──────────────────────────────────────────────────────────────
# Métricas de classificação
# ──────────────────────────────────────────────────────────────
def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Acurácia: proporção de predições corretas.

    Example:
        >>> accuracy([1, 0, 1, 1], [1, 0, 0, 1])
        0.75
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.shape != yp.shape:
        raise ValueError("Shapes incompatíveis.")
    return float(np.mean(yt == yp))


def regression_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """
    Calcula o conjunto-padrão de métricas de regressão de uma vez.

    Returns:
        Dict com rmse, mae, mape, r2 e bias.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
