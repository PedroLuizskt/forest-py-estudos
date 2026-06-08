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


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: list | None = None,
) -> np.ndarray:
    """
    Constrói a matriz de confusão.

    Convenção: linhas = classe real, colunas = classe predita.

    Args:
        y_true: Rótulos observados.
        y_pred: Rótulos preditos.
        labels: Ordem das classes na matriz. Se None, infere de `y_true ∪ y_pred`.

    Returns:
        Matriz quadrada de inteiros (n_classes × n_classes).

    Example:
        >>> cm = confusion_matrix([0, 1, 2, 0, 1], [0, 2, 2, 0, 1], labels=[0, 1, 2])
        >>> cm[0, 0]  # acertos da classe 0
        2
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()

    if labels is None:
        labels = sorted(set(yt.tolist()) | set(yp.tolist()))

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)

    for yt_i, yp_i in zip(yt, yp, strict=False):
        if yt_i in label_to_idx and yp_i in label_to_idx:
            cm[label_to_idx[yt_i], label_to_idx[yp_i]] += 1

    return cm


def precision_per_class(y_true: ArrayLike, y_pred: ArrayLike,
                        labels: list | None = None) -> np.ndarray:
    """
    Precision por classe: TP / (TP + FP).

    De todas as predições da classe k, quantas estavam corretas?
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    soma_col = cm.sum(axis=0)
    # Evita divisão por zero: classes nunca preditas → precision = 0
    return np.where(soma_col > 0, np.diag(cm) / np.maximum(soma_col, 1), 0.0)


def recall_per_class(y_true: ArrayLike, y_pred: ArrayLike,
                     labels: list | None = None) -> np.ndarray:
    """
    Recall por classe: TP / (TP + FN).

    De todas as instâncias reais da classe k, quantas foram capturadas?
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    soma_linha = cm.sum(axis=1)
    return np.where(soma_linha > 0, np.diag(cm) / np.maximum(soma_linha, 1), 0.0)


def f1_per_class(y_true: ArrayLike, y_pred: ArrayLike,
                 labels: list | None = None) -> np.ndarray:
    """F1 por classe: média harmônica de precision e recall."""
    p = precision_per_class(y_true, y_pred, labels=labels)
    r = recall_per_class(y_true, y_pred, labels=labels)
    denom = p + r
    return np.where(denom > 0, 2 * p * r / np.maximum(denom, 1e-12), 0.0)


def classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: list | None = None,
) -> dict[str, float]:
    """
    Conjunto-padrão de métricas de classificação.

    Returns:
        Dict com `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`.
        Macro = média não-ponderada entre classes (trata classes minoritárias
        com o mesmo peso que majoritárias).
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision_macro": float(np.mean(precision_per_class(y_true, y_pred, labels))),
        "recall_macro": float(np.mean(recall_per_class(y_true, y_pred, labels))),
        "f1_macro": float(np.mean(f1_per_class(y_true, y_pred, labels))),
    }


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
