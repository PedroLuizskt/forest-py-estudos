"""
Avaliação de modelos: validação cruzada e bootstrap.

Centraliza os procedimentos de avaliação rigorosa de modelos preditivos
(clássicos e neurais), garantindo comparações justas e métricas com
intervalos de confiança via reamostragem.

Procedimentos implementados:
    - K-Fold Cross-Validation: divide os dados em k partes, treina em k-1 e
      avalia na restante, repetindo k vezes.
    - Bootstrap: reamostra com reposição para gerar IC das métricas.

Referências:
    Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of
        Statistical Learning* (2ª ed.). Springer.
    Efron, B.; Tibshirani, R. (1993). *An Introduction to the Bootstrap*. CRC.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from forestpy.ml.metrics import regression_report


@dataclass
class CVResult:
    """
    Resultado de uma validação cruzada k-fold.

    Attributes:
        model_name: Nome do modelo avaliado.
        n_splits: Número de dobras (k).
        fold_metrics: Lista de dicts (uma entrada por dobra).
        mean_metrics: Média das métricas entre dobras.
        std_metrics: Desvio-padrão das métricas entre dobras.
    """

    model_name: str
    n_splits: int
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Converte os resultados em DataFrame (uma linha por dobra)."""
        df = pd.DataFrame(self.fold_metrics)
        df.insert(0, "fold", range(1, len(self.fold_metrics) + 1))
        return df

    def summary(self) -> str:
        """Resumo textual: média ± desvio das métricas."""
        linhas = [
            f"Validação Cruzada ({self.n_splits}-fold): {self.model_name}",
            "─" * 55,
        ]
        for metrica in self.mean_metrics:
            m = self.mean_metrics[metrica]
            s = self.std_metrics[metrica]
            linhas.append(f"  {metrica:6s} = {m:>10.4f} ± {s:.4f}")
        return "\n".join(linhas)


def kfold_cv(
    fit_predict_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    model_name: str = "modelo",
) -> CVResult:
    """
    Executa validação cruzada k-fold para uma função de ajuste/predição.

    Args:
        fit_predict_fn: Função que recebe (X_train, y_train, X_test) e retorna
            (y_true_test, y_pred_test). Permite usar qualquer modelo (sklearn,
            PyTorch, ajuste paramétrico).
        X: Features (DataFrame ou array 2D).
        y: Variável resposta.
        n_splits: Número de dobras k.
        shuffle: Embaralha os dados antes de dividir.
        random_state: Seed para reprodutibilidade.
        model_name: Nome do modelo (identificação no resultado).

    Returns:
        CVResult com métricas por dobra e agregadas (média ± desvio).
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []

    for train_idx, test_idx in kf.split(X_arr):
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]
        X_test, y_test = X_arr[test_idx], y_arr[test_idx]

        y_true_fold, y_pred_fold = fit_predict_fn(X_train, y_train, X_test)
        if y_true_fold is None:
            y_true_fold = y_test

        fold_metrics.append(regression_report(y_true_fold, y_pred_fold))

    df_metrics = pd.DataFrame(fold_metrics)
    return CVResult(
        model_name=model_name,
        n_splits=n_splits,
        fold_metrics=fold_metrics,
        mean_metrics=df_metrics.mean().to_dict(),
        std_metrics=df_metrics.std().to_dict(),
    )


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Estima a média e IC de uma métrica via bootstrap.

    Args:
        y_true: Valores observados.
        y_pred: Valores preditos.
        metric_fn: Função de métrica (ex.: rmse, r2).
        n_bootstrap: Número de reamostragens.
        confidence_level: Nível de confiança do IC.
        random_state: Seed para reprodutibilidade.

    Returns:
        Dict com `mean`, `std`, `ci_lower`, `ci_upper`.

    Example:
        >>> from forestpy.ml.metrics import rmse
        >>> import numpy as np
        >>> y_t = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_p = np.array([1.1, 1.9, 3.1, 4.0])
        >>> res = bootstrap_metric(y_t, y_p, rmse, n_bootstrap=100)
        >>> "mean" in res and "ci_lower" in res
        True
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)

    valores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        valores.append(metric_fn(y_true[idx], y_pred[idx]))

    valores = np.array(valores)
    alpha = 1 - confidence_level

    return {
        "mean": float(np.mean(valores)),
        "std": float(np.std(valores, ddof=1)),
        "ci_lower": float(np.quantile(valores, alpha / 2)),
        "ci_upper": float(np.quantile(valores, 1 - alpha / 2)),
    }
