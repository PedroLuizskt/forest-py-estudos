"""
Machine Learning aplicado à floresta — clássico e deep learning.

Submódulos:
    metrics        — RMSE, MAE, MAPE, R², bias, accuracy
    evaluation     — Cross-validation k-fold, bootstrap
    mlp            — Redes neurais tabulares (PyTorch)
    cnn            — Redes convolucionais (PyTorch + torchvision)
"""

from forestpy.ml.evaluation import CVResult, bootstrap_metric, kfold_cv
from forestpy.ml.metrics import (
    accuracy,
    bias,
    mae,
    mape,
    r2,
    regression_report,
    rmse,
)

__all__ = [
    # Métricas
    "rmse", "mae", "mape", "r2", "bias", "accuracy", "regression_report",
    # Avaliação
    "kfold_cv", "bootstrap_metric", "CVResult",
]
