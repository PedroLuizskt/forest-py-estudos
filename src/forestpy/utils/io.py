"""
I/O de modelos e artefatos.

Funções para salvar/carregar:
    - Modelos PyTorch (.pt)
    - Modelos scikit-learn (.pkl via joblib)
    - DataFrames processados (.parquet)
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def save_sklearn(model: Any, path: str | Path) -> None:
    """Persiste um modelo sklearn-like usando joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_sklearn(path: str | Path) -> Any:
    """Carrega um modelo sklearn-like."""
    return joblib.load(path)


def save_torch(model: Any, path: str | Path, **extra: Any) -> None:
    """
    Persiste um state_dict do PyTorch + metadados.

    Args:
        model: Instância de nn.Module.
        path: Destino (.pt).
        **extra: Metadados adicionais (config, métricas, epoch, etc.).
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"state_dict": model.state_dict(), **extra}
    torch.save(payload, path)


def load_torch(path: str | Path) -> dict[str, Any]:
    """Carrega payload salvo por `save_torch`."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Salva DataFrame em Parquet (formato colunar comprimido)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Carrega DataFrame de Parquet."""
    return pd.read_parquet(path)
