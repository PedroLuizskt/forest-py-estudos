"""
Carregamento e validação de arquivos de configuração YAML.

Cada experimento em `configs/` define hiperparâmetros, arquitetura e caminhos
de dados, permitindo reprodutibilidade total via:

    >>> from forestpy.utils.config import load_config
    >>> cfg = load_config("configs/mlp_volumetria.yaml")
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Carrega um arquivo YAML como dicionário.

    Args:
        path: Caminho do YAML.

    Returns:
        Dicionário com a configuração.

    Raises:
        FileNotFoundError: Se o caminho não existir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config não encontrada: {path}")

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Salva um dicionário como YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
