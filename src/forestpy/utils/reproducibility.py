"""
Reprodutibilidade: fixa todas as fontes de aleatoriedade do projeto.

Uso recomendado no início de todo notebook/script:

    >>> from forestpy.utils import set_seed
    >>> set_seed(42)
"""

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """
    Fixa as seeds de `random`, `numpy`, `torch` (CPU e CUDA) e variável de ambiente.

    Args:
        seed: Valor inteiro da seed.
        deterministic_torch: Se True, força operações determinísticas no PyTorch.
            Pode reduzir levemente a performance em GPU.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch é dependência opcional para sessões iniciais
        pass
