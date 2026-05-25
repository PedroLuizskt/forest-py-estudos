"""
Logger estruturado com formatação rica via `rich`.

Uso:
    >>> from forestpy.utils import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("Iniciando treinamento...")
"""

import logging
import os
import sys

try:
    from rich.logging import RichHandler

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def get_logger(name: str = "forestpy", level: str | None = None) -> logging.Logger:
    """
    Retorna um logger configurado.

    Args:
        name: Nome do logger (geralmente `__name__` do módulo).
        level: Nível ('DEBUG', 'INFO', 'WARNING', 'ERROR'). Se None, lê de
            `LOG_LEVEL` no ambiente; padrão = 'INFO'.

    Returns:
        Logger pronto para uso.
    """
    log = logging.getLogger(name)

    if log.handlers:
        return log  # já configurado

    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log.setLevel(level)

    if _HAS_RICH:
        handler = RichHandler(rich_tracebacks=True, show_path=False)
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    else:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s  %(name)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.propagate = False

    return log
