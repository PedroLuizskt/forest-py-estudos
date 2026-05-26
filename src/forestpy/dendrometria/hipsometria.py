"""
Relação hipsométrica — equações altura-diâmetro (H–DAP).

A relação hipsométrica estima a altura total de árvores a partir do DAP,
evitando a medição cara e demorada da altura de todas as árvores do
inventário. Mede-se a altura de uma subamostra, ajusta-se um modelo e
estimam-se as demais.

Referências:
    Campos, J. C. C.; Leite, H. G. (2017). Mensuração Florestal. 5ª ed. UFV.
    Curtis, R. O. (1967). Height-diameter and height-diameter-age equations.

Convenções:
    - DAP em centímetros (cm)
    - Altura em metros (m)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _validate_positive(name: str, value: ArrayLike) -> np.ndarray:
    """Valida que todos os elementos são estritamente positivos."""
    arr = np.asarray(value, dtype=float)
    if np.any(arr <= 0):
        raise ValueError(f"{name} deve ser estritamente positivo. Recebido: {value!r}")
    return arr


# ──────────────────────────────────────────────────────────────
# Curtis (1967)
# ──────────────────────────────────────────────────────────────
def curtis(
    dap: ArrayLike,
    b0: float = 3.2,
    b1: float = -8.0,
) -> np.ndarray:
    """
    Modelo hipsométrico de Curtis.

    Forma:
        ln(H) = β₀ + β₁·(1/DAP)
        ⇒ H = exp(β₀ + β₁/DAP)

    Modelo logarítmico de uma entrada (apenas DAP). Muito usado por sua
    simplicidade e bom ajuste em povoamentos equiâneos.

    Args:
        dap: Diâmetro à altura do peito (cm).
        b0:  Intercepto.
        b1:  Coeficiente do termo 1/DAP (geralmente negativo).

    Returns:
        Altura estimada (m).

    Raises:
        ValueError: Se DAP contiver valores ≤ 0.

    Example:
        >>> round(float(curtis(20.0)), 2)
        16.44
    """
    dap_arr = _validate_positive("DAP", dap)
    return np.exp(b0 + b1 / dap_arr)


# ──────────────────────────────────────────────────────────────
# Stoffels (1953)
# ──────────────────────────────────────────────────────────────
def stoffels(
    dap: ArrayLike,
    b0: float = 0.5,
    b1: float = 0.85,
) -> np.ndarray:
    """
    Modelo hipsométrico de Stoffels (potência).

    Forma:
        ln(H) = β₀ + β₁·ln(DAP)
        ⇒ H = exp(β₀) · DAP^β₁

    Modelo de potência (log-log). Flexível e amplamente aplicado.

    Args:
        dap: Diâmetro à altura do peito (cm).
        b0:  Intercepto da forma logarítmica.
        b1:  Expoente sobre o DAP.

    Returns:
        Altura estimada (m).

    Raises:
        ValueError: Se DAP contiver valores ≤ 0.

    Example:
        >>> round(float(stoffels(20.0)), 2)
        21.04
    """
    dap_arr = _validate_positive("DAP", dap)
    return np.exp(b0) * np.power(dap_arr, b1)


# ──────────────────────────────────────────────────────────────
# Henricksen (1950)
# ──────────────────────────────────────────────────────────────
def henricksen(
    dap: ArrayLike,
    b0: float = 1.5,
    b1: float = 5.0,
) -> np.ndarray:
    """
    Modelo hipsométrico de Henricksen (logarítmico simples).

    Forma:
        H = β₀ + β₁·ln(DAP)

    Modelo linear nos parâmetros, com transformação logarítmica do DAP.
    Bom comportamento para amplitudes diamétricas moderadas.

    Args:
        dap: Diâmetro à altura do peito (cm).
        b0:  Intercepto.
        b1:  Coeficiente do ln(DAP).

    Returns:
        Altura estimada (m).

    Raises:
        ValueError: Se DAP contiver valores ≤ 0.

    Example:
        >>> round(float(henricksen(20.0)), 2)
        16.48
    """
    dap_arr = _validate_positive("DAP", dap)
    return b0 + b1 * np.log(dap_arr)


# ──────────────────────────────────────────────────────────────
# Trorey (1932) — parabólico
# ──────────────────────────────────────────────────────────────
def trorey(
    dap: ArrayLike,
    b0: float = 1.3,
    b1: float = 1.0,
    b2: float = -0.015,
) -> np.ndarray:
    """
    Modelo hipsométrico de Trorey (parabólico de 2º grau).

    Forma:
        H = β₀ + β₁·DAP + β₂·DAP²

    Polinômio de segundo grau. O termo quadrático (β₂ < 0) captura a
    estabilização/queda da altura em diâmetros grandes.

    Args:
        dap: Diâmetro à altura do peito (cm).
        b0:  Intercepto.
        b1:  Coeficiente linear.
        b2:  Coeficiente quadrático (geralmente negativo).

    Returns:
        Altura estimada (m).

    Raises:
        ValueError: Se DAP contiver valores ≤ 0.

    Example:
        >>> round(float(trorey(20.0)), 2)
        15.3
    """
    dap_arr = _validate_positive("DAP", dap)
    return b0 + b1 * dap_arr + b2 * np.power(dap_arr, 2)
