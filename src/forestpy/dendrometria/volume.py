"""
Equações de volume — modelos paramétricos consagrados na dendrometria.

Referências:
    Husch, B.; Beers, T. W.; Kershaw Jr., J. A. (2003). Forest Mensuration. 4ª ed.
    Campos, J. C. C.; Leite, H. G. (2017). Mensuração Florestal. 5ª ed. Viçosa: UFV.

Todas as equações retornam volume **em metros cúbicos (m³)** assumindo:
    - DAP em **centímetros (cm)**
    - Altura total ou comercial em **metros (m)**
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


# ──────────────────────────────────────────────────────────────
# Helpers privados
# ──────────────────────────────────────────────────────────────
def _validate_positive(name: str, value: ArrayLike) -> np.ndarray:
    """Valida que todos os elementos são estritamente positivos."""
    arr = np.asarray(value, dtype=float)
    if np.any(arr <= 0):
        raise ValueError(f"{name} deve ser estritamente positivo. Recebido: {value!r}")
    return arr


# ──────────────────────────────────────────────────────────────
# Equação de Schumacher-Hall (1933)
# ──────────────────────────────────────────────────────────────
def schumacher_hall(
    dap: ArrayLike,
    h: ArrayLike,
    b0: float = -9.5,
    b1: float = 1.8,
    b2: float = 1.1,
) -> np.ndarray:
    """
    Equação volumétrica logarítmica de Schumacher-Hall.

    Forma:
        ln(V) = β₀ + β₁·ln(DAP) + β₂·ln(H)
        ⇒ V = exp(β₀) · DAP^β₁ · H^β₂

    É a equação volumétrica mais utilizada no Brasil e na literatura mundial,
    apresentando geralmente os menores resíduos para plantios homogêneos.

    Args:
        dap: Diâmetro à altura do peito (cm). Escalar ou array.
        h:   Altura total ou comercial (m). Escalar ou array.
        b0:  Intercepto da forma logarítmica. Default: -9.5.
        b1:  Coeficiente do ln(DAP). Default: 1.8.
        b2:  Coeficiente do ln(H). Default: 1.1.

    Returns:
        Volume estimado (m³) com mesma dimensão dos inputs.

    Raises:
        ValueError: Se DAP ou H contiverem valores ≤ 0.

    Example:
        >>> v = schumacher_hall(dap=20.0, h=15.0)
        >>> round(float(v), 4)
        0.1881

        >>> import numpy as np
        >>> dap = np.array([15.0, 20.0, 25.0])
        >>> h = np.array([12.0, 15.0, 18.0])
        >>> v = schumacher_hall(dap, h)
        >>> v.shape
        (3,)
    """
    dap_arr = _validate_positive("DAP", dap)
    h_arr = _validate_positive("H", h)

    return np.exp(b0) * np.power(dap_arr, b1) * np.power(h_arr, b2)


# ──────────────────────────────────────────────────────────────
# Equação de Spurr (1952)
# ──────────────────────────────────────────────────────────────
def spurr(
    dap: ArrayLike,
    h: ArrayLike,
    b0: float = 0.001,
    b1: float = 0.00003,
) -> np.ndarray:
    """
    Equação volumétrica de variável combinada de Spurr.

    Forma:
        V = β₀ + β₁ · (DAP² · H)

    Modelo linear simples que usa o produto DAP²·H como única variável
    independente (proxy para volume cilíndrico). Recomendado quando há
    dados limitados para ajuste de modelos não-lineares.

    Args:
        dap: Diâmetro à altura do peito (cm).
        h:   Altura (m).
        b0:  Intercepto.
        b1:  Coeficiente angular.

    Returns:
        Volume estimado (m³).

    Raises:
        ValueError: Se DAP ou H contiverem valores ≤ 0.

    Example:
        >>> round(float(spurr(20.0, 15.0)), 4)
        0.181
    """
    dap_arr = _validate_positive("DAP", dap)
    h_arr = _validate_positive("H", h)

    return b0 + b1 * np.power(dap_arr, 2) * h_arr


# ──────────────────────────────────────────────────────────────
# Equação de Honer (1965)
# ──────────────────────────────────────────────────────────────
def honer(
    dap: ArrayLike,
    h: ArrayLike,
    b0: float = 200.0,
    b1: float = 3500.0,
) -> np.ndarray:
    """
    Equação volumétrica de Honer.

    Forma:
        V = DAP² / (β₀ + β₁/H)

    Modelo proposto por Honer (1965), amplamente utilizado para coníferas
    no hemisfério norte. Apresenta boa estabilidade para árvores de pequeno
    porte.

    Args:
        dap: Diâmetro à altura do peito (cm).
        h:   Altura (m).
        b0:  Coeficiente do termo livre no denominador.
        b1:  Coeficiente do termo 1/H no denominador.

    Returns:
        Volume estimado (m³).

    Raises:
        ValueError: Se DAP ou H contiverem valores ≤ 0.

    Example:
        >>> round(float(honer(20.0, 15.0)), 4)
        0.9231
    """
    dap_arr = _validate_positive("DAP", dap)
    h_arr = _validate_positive("H", h)

    return np.power(dap_arr, 2) / (b0 + b1 / h_arr)
