"""
Ajuste de modelos dendrométricos por mínimos quadrados.

Enquanto os módulos `volume` e `hipsometria` fornecem as *formas funcionais*
das equações com coeficientes-padrão, este módulo **ajusta** esses coeficientes
aos dados reais via `scipy.optimize.curve_fit` (mínimos quadrados não-lineares),
retornando os parâmetros estimados e métricas de qualidade de ajuste.

Fluxo típico:
    1. Define-se a forma funcional (ex.: Schumacher-Hall)
    2. `fit_model` estima os coeficientes ótimos para os dados
    3. Retorna um `FitResult` com coeficientes, predições e métricas
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from forestpy.ml.metrics import bias, mae, mape, r2, rmse


@dataclass
class FitResult:
    """
    Resultado de um ajuste de modelo.

    Attributes:
        model_name: Nome do modelo ajustado.
        coefficients: Coeficientes estimados (dict nome→valor).
        y_pred: Valores preditos para os dados de ajuste.
        metrics: Dicionário de métricas (rmse, mae, mape, r2, bias).
        success: Se a otimização convergiu.
    """

    model_name: str
    coefficients: dict[str, float]
    y_pred: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)
    success: bool = True

    def summary(self) -> str:
        """Retorna um resumo textual formatado do ajuste."""
        linhas = [f"📊 Ajuste: {self.model_name}", "─" * 40]
        linhas.append("Coeficientes:")
        for nome, valor in self.coefficients.items():
            linhas.append(f"  {nome:6s} = {valor:>12.6f}")
        linhas.append("Métricas:")
        for nome, valor in self.metrics.items():
            linhas.append(f"  {nome:6s} = {valor:>12.6f}")
        return "\n".join(linhas)


# ──────────────────────────────────────────────────────────────
# Formas funcionais para ajuste (parametrizadas para curve_fit)
# ──────────────────────────────────────────────────────────────
def _schumacher_hall_form(X: np.ndarray, b0: float, b1: float, b2: float) -> np.ndarray:
    """Forma de Schumacher-Hall para curve_fit. X = (dap, h) empilhados."""
    dap, h = X
    return np.exp(b0) * np.power(dap, b1) * np.power(h, b2)


def _spurr_form(X: np.ndarray, b0: float, b1: float) -> np.ndarray:
    """Forma de Spurr para curve_fit."""
    dap, h = X
    return b0 + b1 * np.power(dap, 2) * h


def _curtis_form(dap: np.ndarray, b0: float, b1: float) -> np.ndarray:
    """Forma de Curtis (hipsométrica) para curve_fit."""
    return np.exp(b0 + b1 / dap)


def _stoffels_form(dap: np.ndarray, b0: float, b1: float) -> np.ndarray:
    """Forma de Stoffels (hipsométrica) para curve_fit."""
    return np.exp(b0) * np.power(dap, b1)


# Registro de modelos disponíveis: nome → (forma, nomes_coef, p0, tipo_entrada)
_MODELS: dict[str, dict] = {
    "schumacher_hall": {
        "form": _schumacher_hall_form,
        "coef_names": ["b0", "b1", "b2"],
        "p0": [-9.5, 1.8, 1.1],
        "input": "dap_h",
    },
    "spurr": {
        "form": _spurr_form,
        "coef_names": ["b0", "b1"],
        "p0": [0.001, 0.00003],
        "input": "dap_h",
    },
    "curtis": {
        "form": _curtis_form,
        "coef_names": ["b0", "b1"],
        "p0": [3.2, -8.0],
        "input": "dap",
    },
    "stoffels": {
        "form": _stoffels_form,
        "coef_names": ["b0", "b1"],
        "p0": [0.5, 0.85],
        "input": "dap",
    },
}


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula o conjunto-padrão de métricas de regressão."""
    return {
        "rmse": float(rmse(y_true, y_pred)),
        "mae": float(mae(y_true, y_pred)),
        "mape": float(mape(y_true, y_pred)),
        "r2": float(r2(y_true, y_pred)),
        "bias": float(bias(y_true, y_pred)),
    }


def fit_model(
    model_name: str,
    y: ArrayLike,
    dap: ArrayLike,
    h: ArrayLike | None = None,
    p0: list[float] | None = None,
    maxfev: int = 10000,
) -> FitResult:
    """
    Ajusta um modelo dendrométrico aos dados via mínimos quadrados não-lineares.

    Args:
        model_name: Um de {'schumacher_hall', 'spurr', 'curtis', 'stoffels'}.
        y: Variável resposta observada (volume para volumetria; altura para hipsometria).
        dap: Diâmetro à altura do peito (cm).
        h: Altura (m). Obrigatório para modelos volumétricos; ignorado para hipsométricos.
        p0: Chute inicial dos coeficientes. Se None, usa o padrão do modelo.
        maxfev: Máximo de avaliações da função na otimização.

    Returns:
        FitResult com coeficientes estimados, predições e métricas.

    Raises:
        ValueError: Se o modelo for desconhecido ou faltar `h` para modelo volumétrico.

    Example:
        >>> import numpy as np
        >>> from forestpy.data.loaders import load_pef_vinhedo
        >>> df = load_pef_vinhedo(synthetic_fallback=True, n_synthetic=200)
        >>> res = fit_model("schumacher_hall", df["volume"], df["dap"], df["h"])
        >>> res.metrics["r2"] > 0.9
        True
    """
    if model_name not in _MODELS:
        disponiveis = ", ".join(_MODELS.keys())
        raise ValueError(f"Modelo '{model_name}' desconhecido. Disponíveis: {disponiveis}")

    spec = _MODELS[model_name]
    y_arr = np.asarray(y, dtype=float)
    dap_arr = np.asarray(dap, dtype=float)

    # Monta a entrada conforme o tipo do modelo
    if spec["input"] == "dap_h":
        if h is None:
            raise ValueError(f"Modelo '{model_name}' requer altura (h).")
        h_arr = np.asarray(h, dtype=float)
        xdata = np.vstack([dap_arr, h_arr])
    else:  # 'dap'
        xdata = dap_arr

    form: Callable = spec["form"]
    p0 = p0 or spec["p0"]

    # Ajuste
    try:
        popt, _ = curve_fit(form, xdata, y_arr, p0=p0, maxfev=maxfev)
        success = True
    except RuntimeError:
        popt = np.array(p0)
        success = False

    y_pred = form(xdata, *popt)
    coefficients = dict(zip(spec["coef_names"], popt, strict=True))
    metrics = _compute_metrics(y_arr, y_pred)

    return FitResult(
        model_name=model_name,
        coefficients=coefficients,
        y_pred=y_pred,
        metrics=metrics,
        success=success,
    )


def compare_models(
    model_names: list[str],
    y: ArrayLike,
    dap: ArrayLike,
    h: ArrayLike | None = None,
) -> list[FitResult]:
    """
    Ajusta múltiplos modelos aos mesmos dados e retorna os resultados ordenados
    por RMSE crescente (melhor primeiro).

    Args:
        model_names: Lista de nomes de modelos a comparar.
        y: Variável resposta.
        dap: DAP (cm).
        h: Altura (m), se aplicável.

    Returns:
        Lista de FitResult ordenada do melhor (menor RMSE) ao pior.
    """
    resultados = [fit_model(nome, y, dap, h) for nome in model_names]
    return sorted(resultados, key=lambda r: r.metrics["rmse"])
