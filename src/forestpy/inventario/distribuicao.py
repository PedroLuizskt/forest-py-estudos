"""
Distribuição diamétrica e ajuste por Weibull.

A distribuição de diâmetros (DAP) em uma parcela florestal segue, em geral,
uma forma unimodal assimétrica que a distribuição de Weibull com 2 parâmetros
modela bem. Esta é a abordagem clássica para projeções de inventário e
regulação da produção (Bailey & Dell, 1973; Campos & Leite, 2017).

Forma da densidade de Weibull:

    f(x; α, β) = (β/α) (x/α)^(β-1) exp(-(x/α)^β)

onde α > 0 é o parâmetro de escala (próximo da média) e β > 0 controla a
forma (β < 1 = J-invertido; β ≈ 3.5 ≈ normal; β > 5 = altamente assimétrica).

Referências:
    Bailey, R. L.; Dell, T. R. (1973). Quantifying diameter distributions
        with the Weibull function. *Forest Science*, 19(2), 97-104.
    Campos, J. C. C.; Leite, H. G. (2017). *Mensuração Florestal*. UFV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats


@dataclass
class WeibullFit:
    """
    Resultado do ajuste de Weibull a uma amostra de diâmetros.

    Attributes:
        shape: Parâmetro β (forma).
        scale: Parâmetro α (escala).
        loc: Parâmetro de localização (usualmente 0 ou DAP mínimo).
        n_observations: Tamanho da amostra ajustada.
    """

    shape: float
    scale: float
    loc: float = 0.0
    n_observations: int = 0

    def pdf(self, x: ArrayLike) -> np.ndarray:
        """Densidade de probabilidade da Weibull ajustada."""
        return stats.weibull_min.pdf(x, c=self.shape, loc=self.loc, scale=self.scale)

    def cdf(self, x: ArrayLike) -> np.ndarray:
        """Função de distribuição acumulada."""
        return stats.weibull_min.cdf(x, c=self.shape, loc=self.loc, scale=self.scale)

    def expected_counts(
        self,
        bin_edges: ArrayLike,
        n_total: int,
    ) -> np.ndarray:
        """
        Contagens esperadas por classe diamétrica.

        Args:
            bin_edges: Limites das classes (k+1 valores definem k classes).
            n_total: Número total de árvores a distribuir.

        Returns:
            Array com a contagem esperada em cada classe.
        """
        edges = np.asarray(bin_edges)
        cdfs = self.cdf(edges)
        probs = np.diff(cdfs)
        return probs * n_total


def fit_weibull(
    dap: ArrayLike,
    floc: float | None = 0.0,
) -> WeibullFit:
    """
    Ajusta Weibull aos diâmetros por máxima verossimilhança.

    Args:
        dap: Vetor de diâmetros (cm).
        floc: Se informado, fixa o parâmetro de localização (recomendado:
            0 para evitar superparametrização; ou DAP mínimo - epsilon).

    Returns:
        Objeto WeibullFit com os parâmetros estimados.

    Raises:
        ValueError: Se houver menos de 5 observações ou DAPs não-positivos.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> daps = rng.weibull(2.5, size=200) * 15
        >>> fit = fit_weibull(daps)
        >>> 2.0 < fit.shape < 3.0
        True
    """
    x = np.asarray(dap, dtype=float).ravel()

    if len(x) < 5:
        raise ValueError(f"Ajuste exige no mínimo 5 observações. Recebido: {len(x)}")
    if np.any(x <= 0):
        raise ValueError("DAP deve ser estritamente positivo.")

    kwargs = {"floc": floc} if floc is not None else {}
    shape, loc, scale = stats.weibull_min.fit(x, **kwargs)

    return WeibullFit(
        shape=float(shape),
        scale=float(scale),
        loc=float(loc),
        n_observations=len(x),
    )


def diametric_distribution(
    dap: ArrayLike,
    bin_width: float = 5.0,
    bin_min: float | None = None,
    bin_max: float | None = None,
) -> pd.DataFrame:
    """
    Constrói o histograma de classes diamétricas.

    Args:
        dap: Vetor de DAP (cm).
        bin_width: Largura das classes (padrão 5 cm — usual no inventário
            florestal brasileiro).
        bin_min: Limite inferior. Se None, usa o menor múltiplo de bin_width
            abaixo do DAP mínimo.
        bin_max: Limite superior. Se None, usa o maior múltiplo acima do DAP máximo.

    Returns:
        DataFrame com colunas: bin_lower, bin_upper, bin_center, count.
    """
    x = np.asarray(dap, dtype=float).ravel()

    if bin_min is None:
        bin_min = float(np.floor(x.min() / bin_width) * bin_width)
    if bin_max is None:
        bin_max = float(np.ceil(x.max() / bin_width) * bin_width)

    edges = np.arange(bin_min, bin_max + bin_width, bin_width)
    counts, _ = np.histogram(x, bins=edges)

    return pd.DataFrame({
        "bin_lower": edges[:-1],
        "bin_upper": edges[1:],
        "bin_center": (edges[:-1] + edges[1:]) / 2,
        "count": counts,
    })
