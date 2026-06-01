"""
Amostragem em inventário florestal.

Implementa os dois processos amostrais mais utilizados no inventário florestal
brasileiro:

    - **Amostragem Aleatória Simples (AAS)**: seleção de unidades amostrais
      independentes de igual probabilidade.
    - **Amostragem Estratificada (AE)**: divisão da população em estratos
      homogêneos seguida de AAS dentro de cada estrato.

Referências:
    Péllico Netto, S.; Brena, D. A. (1997). *Inventário Florestal*. Curitiba.
    Cochran, W. G. (1977). *Sampling Techniques* (3ª ed.). Wiley.
    Campos, J. C. C.; Leite, H. G. (2017). *Mensuração Florestal*. UFV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SamplingResult:
    """
    Resultado de um processo amostral.

    Attributes:
        method: Nome do método aplicado ('AAS' ou 'AE').
        mean: Estimativa pontual da média da população.
        variance: Variância da média.
        std_error: Erro-padrão da média (√variância).
        ci_lower: Limite inferior do intervalo de confiança.
        ci_upper: Limite superior do intervalo de confiança.
        sampling_error_pct: Erro de amostragem relativo (%).
        n: Tamanho da amostra utilizada.
        confidence_level: Nível de confiança adotado (ex.: 0.95).
    """

    method: str
    mean: float
    variance: float
    std_error: float
    ci_lower: float
    ci_upper: float
    sampling_error_pct: float
    n: int
    confidence_level: float

    def summary(self) -> str:
        """Retorna um resumo textual formatado."""
        ci_pct = int(self.confidence_level * 100)
        return (
            f"Resultado: {self.method}\n"
            f"{'─' * 50}\n"
            f"  Tamanho da amostra (n)     : {self.n}\n"
            f"  Média estimada             : {self.mean:.4f}\n"
            f"  Variância da média         : {self.variance:.6f}\n"
            f"  Erro-padrão da média       : {self.std_error:.4f}\n"
            f"  IC {ci_pct}%                     : "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"  Erro de amostragem (%)     : {self.sampling_error_pct:.2f}%"
        )


# ──────────────────────────────────────────────────────────────
# Amostragem Aleatória Simples
# ──────────────────────────────────────────────────────────────
def aas(
    valores: np.ndarray | pd.Series,
    confidence_level: float = 0.95,
    population_size: int | None = None,
) -> SamplingResult:
    """
    Estima a média populacional por Amostragem Aleatória Simples.

    Aplica correção para população finita (FPC) quando `population_size` é informado.

    Args:
        valores: Vetor de valores observados nas unidades amostrais
            (por exemplo, volume por parcela em m³/ha).
        confidence_level: Nível de confiança do IC (0.90, 0.95 ou 0.99).
        population_size: Tamanho total da população (N). Se informado, aplica
            o fator de correção (N - n) / N. Se None, assume população infinita.

    Returns:
        SamplingResult com média, variância, IC e erro de amostragem.

    Raises:
        ValueError: Se a amostra tiver menos de 2 observações.

    Example:
        >>> import numpy as np
        >>> volumes = np.array([180.0, 195.0, 210.0, 188.0, 202.0])
        >>> res = aas(volumes, confidence_level=0.95)
        >>> 170 < res.mean < 210
        True
    """
    x = np.asarray(valores, dtype=float).ravel()
    n = len(x)

    if n < 2:
        raise ValueError(f"AAS requer no mínimo 2 observações. Recebido: {n}")

    mean = float(np.mean(x))
    # Variância amostral (com correção de Bessel: ddof=1)
    s2 = float(np.var(x, ddof=1))

    # Variância da média: s² / n (com FPC se aplicável)
    if population_size is not None and population_size > n:
        fpc = (population_size - n) / population_size
        var_mean = (s2 / n) * fpc
    else:
        var_mean = s2 / n

    std_error = float(np.sqrt(var_mean))

    # IC pela distribuição t de Student (gl = n - 1)
    alpha = 1 - confidence_level
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    margin = t_crit * std_error

    ci_lower = mean - margin
    ci_upper = mean + margin
    sampling_error_pct = (margin / mean) * 100 if mean != 0 else float("nan")

    return SamplingResult(
        method="AAS",
        mean=mean,
        variance=var_mean,
        std_error=std_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        sampling_error_pct=sampling_error_pct,
        n=n,
        confidence_level=confidence_level,
    )


# ──────────────────────────────────────────────────────────────
# Amostragem Estratificada
# ──────────────────────────────────────────────────────────────
def estratificada(
    df: pd.DataFrame,
    valor_col: str,
    estrato_col: str,
    confidence_level: float = 0.95,
    pesos: dict | None = None,
) -> SamplingResult:
    """
    Estima a média populacional por Amostragem Estratificada.

    A média global é a soma ponderada das médias dos estratos, com pesos iguais
    à proporção de cada estrato (Nh / N). A variância da média estratificada é
    geralmente **menor** que a da AAS para a mesma amostra, desde que a variável
    de estratificação esteja correlacionada com a variável de interesse.

    Args:
        df: DataFrame contendo os dados amostrais.
        valor_col: Nome da coluna numérica a estimar.
        estrato_col: Nome da coluna categórica que define os estratos.
        confidence_level: Nível de confiança do IC.
        pesos: Dicionário {estrato: peso} com a proporção de cada estrato na
            população. Se None, os pesos são estimados pela frequência na amostra
            (allocação proporcional implícita).

    Returns:
        SamplingResult com média ponderada, variância estratificada e IC.

    Raises:
        ValueError: Se algum estrato tiver menos de 2 observações.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'volume': [180, 195, 210, 188, 202, 165, 170, 175],
        ...     'classe': ['I', 'I', 'I', 'I', 'I', 'III', 'III', 'III'],
        ... })
        >>> res = estratificada(df, 'volume', 'classe')
        >>> res.method
        'AE'
    """
    estratos = df.groupby(estrato_col)[valor_col]

    # Validação: cada estrato precisa ter ≥ 2 observações
    for nome, grupo in estratos:
        if len(grupo) < 2:
            raise ValueError(
                f"Estrato '{nome}' tem apenas {len(grupo)} observação(ões). "
                "Mínimo: 2."
            )

    # Pesos: proporção de cada estrato na população
    if pesos is None:
        total = len(df)
        pesos = {nome: len(grupo) / total for nome, grupo in estratos}

    # Estatísticas por estrato
    medias_h = estratos.mean()
    vars_h = estratos.var(ddof=1)  # variância amostral dentro do estrato
    n_h = estratos.size()

    # Média estratificada: soma ponderada das médias
    mean = float(sum(pesos[h] * medias_h[h] for h in medias_h.index))

    # Variância da média estratificada: Σ (Wh² * sh² / nh)
    var_mean = float(sum(
        (pesos[h] ** 2) * (vars_h[h] / n_h[h])
        for h in medias_h.index
    ))

    std_error = float(np.sqrt(var_mean))

    # Graus de liberdade (Satterthwaite, aproximação): n_total - L
    n_total = int(n_h.sum())
    n_estratos = len(medias_h)
    gl = max(n_total - n_estratos, 1)

    alpha = 1 - confidence_level
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=gl))
    margin = t_crit * std_error

    ci_lower = mean - margin
    ci_upper = mean + margin
    sampling_error_pct = (margin / mean) * 100 if mean != 0 else float("nan")

    return SamplingResult(
        method="AE",
        mean=mean,
        variance=var_mean,
        std_error=std_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        sampling_error_pct=sampling_error_pct,
        n=n_total,
        confidence_level=confidence_level,
    )


# ──────────────────────────────────────────────────────────────
# Suficiência amostral
# ──────────────────────────────────────────────────────────────
def tamanho_amostra(
    cv_percent: float,
    erro_admissivel_percent: float,
    confidence_level: float = 0.95,
    population_size: int | None = None,
) -> int:
    """
    Calcula o tamanho mínimo de amostra necessário para um erro-alvo.

    Fórmula clássica para amostragem aleatória simples:
        n = (t² · CV²) / E²

    Com correção para população finita:
        n = (N · t² · CV²) / (N · E² + t² · CV²)

    Args:
        cv_percent: Coeficiente de variação da variável de interesse (%).
        erro_admissivel_percent: Erro de amostragem admissível (%).
        confidence_level: Nível de confiança (0.90, 0.95 ou 0.99).
        population_size: Tamanho da população (N). Se None, assume infinita.

    Returns:
        Tamanho de amostra recomendado (arredondado para cima).

    Example:
        >>> tamanho_amostra(cv_percent=25.0, erro_admissivel_percent=10.0)
        25
    """
    alpha = 1 - confidence_level
    # Aproximação inicial pelo z (após convergir, refina com t se necessário)
    t = float(stats.norm.ppf(1 - alpha / 2))

    cv2 = cv_percent ** 2
    e2 = erro_admissivel_percent ** 2

    if population_size is None:
        n = (t ** 2 * cv2) / e2
    else:
        numerador = population_size * (t ** 2) * cv2
        denominador = population_size * e2 + (t ** 2) * cv2
        n = numerador / denominador

    return int(np.ceil(n))
