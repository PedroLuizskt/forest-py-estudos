"""
Loaders para bases florestais.

O dataset principal é o **PEF Vinhedo (SP)** — Parcelas de Eucalyptus do
Centro Experimental de Vinhedo, originalmente disponibilizado pelo pacote
`fptools`. Este módulo provê uma versão sintética compatível enquanto o
dataset oficial não está em mãos, permitindo desenvolvimento offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Diretório padrão para dados crus
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def load_pef_vinhedo(
    path: str | Path | None = None,
    synthetic_fallback: bool = True,
    n_synthetic: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Carrega o dataset PEF Vinhedo ou gera versão sintética compatível.

    Estrutura esperada (colunas):
        - parcela    : ID da parcela (int)
        - arvore     : ID da árvore dentro da parcela (int)
        - especie    : Espécie (str) — ex.: 'Eucalyptus grandis'
        - dap        : Diâmetro à altura do peito (cm)
        - h          : Altura total (m)
        - h_com      : Altura comercial (m)
        - idade      : Idade do plantio (anos)
        - classe     : Classe de qualidade ('I', 'II', 'III')
        - volume     : Volume individual estimado (m³), com ruído de medição

    Args:
        path: Caminho para o CSV oficial. Se None, busca em `data/raw/pef_vinhedo.csv`.
        synthetic_fallback: Se True e o arquivo não existir, gera dados sintéticos
            com distribuições realistas para Eucalyptus grandis.
        n_synthetic: Número de árvores na versão sintética.
        seed: Seed para reprodutibilidade dos dados sintéticos.

    Returns:
        DataFrame com as colunas listadas acima.

    Raises:
        FileNotFoundError: Se o caminho não existir e `synthetic_fallback=False`.
    """
    path = Path(path) if path else DEFAULT_DATA_DIR / "pef_vinhedo.csv"

    if path.exists():
        return pd.read_csv(path)

    if not synthetic_fallback:
        raise FileNotFoundError(
            f"Dataset PEF Vinhedo não encontrado em {path}. "
            "Use synthetic_fallback=True para gerar dados sintéticos."
        )

    return _generate_synthetic_pef(n=n_synthetic, seed=seed)


def _generate_synthetic_pef(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Gera dataset sintético com distribuições realistas de Eucalyptus grandis
    no Sudeste brasileiro.

    Modelo gerador:
        - DAP ~ Gamma(shape=8, scale=2.5) → média ~20 cm, cauda longa
        - idade ~ Discreta uniforme em {3, 5, 7, 10} anos
        - H = a(idade) · DAP^b(idade) + ε   (forma alométrica)
        - classe atribuída por quantis de produtividade simulada
    """
    rng = np.random.default_rng(seed)

    # ── Diâmetros ──
    dap = rng.gamma(shape=8.0, scale=2.5, size=n)
    dap = np.clip(dap, 5.0, 50.0)

    # ── Idade ──
    idade = rng.choice([3, 5, 7, 10], size=n, p=[0.15, 0.30, 0.35, 0.20])

    # ── Altura: forma alométrica com ruído ──
    a = np.where(idade <= 5, 1.4, 1.8)
    b = np.where(idade <= 5, 0.70, 0.78)
    ruido = rng.normal(0, 1.2, size=n)
    h = a * np.power(dap, b) + ruido
    h = np.clip(h, 3.0, 45.0)

    # Altura comercial ~ 90% da total (variação por classe)
    h_com = h * rng.uniform(0.82, 0.95, size=n)

    # ── Classe de qualidade por quantis de H/DAP ──
    rhd = h / dap
    q33, q66 = np.quantile(rhd, [0.33, 0.66])
    classe = np.where(rhd >= q66, "I", np.where(rhd >= q33, "II", "III"))

    # ── Volume real (Schumacher-Hall) + ruído de medição ──
    # Importação local evita dependência circular no import do pacote
    from forestpy.dendrometria.volume import schumacher_hall

    volume_teorico = schumacher_hall(dap, h)
    ruido_vol = rng.normal(1.0, 0.05, size=n)  # ±5% de erro de medição
    volume = volume_teorico * ruido_vol

    # ── Parcelas e árvores ──
    parcelas = rng.integers(1, 11, size=n)  # 10 parcelas
    arvores = np.arange(1, n + 1)

    return pd.DataFrame(
        {
            "parcela": parcelas,
            "arvore": arvores,
            "especie": "Eucalyptus grandis",
            "dap": np.round(dap, 2),
            "h": np.round(h, 2),
            "h_com": np.round(h_com, 2),
            "idade": idade,
            "classe": classe,
            "volume": np.round(volume, 4),
        }
    )
