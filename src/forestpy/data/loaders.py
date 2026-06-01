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

    Modelo gerador (estrutura ecológica realista):
        1. Define-se 30 parcelas, cada uma com classe de sítio fixa (I, II ou III)
        2. Cada parcela recebe ~17 árvores em média
        3. DAP ~ Gamma — média maior em sítios melhores (classe I)
        4. H = a(idade) * DAP^b(idade) + ε
        5. Volume = Schumacher-Hall(DAP, H) * ruído de medição
    """
    rng = np.random.default_rng(seed)

    # ── 1. Parcelas com classe fixa (homogeneidade intra-parcela) ──
    N_PARCELAS = 30
    classes_parcela = rng.choice(["I", "II", "III"], size=N_PARCELAS, p=[0.3, 0.4, 0.3])

    # Distribui árvores entre parcelas (~uniforme)
    parcela_de_cada_arvore = rng.integers(1, N_PARCELAS + 1, size=n)
    classe_de_cada_arvore = np.array([
        classes_parcela[p - 1] for p in parcela_de_cada_arvore
    ])

    # ── 2. DAP: média varia por classe de sítio (realismo ecológico) ──
    # Classe I (melhor sítio) → árvores maiores em média
    shape_por_classe = {"I": 10.0, "II": 8.0, "III": 6.5}
    scale_por_classe = {"I": 2.8, "II": 2.5, "III": 2.2}

    dap = np.array([
        rng.gamma(shape_por_classe[c], scale_por_classe[c])
        for c in classe_de_cada_arvore
    ])
    dap = np.clip(dap, 5.0, 50.0)

    # ── 3. Idade do plantio (varia por parcela) ──
    idade_por_parcela = rng.choice([3, 5, 7, 10], size=N_PARCELAS, p=[0.15, 0.30, 0.35, 0.20])
    idade = np.array([idade_por_parcela[p - 1] for p in parcela_de_cada_arvore])

    # ── 4. Altura por relação alométrica ──
    a = np.where(idade <= 5, 1.4, 1.8)
    b = np.where(idade <= 5, 0.70, 0.78)
    ruido = rng.normal(0, 1.2, size=n)
    h = a * np.power(dap, b) + ruido
    h = np.clip(h, 3.0, 45.0)

    # Altura comercial ~ 82-95% da total
    h_com = h * rng.uniform(0.82, 0.95, size=n)

    # ── 5. Volume real (Schumacher-Hall) + ruído de medição ──
    from forestpy.dendrometria.volume import schumacher_hall

    volume_teorico = schumacher_hall(dap, h)
    ruido_vol = rng.normal(1.0, 0.05, size=n)
    volume = volume_teorico * ruido_vol

    arvores = np.arange(1, n + 1)

    return pd.DataFrame(
        {
            "parcela": parcela_de_cada_arvore,
            "arvore": arvores,
            "especie": "Eucalyptus grandis",
            "dap": np.round(dap, 2),
            "h": np.round(h, 2),
            "h_com": np.round(h_com, 2),
            "idade": idade,
            "classe": classe_de_cada_arvore,
            "volume": np.round(volume, 4),
        }
    )
