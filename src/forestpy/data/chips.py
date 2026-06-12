"""
Gerador de chips raster sintéticos para sensoriamento remoto.

Simula chips de imagem multi-banda compatíveis com produtos como Sentinel-2
(bandas RGB + NIR) para 4 classes de uso/cobertura comuns em paisagens
florestais brasileiras:

    0 = Eucalipto (textura regular, vermelho moderado, NIR alto)
    1 = Pinus     (textura mais densa, vermelho baixo, NIR muito alto)
    2 = Pastagem  (textura suave, verde médio, NIR moderado)
    3 = Solo      (textura quase nula, marrom-vermelho, NIR baixo)

A modelagem dos perfis espectrais é simplificada mas qualitativamente
coerente com assinaturas reais. Reprodutibilidade total via seed.

Referência:
    Jensen, J. R. (2015). *Introductory Digital Image Processing: A Remote
    Sensing Perspective* (4ª ed.). Pearson.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Perfis médios por classe (R, G, B, NIR) em escala 0-1
# Espectros propositalmente sobrepostos para forçar a CNN a usar TEXTURA,
# não apenas estatísticas espectrais médias.
_PROFILES = {
    0: {  # Eucalipto
        "mean": np.array([0.10, 0.18, 0.08, 0.42]),
        "std": np.array([0.040, 0.045, 0.030, 0.060]),
        "texture": 0.06,
    },
    1: {  # Pinus — espectro ligeiramente parecido com Eucalipto
        "mean": np.array([0.09, 0.17, 0.07, 0.46]),
        "std": np.array([0.035, 0.040, 0.030, 0.060]),
        "texture": 0.08,
    },
    2: {  # Pastagem
        "mean": np.array([0.14, 0.25, 0.11, 0.36]),
        "std": np.array([0.040, 0.045, 0.035, 0.055]),
        "texture": 0.04,
    },
    3: {  # Solo — espectro com sobreposição moderada com pastagem
        "mean": np.array([0.22, 0.24, 0.16, 0.28]),
        "std": np.array([0.050, 0.045, 0.040, 0.045]),
        "texture": 0.03,
    },
}

CLASS_NAMES = ["Eucalipto", "Pinus", "Pastagem", "Solo"]


@dataclass
class ChipsDataset:
    """Dataset de chips raster com features e rótulos.

    Attributes:
        X: Array (n, c, h, w) — n chips, c=4 bandas, h×w pixels.
        y: Array (n,) — rótulos inteiros de classe (0..3).
        class_names: Nomes das classes.
    """

    X: np.ndarray
    y: np.ndarray
    class_names: list[str]


def _generate_chip(
    rng: np.random.Generator,
    class_id: int,
    size: int,
) -> np.ndarray:
    """
    Gera um chip raster (4, size, size) para a classe especificada.

    Combina:
    - Cor base do perfil espectral médio da classe
    - Variabilidade entre chips (mesma classe, parcelas diferentes)
    - **Estrutura espacial** específica por classe:
        * Eucalipto: linhas regulares (linhas de plantio)
        * Pinus: textura granular densa
        * Pastagem: gradiente suave
        * Solo: quase uniforme (poucos pixels destoam)
    - Textura intra-chip (ruído pixel-a-pixel)
    """
    profile = _PROFILES[class_id]
    # Cor base do chip
    base = profile["mean"] + rng.normal(0, profile["std"])

    # Inicializa chip com cor base
    chip = np.tile(base[:, None, None], (1, size, size)).astype(np.float32)

    # ─── Estrutura espacial específica por classe ───
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")

    if class_id == 0:  # Eucalipto: linhas de plantio (espaçamento regular)
        spacing = rng.integers(4, 7)
        offset = rng.integers(0, spacing)
        # Linhas claras nas posições do plantio
        line_mask = ((xx + offset) % spacing) < 1
        # Reforça NIR nas linhas (copas), reduz nas entrelinhas (sombra)
        chip[3] += line_mask.astype(np.float32) * 0.05
        chip[0] -= line_mask.astype(np.float32) * 0.015

    elif class_id == 1:  # Pinus: granulação densa (textura forte)
        # Múltiplos blobs pequenos espalhados
        granules = rng.normal(0, 0.04, size=(size, size))
        from scipy.ndimage import gaussian_filter
        granules = gaussian_filter(granules, sigma=1.0)
        chip[3] += granules.astype(np.float32)
        chip[1] += (granules * 0.5).astype(np.float32)

    elif class_id == 2:  # Pastagem: gradiente suave
        gx = rng.uniform(-0.5, 0.5)
        gy = rng.uniform(-0.5, 0.5)
        gradient = (gx * (xx / size - 0.5) + gy * (yy / size - 0.5)) * 0.04
        chip[:] += gradient[None, :, :].astype(np.float32)

    else:  # Solo: manchas raras (afloramentos, vegetação esparsa)
        from scipy.ndimage import gaussian_filter
        n_spots = rng.integers(2, 5)
        spots = np.zeros((size, size), dtype=np.float32)
        for _ in range(n_spots):
            cy, cx = rng.integers(0, size, size=2)
            r = rng.integers(2, 4)
            yy_c, xx_c = np.ogrid[:size, :size]
            mask = (yy_c - cy)**2 + (xx_c - cx)**2 < r**2
            spots[mask] += rng.uniform(0.02, 0.06)
        spots = gaussian_filter(spots, sigma=0.5)
        chip[3] += spots
        chip[1] += spots * 0.5

    # ─── Ruído pixel-a-pixel (textura fina) ───
    from scipy.ndimage import uniform_filter
    noise = rng.normal(0, profile["texture"], size=(4, size, size))
    noise = uniform_filter(noise, size=(1, 2, 2))
    chip = chip + noise.astype(np.float32)

    return np.clip(chip, 0.0, 1.0)


def generate_chips(
    n_per_class: int = 60,
    chip_size: int = 32,
    seed: int = 42,
) -> ChipsDataset:
    """
    Gera um conjunto balanceado de chips sintéticos para as 4 classes.

    Args:
        n_per_class: Número de chips por classe.
        chip_size: Lado do chip (chip_size × chip_size pixels).
        seed: Semente do gerador para reprodutibilidade.

    Returns:
        ChipsDataset com `X` (n, 4, chip_size, chip_size), `y` (n,),
        e nomes das classes.

    Example:
        >>> ds = generate_chips(n_per_class=10, chip_size=16, seed=0)
        >>> ds.X.shape
        (40, 4, 16, 16)
    """
    rng = np.random.default_rng(seed)
    X_list = []
    y_list = []

    for cls in range(4):
        for _ in range(n_per_class):
            X_list.append(_generate_chip(rng, cls, chip_size))
            y_list.append(cls)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Embaralha
    idx = rng.permutation(len(y))
    return ChipsDataset(
        X=X[idx],
        y=y[idx],
        class_names=CLASS_NAMES.copy(),
    )


def compute_ndvi(chip: np.ndarray) -> np.ndarray:
    """
    Calcula o NDVI = (NIR - Red) / (NIR + Red) para um chip ou batch.

    Args:
        chip: Array com 4 bandas. Pode ser (4, h, w) ou (n, 4, h, w).

    Returns:
        NDVI no mesmo formato espacial, com bandas reduzidas.
    """
    if chip.ndim == 3:
        red, nir = chip[0], chip[3]
    elif chip.ndim == 4:
        red, nir = chip[:, 0], chip[:, 3]
    else:
        raise ValueError(f"Esperado 3 ou 4 dimensões, recebido {chip.ndim}.")

    eps = 1e-8
    return (nir - red) / (nir + red + eps)
