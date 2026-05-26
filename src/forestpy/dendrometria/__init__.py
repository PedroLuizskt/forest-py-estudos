"""
Dendrometria clássica — equações paramétricas consagradas na literatura florestal.

Submódulos:
    volume       — Schumacher-Hall, Spurr, Honer, Naslund
    afilamento   — Funções de taper (Kozak, Garay, Demaerschalk)
    hipsometria  — Curtis, Stoffels, Henricksen, Trorey
    sortimentos  — Algoritmos de toragem
"""
from forestpy.dendrometria.hipsometria import (
    curtis,
    henricksen,
    stoffels,
    trorey,
)
from forestpy.dendrometria.volume import (
    honer,
    schumacher_hall,
    spurr,
)

__all__ = [
    # Volume
    "schumacher_hall",
    "spurr",
    "honer",
    # Hipsometria
    "curtis",
    "stoffels",
    "henricksen",
    "trorey",
]
