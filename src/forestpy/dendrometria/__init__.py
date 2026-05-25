"""
Dendrometria clássica — equações paramétricas consagradas na literatura florestal.

Submódulos:
    volume       — Schumacher-Hall, Spurr, Honer, Naslund
    afilamento   — Funções de taper (Kozak, Garay, Demaerschalk)
    hipsometria  — Curtis, Stoffels, Henricksen, Trorey
    sortimentos  — Algoritmos de toragem
"""
from forestpy.dendrometria.volume import (
    schumacher_hall,
    spurr,
    honer,
)

__all__ = ["schumacher_hall", "spurr", "honer"]
