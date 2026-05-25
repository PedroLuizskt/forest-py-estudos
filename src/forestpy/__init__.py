"""
forestpy — Estudos avançados em Machine Learning aplicado à Engenharia Florestal.

Pacote modular que combina dendrometria clássica, inventário florestal,
classificação de sítio e modelagem com redes neurais (MLP, CNN) para
análises do PEF Vinhedo e demais bases florestais brasileiras.

Subpacotes:
    data           — Loaders, validadores Pydantic e geradores sintéticos
    dendrometria   — Volume, afilamento, hipsometria, sortimentos
    inventario     — Amostragem e distribuições diamétricas
    sitio          — Curvas de crescimento e índices de produtividade
    ml             — Pipelines de ML clássico e deep learning (PyTorch)
    viz            — Toolkit de visualização (matplotlib, seaborn, plotly)
    utils          — Configuração, I/O, logging, reprodutibilidade
"""

__version__ = "0.1.0"
__author__ = "Pedro Luiz R. Vaz de Melo"

# Expõe os módulos principais para imports curtos:
#   from forestpy import dendrometria, ml, viz
from forestpy import dendrometria, inventario, ml, sitio, utils, viz
from forestpy import data as fpdata  # 'data' evita conflito com pandas idioms

__all__ = [
    "__version__",
    "__author__",
    "fpdata",
    "dendrometria",
    "inventario",
    "ml",
    "sitio",
    "utils",
    "viz",
]
