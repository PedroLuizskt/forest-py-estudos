"""
Codificação de variáveis categóricas para Machine Learning.

Modelos numéricos exigem features numéricas. Para variáveis categóricas
(ex.: classe de sítio = {I, II, III}, espécie), aplicam-se transformações
que preservam a informação sem introduzir ordem artificial.

Estratégias implementadas:
    - One-Hot Encoding: cria uma coluna binária por categoria
      (apropriado para categorias nominais sem ordem)
    - Ordinal Encoding: mapeia categorias para inteiros
      (apropriado quando há ordem natural, como classe I > II > III)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class OneHotEncoderForest:
    """
    One-Hot Encoder com persistência das categorias aprendidas.

    Reimplementação minimalista do OneHotEncoder do scikit-learn,
    transparente e adequada ao uso didático do projeto.

    Attributes:
        categories_: Dict {coluna: [categorias na ordem de codificação]}
        column_names_: Nomes das colunas geradas (formato 'coluna_categoria').
    """

    categories_: dict[str, list] = field(default_factory=dict)
    column_names_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "OneHotEncoderForest":
        """
        Aprende as categorias de cada coluna.

        Args:
            df: DataFrame contendo as colunas categóricas.
            cols: Lista de nomes de colunas a codificar.

        Returns:
            self.
        """
        self.categories_ = {}
        self.column_names_ = []

        for col in cols:
            cats = sorted(df[col].dropna().unique().tolist())
            self.categories_[col] = cats
            self.column_names_.extend([f"{col}_{c}" for c in cats])

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Aplica one-hot encoding usando as categorias aprendidas.

        Categorias não vistas no fit são codificadas como zeros em todas
        as colunas daquela variável (estratégia conservadora).

        Args:
            df: DataFrame com as colunas categóricas.

        Returns:
            Array 2D com as colunas codificadas (float64).

        Raises:
            RuntimeError: Se chamado antes de fit().
        """
        if not self.categories_:
            raise RuntimeError("Encoder não foi ajustado. Chame fit() primeiro.")

        colunas: list[np.ndarray] = []
        for col, cats in self.categories_.items():
            for cat in cats:
                colunas.append((df[col] == cat).values.astype(float))

        return np.column_stack(colunas)

    def fit_transform(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        """Atalho: ajusta e transforma em uma chamada."""
        return self.fit(df, cols).transform(df)
