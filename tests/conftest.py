"""Fixtures compartilhadas entre os testes."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def fixed_seed():
    """Fixa seed do numpy antes de cada teste."""
    np.random.seed(42)


@pytest.fixture
def sample_dap_h():
    """Pares (DAP, H) realistas de Eucalyptus."""
    return {
        "dap": np.array([10.0, 15.0, 20.0, 25.0, 30.0]),
        "h": np.array([8.0, 12.0, 15.0, 18.0, 22.0]),
    }


@pytest.fixture
def synthetic_pef():
    """Dataset sintético pequeno para testes (50 árvores)."""
    from forestpy.data.loaders import load_pef_vinhedo

    return load_pef_vinhedo(synthetic_fallback=True, n_synthetic=50, seed=42)


@pytest.fixture
def regression_results():
    """Pares (y_true, y_pred) para testes de viz/diagnostics."""
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0.1, 2.0, size=100)
    y_pred = y_true + rng.normal(0, 0.1, size=100)
    return y_true, y_pred
