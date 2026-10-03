import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def clf_df() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], n)
    y = ((x1 + (cat == "a") + rng.normal(scale=.5, size=n)) > 0.5).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "when": pd.date_range("2021-01-01", periods=n).astype(str), "y": np.where(y, "yes", "no")})
    df.loc[rng.choice(n, 20, replace=False), "x1"] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "cat"] = None
    return df


@pytest.fixture(scope="session")
def reg_df() -> pd.DataFrame:
    rng = np.random.RandomState(1)
    n = 250
    x1, x2 = rng.normal(size=n), rng.uniform(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": rng.choice(["u", "v"], n), "y": 3 * x1 + 2 * x2 + rng.normal(scale=.3, size=n)})
