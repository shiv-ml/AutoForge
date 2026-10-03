"""Small shared helpers."""
from __future__ import annotations

import importlib.metadata as md
import platform
import sys

_LIBS = ("scikit-learn", "pandas", "numpy", "scipy", "joblib", "optuna", "xgboost", "lightgbm", "catboost", "streamlit", "plotly")


def library_versions() -> dict[str, str]:
    """Versions of Python and relevant installed libraries (for reproducibility records)."""
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for lib in _LIBS:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            continue
    return out
