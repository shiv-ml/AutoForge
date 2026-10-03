"""Heuristic over/underfitting indicators. These are hints, not verdicts."""
from __future__ import annotations

import numpy as np


def _better_by(greater_is_better: bool, a: float, b: float) -> float:
    """How much better ``a`` is than ``b`` (positive = a better)."""
    return a - b if greater_is_better else b - a


def diagnose(greater_is_better: bool, train: float, val: float, val_std: float, baseline_val: float,
             test: float | None = None) -> list[str]:
    """Return human-readable diagnostic indicators for one model."""
    notes: list[str] = []
    if np.isnan(train) or np.isnan(val):
        return notes
    scale = max(abs(val), abs(train), 1e-12)
    if _better_by(greater_is_better, train, val) / scale > 0.07:
        notes.append(f"Possible overfitting: training score is noticeably better than cross-validation "
                     f"({train:.3f} vs {val:.3f}). This is an indicator, not proof; more data or stronger "
                     "regularisation may help.")
    if not np.isnan(baseline_val):
        tol = 0.05 * max(abs(baseline_val), scale) + (0.02 if greater_is_better else 0.0)
        val_adv = _better_by(greater_is_better, val, baseline_val)
        train_adv = _better_by(greater_is_better, train, baseline_val)
        if val_adv <= tol and train_adv <= tol:
            notes.append("Possible underfitting: both training and validation scores are close to the naive baseline.")
        elif val_adv <= 0:
            notes.append("Cross-validation score is no better than the naive baseline.")
    if not np.isnan(val_std) and abs(val) > 1e-12 and val_std / abs(val) > 0.10:
        notes.append(f"Unstable cross-validation: fold-to-fold std is {val_std / abs(val):.0%} of the mean "
                     "(high-variance indicator).")
    if test is not None and not np.isnan(test) and not np.isnan(val_std):
        if abs(test - val) > max(2 * val_std, 0.02 * scale):
            notes.append(f"Holdout test score ({test:.3f}) differs from CV ({val:.3f}) by more than fold variation explains.")
    return notes
