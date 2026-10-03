"""Problem-type detection from target dtype, cardinality and distribution."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as pdt

BINARY, MULTICLASS, REGRESSION = "binary", "multiclass", "regression"
TASKS = (BINARY, MULTICLASS, REGRESSION)


def is_classification(task: str) -> bool:
    return task in (BINARY, MULTICLASS)


@dataclass
class ProblemDetection:
    task: str
    confidence: str            # "high" | "medium" | "low"
    reasons: list[str] = field(default_factory=list)


def detect_problem_type(y: pd.Series) -> ProblemDetection:
    """Infer the task. The caller must always let the user override it."""
    y = y.dropna()
    n, k = len(y), int(y.nunique())
    if n == 0 or k < 2:
        return ProblemDetection(MULTICLASS, "low", [f"Target has {k} distinct value(s); cannot train a model."])
    if not pdt.is_numeric_dtype(y) or pdt.is_bool_dtype(y):
        task = BINARY if k == 2 else MULTICLASS
        return ProblemDetection(task, "high", [f"Target dtype is {y.dtype} (non-numeric) with {k} classes."])
    reasons = [f"Numeric target with {k} distinct values over {n} rows ({k / n:.1%} unique)."]
    if k == 2:
        return ProblemDetection(BINARY, "high", reasons + ["Exactly two distinct values."])
    values = y.to_numpy(dtype=float)
    integer_valued = bool(np.all(np.isclose(values, np.round(values))))
    if not integer_valued:
        return ProblemDetection(REGRESSION, "high", reasons + ["Contains non-integer values."])
    ratio = k / n
    if k <= 10 and ratio < 0.05:
        return ProblemDetection(MULTICLASS, "medium", reasons + ["Few integer levels relative to rows; looks like class labels."])
    if k <= 20 and ratio < 0.02:
        return ProblemDetection(MULTICLASS, "low", reasons + ["Integer target with limited levels; could be counts. Please confirm."])
    return ProblemDetection(REGRESSION, "medium" if k > 30 else "low",
                            reasons + ["Integer values with many distinct levels; treated as a count/continuous target."])
