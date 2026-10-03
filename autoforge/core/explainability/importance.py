"""Model-agnostic explanations computed on the *original input columns*.

Permuting raw columns of the complete pipeline means every importance is attributed to a
source feature directly (no one-hot level -> column mapping needed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from autoforge.core.evaluation.metrics import make_signed_scorer, predict_scores
from autoforge.core.evaluation.problem import is_classification


def global_importance(pipeline, X: pd.DataFrame, y: np.ndarray, task: str, metric: str, classes=None,
                      seed: int = 42, n_repeats: int = 5, max_rows: int = 2000) -> pd.DataFrame:
    """Permutation importance: drop in the chosen metric when one source column is shuffled (held-out data)."""
    if len(X) > max_rows:
        idx = np.random.RandomState(seed).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y[idx]
    res = permutation_importance(pipeline, X, y, scoring=make_signed_scorer(task, metric, classes),
                                 n_repeats=n_repeats, random_state=seed, n_jobs=1)
    out = pd.DataFrame({"feature": X.columns, "importance": res.importances_mean, "std": res.importances_std})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def native_importance(pipeline, schema_columns: list[str]) -> pd.DataFrame | None:
    """Native importances, only when they map 1:1 to input columns (not the case after one-hot encoding)."""
    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["prep"]
    values = getattr(model, "feature_importances_", None)
    if values is None:
        coef = getattr(model, "coef_", None)
        values = None if coef is None else np.abs(np.atleast_2d(coef)).mean(axis=0)
    if values is None:
        return None
    names = prep.get_feature_names_out()
    # Aggregate transformed columns back to their source column by name prefix.
    totals = {c: 0.0 for c in schema_columns}
    for name, v in zip(names, values):
        owner = next((c for c in sorted(schema_columns, key=len, reverse=True) if name == c or name.startswith(c + "_")), None)
        if owner is not None:
            totals[owner] += float(abs(v))
    df = pd.DataFrame({"feature": list(totals), "importance": list(totals.values())})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _target_value(pipeline, X: pd.DataFrame, task: str, class_idx: int | None) -> np.ndarray:
    if not is_classification(task):
        return pipeline.predict(X).astype(float)
    scores, kind = predict_scores(pipeline, X)
    if kind == "proba":
        return scores[:, class_idx]
    return (pipeline.predict(X) == class_idx).astype(float)


def local_explanation(pipeline, row: pd.DataFrame, background: pd.DataFrame, task: str,
                      n_background: int = 50, seed: int = 42) -> tuple[pd.DataFrame, float, int | None]:
    """Occlusion-style explanation of one prediction.

    For each feature, replace its value by values drawn from the background data and measure how the
    prediction (probability of the predicted class, or regression output) changes. A positive impact
    means the feature's actual value pushed the prediction up relative to typical values.
    This is a simple, model-agnostic what-if attribution, not SHAP.
    """
    bg = background.sample(min(n_background, len(background)), random_state=seed)
    pred = pipeline.predict(row)[0]
    class_idx = int(pred) if is_classification(task) else None
    base = float(_target_value(pipeline, row, task, class_idx)[0])
    impacts = []
    for col in row.columns:
        variants = pd.concat([row] * len(bg), ignore_index=True)
        variants[col] = bg[col].to_numpy()
        impacts.append(base - float(np.mean(_target_value(pipeline, variants, task, class_idx))))
    df = pd.DataFrame({"feature": row.columns, "value": [row.iloc[0][c] for c in row.columns], "impact": impacts})
    df = df.reindex(df["impact"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return df, base, class_idx
