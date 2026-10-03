"""Error-analysis tables for the holdout set."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from autoforge.core.evaluation.metrics import predict_scores
from autoforge.core.evaluation.problem import is_classification


def classification_errors(pipeline, X: pd.DataFrame, y: np.ndarray, classes: list) -> dict:
    labels = list(range(len(classes)))
    pred = pipeline.predict(X)
    scores, kind = predict_scores(pipeline, X)
    names = [str(c) for c in classes]
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=[f"actual {n}" for n in names],
                      columns=[f"pred {n}" for n in names])
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"class": names, "precision": p, "recall": r, "f1": f, "support": s})
    table = X.copy()
    table.insert(0, "actual", [classes[i] for i in y])
    table.insert(1, "predicted", [classes[i] for i in pred])
    if kind == "proba":
        table.insert(2, "confidence", scores.max(axis=1))
    wrong = table[y != pred]
    if "confidence" in wrong:
        wrong = wrong.sort_values("confidence", ascending=False)   # confident mistakes first
    return {"confusion": cm, "per_class": per_class, "misclassified": wrong, "all": table, "has_confidence": kind == "proba"}


def regression_errors(pipeline, X: pd.DataFrame, y: np.ndarray) -> dict:
    pred = pipeline.predict(X)
    table = X.copy()
    table.insert(0, "actual", y)
    table.insert(1, "predicted", pred)
    table.insert(2, "residual", y - pred)
    table.insert(3, "abs_error", np.abs(y - pred))
    return {"all": table, "largest": table.sort_values("abs_error", ascending=False)}


def error_analysis(pipeline, X, y, task, classes=None) -> dict:
    return classification_errors(pipeline, X, y, classes) if is_classification(task) else regression_errors(pipeline, X, y)
