"""Task-appropriate metrics. Metrics that are mathematically undefined return NaN, never a wrong number."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm

from autoforge.core.evaluation.problem import BINARY, is_classification


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    greater_is_better: bool


CLASSIFICATION_METRICS = {m.key: m for m in [
    Metric("accuracy", "Accuracy", True), Metric("balanced_accuracy", "Balanced accuracy", True),
    Metric("precision", "Precision (binary / macro)", True), Metric("recall", "Recall (binary / macro)", True),
    Metric("f1", "F1 (binary / macro)", True), Metric("roc_auc", "ROC-AUC", True),
    Metric("pr_auc", "PR-AUC (binary only)", True), Metric("log_loss", "Log loss", False),
    Metric("mcc", "Matthews corr. coef.", True)]}
REGRESSION_METRICS = {m.key: m for m in [
    Metric("rmse", "RMSE", False), Metric("mae", "MAE", False), Metric("mse", "MSE", False),
    Metric("r2", "R²", True), Metric("medae", "Median abs. error", False), Metric("mape", "MAPE (%)", False)]}


def metric_options(task: str) -> dict[str, Metric]:
    return CLASSIFICATION_METRICS if is_classification(task) else REGRESSION_METRICS


def default_primary_metric(task: str, imbalanced: bool = False) -> str:
    if not is_classification(task):
        return "rmse"
    return "f1" if imbalanced else "accuracy"


def predict_scores(model, X) -> tuple[np.ndarray | None, str | None]:
    """Return (scores, kind) where kind is 'proba' (n, k) or 'decision' (binary 1-D) or (None, None)."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)), "proba"
    if hasattr(model, "decision_function"):
        d = np.asarray(model.decision_function(X))
        if d.ndim == 1:
            return d, "decision"
    return None, None


def compute_metrics(task: str, y_true, y_pred, scores=None, kind=None, classes=None) -> dict[str, float]:
    """Compute every metric valid for ``task``; unavailable ones are NaN."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not is_classification(task):
        mse = skm.mean_squared_error(y_true, y_pred)
        return {"rmse": float(np.sqrt(mse)), "mae": float(skm.mean_absolute_error(y_true, y_pred)), "mse": float(mse),
                "r2": float(skm.r2_score(y_true, y_pred)), "medae": float(skm.median_absolute_error(y_true, y_pred)),
                "mape": float(skm.mean_absolute_percentage_error(y_true, y_pred) * 100) if np.all(y_true != 0) else np.nan}
    binary = task == BINARY
    avg = "binary" if binary else "macro"
    out = {
        "accuracy": skm.accuracy_score(y_true, y_pred), "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
        "precision": skm.precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": skm.recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": skm.f1_score(y_true, y_pred, average=avg, zero_division=0),
        "mcc": skm.matthews_corrcoef(y_true, y_pred),
        "roc_auc": np.nan, "pr_auc": np.nan, "log_loss": np.nan,
    }
    try:
        if scores is not None and binary:
            pos = scores[:, 1] if kind == "proba" else scores
            out["roc_auc"] = skm.roc_auc_score(y_true, pos)
            out["pr_auc"] = skm.average_precision_score(y_true, pos)
        elif scores is not None and kind == "proba":
            out["roc_auc"] = skm.roc_auc_score(y_true, scores, multi_class="ovr", average="macro", labels=classes)
        if scores is not None and kind == "proba":
            out["log_loss"] = skm.log_loss(y_true, scores, labels=classes)
    except ValueError:
        pass  # e.g. only one class present in this fold -> metric undefined
    return {k: float(v) for k, v in out.items()}


def make_signed_scorer(task: str, metric: str, classes=None) -> Callable:
    """sklearn-style scorer ``f(estimator, X, y)`` where higher is always better."""
    sign = 1.0 if metric_options(task)[metric].greater_is_better else -1.0

    def scorer(est, X, y):
        pred = est.predict(X)
        scores, kind = predict_scores(est, X)
        return sign * compute_metrics(task, y, pred, scores, kind, classes)[metric]
    return scorer
