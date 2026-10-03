"""Cross-validation of a *complete* pipeline (preprocessing + model) with per-fold fitting."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from autoforge.core.evaluation.metrics import compute_metrics, predict_scores
from autoforge.core.evaluation.problem import is_classification


def make_splitter(task: str, n_folds: int, seed: int):
    """StratifiedKFold for classification, KFold for regression; always shuffled with a fixed seed."""
    cls = StratifiedKFold if is_classification(task) else KFold
    return cls(n_splits=n_folds, shuffle=True, random_state=seed)


@dataclass
class CVResult:
    val: dict[str, list[float]] = field(default_factory=dict)
    train: dict[str, list[float]] = field(default_factory=dict)
    fit_times: list[float] = field(default_factory=list)

    def mean(self, metric: str, split: str = "val") -> float:
        v = np.asarray((self.val if split == "val" else self.train).get(metric, [np.nan]), dtype=float)
        return float(np.nan) if np.all(np.isnan(v)) else float(np.nanmean(v))

    def std(self, metric: str) -> float:
        v = np.asarray(self.val.get(metric, [np.nan]), dtype=float)
        return float(np.nan) if np.all(np.isnan(v)) else float(np.nanstd(v))


def cross_validate(pipeline, X: pd.DataFrame, y: np.ndarray, task: str, splitter, classes=None,
                   with_train: bool = True) -> CVResult:
    """Fit a fresh clone of ``pipeline`` on each training fold and score the held-out fold.

    The preprocessing inside the pipeline is therefore fitted on training folds only.
    """
    res = CVResult()
    for tr, va in splitter.split(X, y):
        model = clone(pipeline)
        t0 = time.perf_counter()
        model.fit(X.iloc[tr], y[tr])
        res.fit_times.append(time.perf_counter() - t0)
        for split, idx, store in (("val", va, res.val), ("train", tr, res.train)):
            if split == "train" and not with_train:
                continue
            Xs, ys = X.iloc[idx], y[idx]
            scores, kind = predict_scores(model, Xs)
            m = compute_metrics(task, ys, model.predict(Xs), scores, kind, classes)
            for k, v in m.items():
                store.setdefault(k, []).append(v)
    return res
