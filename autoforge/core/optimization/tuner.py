"""Optuna-based hyperparameter search that optimises cross-validated score on training data only."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from autoforge.core.evaluation.cv import cross_validate
from autoforge.core.models.registry import ModelSpec, Space

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:  # optional dependency
    optuna = None
    OPTUNA_AVAILABLE = False


@dataclass
class TuneResult:
    params: dict = field(default_factory=dict)
    best_value: float = float("nan")
    n_trials: int = 0
    seconds: float = 0.0


def suggest(trial, space: Space) -> dict:
    """Translate the registry's search-space grammar into Optuna suggestions."""
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=spec[3])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        else:
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def tune_model(spec: ModelSpec, build_pipeline: Callable[[dict], object], X, y, task: str, primary: str,
               greater_is_better: bool, splitter, classes, n_trials: int, timeout_s: int, seed: int) -> TuneResult:
    """Run an Optuna study. ``X``/``y`` must be the *training* portion only; the test set is never seen."""
    if not OPTUNA_AVAILABLE or n_trials <= 0 or not spec.space:
        return TuneResult()
    t0 = time.perf_counter()

    def objective(trial):
        params = suggest(trial, spec.space)
        try:
            cv = cross_validate(build_pipeline(params), X, y, task, splitter, classes, with_train=False)
        except ValueError as exc:
            logger.debug("Trial failed for %s: %s", spec.name, exc)
            raise optuna.TrialPruned() from exc
        score = cv.mean(primary)
        if np.isnan(score):
            raise optuna.TrialPruned()
        return score

    study = optuna.create_study(direction="maximize" if greater_is_better else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, gc_after_trial=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return TuneResult(n_trials=len(study.trials), seconds=time.perf_counter() - t0)
    return TuneResult(params=dict(study.best_params), best_value=float(study.best_value),
                      n_trials=len(done), seconds=time.perf_counter() - t0)
