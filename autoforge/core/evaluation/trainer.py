"""AutoML orchestration: data preparation, holdout split, CV, tuning, leaderboard, final fit."""
from __future__ import annotations

import logging
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from autoforge.config.settings import BUDGETS, DEFAULT_SEED, THRESHOLDS
from autoforge.core.errors import ConfigError
from autoforge.core.evaluation.cv import cross_validate, make_splitter
from autoforge.core.evaluation.diagnostics import diagnose
from autoforge.core.evaluation.metrics import compute_metrics, metric_options, predict_scores
from autoforge.core.evaluation.problem import TASKS, is_classification
from autoforge.core.models.registry import ModelSpec, get_models
from autoforge.core.optimization.tuner import OPTUNA_AVAILABLE, tune_model
from autoforge.core.preprocessing.pipeline import FeatureSchema, PreprocessConfig, build_preprocessor
from autoforge.core.profiling.profile import dataset_hash
from autoforge.core.utils import library_versions

logger = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]
BASELINE = "Baseline (dummy)"


@dataclass
class ExperimentConfig:
    target: str
    features: list[str]
    task: str
    primary_metric: str
    models: list[str]
    n_folds: int = 5
    test_size: float = 0.2
    budget: str = "Quick"
    seed: int = DEFAULT_SEED
    balanced: bool = False          # class_weight='balanced' for models that support it
    refit_full: bool = True         # refit the winner on train+test after evaluation, for deployment
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = dict(d)
        d["preprocess"] = PreprocessConfig(**d.get("preprocess", {}))
        return cls(**d)


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: np.ndarray
    classes: list | None
    notes: list[str]


@dataclass
class ModelResult:
    name: str
    status: str = "ok"                 # ok | failed | skipped
    message: str = ""
    params: dict = field(default_factory=dict)
    cv_mean: dict[str, float] = field(default_factory=dict)
    cv_std: dict[str, float] = field(default_factory=dict)
    train_mean: dict[str, float] = field(default_factory=dict)
    test: dict[str, float] = field(default_factory=dict)
    fit_time_s: float = 0.0
    inference_ms_per_1k: float = float("nan")
    tune_time_s: float = 0.0
    tune_trials: int = 0
    diagnostics: list[str] = field(default_factory=list)
    pipeline: Pipeline | None = field(default=None, repr=False)   # fitted on the training split only


@dataclass
class ExperimentResult:
    id: str
    created_at: str
    config: ExperimentConfig
    task: str
    classes: list | None
    schema: FeatureSchema
    results: dict[str, ModelResult]
    baseline: ModelResult
    leaderboard: pd.DataFrame
    best_name: str
    final_pipeline: Pipeline = field(repr=False)
    X_train: pd.DataFrame = field(repr=False)
    X_test: pd.DataFrame = field(repr=False)
    y_train: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)
    dataset_hash: str
    n_rows: int
    n_folds_used: int
    notes: list[str]
    warnings: list[str]
    total_time_s: float
    versions: dict[str, str]

    @property
    def best(self) -> ModelResult:
        return self.results[self.best_name]


# ------------------------------------------------------------------ preparation
def validate_config(df: pd.DataFrame, cfg: ExperimentConfig) -> None:
    if cfg.target not in df.columns:
        raise ConfigError(f"Target column '{cfg.target}' not found in the dataset.")
    if cfg.target in cfg.features:
        raise ConfigError("The target column cannot also be a feature.")
    missing = [c for c in cfg.features if c not in df.columns]
    if missing:
        raise ConfigError(f"Selected features not found in dataset: {missing}")
    if not cfg.features:
        raise ConfigError("Select at least one feature.")
    if cfg.task not in TASKS:
        raise ConfigError(f"Unknown task '{cfg.task}'.")
    if cfg.primary_metric not in metric_options(cfg.task):
        raise ConfigError(f"Metric '{cfg.primary_metric}' is not valid for {cfg.task}.")
    if cfg.budget not in BUDGETS:
        raise ConfigError(f"Unknown budget '{cfg.budget}'.")
    if not 2 <= cfg.n_folds <= 20:
        raise ConfigError("Number of folds must be between 2 and 20.")
    if not 0.05 <= cfg.test_size <= 0.5:
        raise ConfigError("Test size must be between 0.05 and 0.5.")
    available = get_models(cfg.task)
    unknown = [m for m in cfg.models if m not in available]
    if unknown:
        raise ConfigError(f"Models not available for {cfg.task}: {unknown}")
    if not cfg.models:
        raise ConfigError("Select at least one model.")


def prepare_dataset(df: pd.DataFrame, cfg: ExperimentConfig) -> PreparedData:
    """Select columns, drop rows with missing target (reported), and encode the target. Input is not modified."""
    validate_config(df, cfg)
    notes: list[str] = []
    sub = df[cfg.features + [cfg.target]].copy()
    n_missing = int(sub[cfg.target].isna().sum())
    if n_missing:
        notes.append(f"{n_missing} rows were removed because the target '{cfg.target}' is missing.")
        sub = sub[sub[cfg.target].notna()]
    X, y_raw = sub[cfg.features], sub[cfg.target]
    if len(X) < THRESHOLDS.min_rows:
        raise ConfigError(f"Only {len(X)} usable rows; need at least {THRESHOLDS.min_rows}.")
    if is_classification(cfg.task):
        try:
            classes = pd.Index(y_raw.unique()).sort_values().tolist()
        except TypeError:
            y_raw = y_raw.astype(str)
            classes = pd.Index(y_raw.unique()).sort_values().tolist()
        y = pd.Categorical(y_raw, categories=classes).codes.astype(int)
        counts = np.bincount(y)
        if len(classes) < 2:
            raise ConfigError("The target has only one class.")
        if counts.min() < 2:
            rare = [classes[i] for i in np.where(counts < 2)[0]]
            raise ConfigError(f"Classes with fewer than 2 rows cannot be split/stratified: {rare}. "
                              "Merge or remove them in your data, or switch the task to regression.")
        if cfg.task == "binary" and len(classes) != 2:
            raise ConfigError(f"Task is 'binary' but the target has {len(classes)} classes.")
        if cfg.task == "multiclass" and len(classes) < 3:
            raise ConfigError("Task is 'multiclass' but the target has fewer than 3 classes.")
        return PreparedData(X, y, classes, notes)
    y_num = pd.to_numeric(y_raw, errors="coerce")
    if y_num.isna().any():
        raise ConfigError("Regression target must be numeric, but some values could not be converted.")
    if not np.isfinite(y_num.to_numpy(dtype=float)).all():
        raise ConfigError("Regression target contains infinite values.")
    return PreparedData(X, y_num.to_numpy(dtype=float), None, notes)


def make_pipeline(spec: ModelSpec, schema: FeatureSchema, cfg: ExperimentConfig, params: dict | None = None) -> Pipeline:
    """Unfitted preprocessing + model pipeline."""
    est = spec.factory(cfg.seed, cfg.balanced)
    if params:
        est.set_params(**params)
    return Pipeline([("prep", build_preprocessor(schema, cfg.preprocess, spec.needs_scaling)), ("model", est)])


def _baseline_spec(task: str) -> ModelSpec:
    factory = ((lambda s, b: DummyClassifier(strategy="prior")) if is_classification(task)
               else (lambda s, b: DummyRegressor(strategy="mean")))
    return ModelSpec(BASELINE, factory)


def _evaluate(spec: ModelSpec, cfg: ExperimentConfig, schema: FeatureSchema, data: dict, cv_split, tune_split,
              budget, class_idx, primary_gib: bool, tune: bool) -> ModelResult:
    """Tune (optional), cross-validate and holdout-evaluate one model."""
    res = ModelResult(spec.name)
    Xtr, ytr, Xte, yte = data["Xtr"], data["ytr"], data["Xte"], data["yte"]
    build = lambda p: make_pipeline(spec, schema, cfg, p)          # noqa: E731
    if tune:
        t = tune_model(spec, build, Xtr, ytr, cfg.task, cfg.primary_metric, primary_gib, tune_split, class_idx,
                       budget.n_trials, budget.timeout_s, cfg.seed)
        res.params, res.tune_time_s, res.tune_trials = t.params, t.seconds, t.n_trials
    pipe = build(res.params)
    cv = cross_validate(pipe, Xtr, ytr, cfg.task, cv_split, class_idx, with_train=True)
    keys = metric_options(cfg.task).keys()
    res.cv_mean = {k: cv.mean(k) for k in keys}
    res.cv_std = {k: cv.std(k) for k in keys}
    res.train_mean = {k: cv.mean(k, "train") for k in keys}
    t0 = time.perf_counter()
    fitted = clone(pipe).fit(Xtr, ytr)
    res.fit_time_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = fitted.predict(Xte)
    res.inference_ms_per_1k = (time.perf_counter() - t0) / max(len(Xte), 1) * 1000 * 1000
    scores, kind = predict_scores(fitted, Xte)
    res.test = compute_metrics(cfg.task, yte, pred, scores, kind, class_idx)
    res.pipeline = fitted
    return res


def _leaderboard(results: dict[str, ModelResult], baseline: ModelResult, cfg: ExperimentConfig) -> pd.DataFrame:
    """Rank by the user's primary metric (cross-validation mean) - never by the test set."""
    metrics = metric_options(cfg.task)
    gib = metrics[cfg.primary_metric].greater_is_better
    rows = []
    for r in list(results.values()) + [baseline]:
        if r.status != "ok":
            continue
        row = {"model": r.name, "cv_score": r.cv_mean[cfg.primary_metric], "cv_std": r.cv_std[cfg.primary_metric],
               "train_score": r.train_mean[cfg.primary_metric], "test_score": r.test[cfg.primary_metric],
               "fit_time_s": round(r.fit_time_s, 3), "inference_ms_per_1k_rows": round(r.inference_ms_per_1k, 2),
               "tune_trials": r.tune_trials, "tune_time_s": round(r.tune_time_s, 1), "best_params": str(r.params),
               "warnings": len(r.diagnostics)}
        for k in metrics:
            row[f"cv_{k}"] = r.cv_mean[k]
            row[f"test_{k}"] = r.test[k]
        rows.append(row)
    df = pd.DataFrame(rows)
    is_base = df["model"] == BASELINE
    body = df[~is_base].sort_values("cv_score", ascending=not gib, na_position="last").reset_index(drop=True)
    body.insert(0, "rank", range(1, len(body) + 1))
    base = df[is_base].copy()
    base.insert(0, "rank", np.nan)
    return pd.concat([body, base], ignore_index=True)


def run_experiment(df: pd.DataFrame, cfg: ExperimentConfig, progress: ProgressFn | None = None) -> ExperimentResult:
    """Execute the full AutoML workflow and return an :class:`ExperimentResult`.

    Leakage controls: the holdout test set is split off first and used only for reporting;
    preprocessing lives inside the pipeline and is refit per fold; tuning only sees training data.
    """
    t_start = time.perf_counter()
    progress = progress or (lambda f, m: None)
    prep = prepare_dataset(df, cfg)
    notes, warns = list(prep.notes), []
    budget = BUDGETS[cfg.budget]
    metrics = metric_options(cfg.task)
    gib = metrics[cfg.primary_metric].greater_is_better
    clf = is_classification(cfg.task)

    Xtr, Xte, ytr, yte = train_test_split(prep.X, prep.y, test_size=cfg.test_size, random_state=cfg.seed,
                                          stratify=prep.y if clf else None)
    n_folds = cfg.n_folds
    if clf:
        min_class = int(np.bincount(ytr).min())
        if min_class < 2:
            raise ConfigError("After the holdout split a class has fewer than 2 training rows; lower the test size.")
        if min_class < n_folds:
            n_folds = min_class
            warns.append(f"Number of CV folds reduced to {n_folds} because the smallest class has only {min_class} training rows.")
    if clf and cfg.primary_metric == "accuracy":
        share = np.bincount(prep.y).min() / len(prep.y)
        if share < THRESHOLDS.imbalance_minority_share:
            warns.append(f"Severe class imbalance (smallest class {share:.1%}); accuracy may be misleading. "
                         "Consider balanced accuracy, F1, PR-AUC or MCC as the primary metric.")
    if cfg.budget != "Quick" and not OPTUNA_AVAILABLE:
        warns.append("Optuna is not installed, so hyperparameter tuning was skipped (default hyperparameters used).")

    schema = FeatureSchema.from_frame(Xtr)
    class_idx = list(range(len(prep.classes))) if clf else None
    cv_split = make_splitter(cfg.task, n_folds, cfg.seed)
    tune_split = make_splitter(cfg.task, n_folds, cfg.seed + 1)      # different folds than the reported CV
    data = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}
    catalogue = get_models(cfg.task)

    steps = len(cfg.models) + 1
    progress(0.0, "Evaluating baseline")
    baseline = _evaluate(_baseline_spec(cfg.task), cfg, schema, data, cv_split, tune_split, budget, class_idx, gib, tune=False)
    results: dict[str, ModelResult] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for i, name in enumerate(cfg.models):
            progress((i + 1) / steps, f"Training {name} ({i + 1}/{len(cfg.models)})")
            spec = catalogue[name]
            if spec.slow_on_large and len(Xtr) > THRESHOLDS.slow_model_row_limit:
                results[name] = ModelResult(name, "skipped", f"Skipped: {name} is too slow above {THRESHOLDS.slow_model_row_limit:,} rows.")
                warns.append(results[name].message)
                continue
            try:
                r = _evaluate(spec, cfg, schema, data, cv_split, tune_split, budget, class_idx, gib, tune=cfg.budget != "Quick")
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                logger.warning("Model %s failed: %s", name, exc)
                results[name] = ModelResult(name, "failed", str(exc))
                warns.append(f"{name} failed and was excluded: {exc}")
                continue
            r.diagnostics = diagnose(gib, r.train_mean[cfg.primary_metric], r.cv_mean[cfg.primary_metric],
                                     r.cv_std[cfg.primary_metric], baseline.cv_mean[cfg.primary_metric], r.test[cfg.primary_metric])
            if r.tune_trials:
                r.diagnostics.append("Tuned on the same training data used for CV, so its CV score is slightly optimistic; "
                                     "the holdout test score is the unbiased estimate.")
            results[name] = r
    ok = [r for r in results.values() if r.status == "ok" and not np.isnan(r.cv_mean[cfg.primary_metric])]
    if not ok:
        raise ConfigError("No model could be trained successfully. See warnings: " + "; ".join(warns))
    board = _leaderboard(results, baseline, cfg)
    best_name = str(board.loc[board["rank"] == 1, "model"].iloc[0])
    best = results[best_name]

    progress(0.97, "Finalising best model")
    final = best.pipeline
    if cfg.refit_full:
        final = clone(best.pipeline).fit(prep.X, prep.y)
        notes.append(f"Reported metrics come from the model trained on the training split only; the exported "
                     f"'{best_name}' pipeline was refit on all {len(prep.X):,} rows using the same hyperparameters.")
    progress(1.0, "Done")
    return ExperimentResult(
        id=uuid.uuid4().hex[:10], created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), config=cfg,
        task=cfg.task, classes=prep.classes, schema=schema, results=results, baseline=baseline, leaderboard=board,
        best_name=best_name, final_pipeline=final, X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        dataset_hash=dataset_hash(df), n_rows=len(prep.X), n_folds_used=n_folds, notes=notes, warnings=warns,
        total_time_s=time.perf_counter() - t_start, versions=library_versions())
