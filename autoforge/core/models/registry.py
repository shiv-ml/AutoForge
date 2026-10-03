"""Model registry: estimators, search spaces and metadata. Optional libraries fail gracefully."""
from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from autoforge.core.evaluation.problem import is_classification

logger = logging.getLogger(__name__)

# Search-space grammar: ("float", lo, hi, log) | ("int", lo, hi) | ("cat", [choices])
Space = dict[str, tuple]


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[int, bool], BaseEstimator]   # (seed, balanced) -> estimator
    space: Space = field(default_factory=dict)
    needs_scaling: bool = False
    slow_on_large: bool = False
    supports_class_weight: bool = False


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _cw(balanced: bool) -> dict[str, Any]:
    return {"class_weight": "balanced"} if balanced else {}


def _tree_space(depth_hi: int = 20) -> Space:
    return {"max_depth": ("int", 2, depth_hi), "min_samples_leaf": ("int", 1, 20)}


def _forest_space() -> Space:
    return {"n_estimators": ("int", 100, 400), "max_depth": ("cat", [None, 5, 10, 20]),
            "min_samples_leaf": ("int", 1, 10), "max_features": ("cat", ["sqrt", "log2", 0.5, 1.0])}


def _gb_space() -> Space:
    return {"n_estimators": ("int", 50, 300), "learning_rate": ("float", 0.01, 0.3, True),
            "max_depth": ("int", 2, 5), "subsample": ("float", 0.6, 1.0, False)}


def _hgb_space() -> Space:
    return {"learning_rate": ("float", 0.01, 0.3, True), "max_iter": ("int", 50, 300),
            "max_leaf_nodes": ("int", 8, 64), "l2_regularization": ("float", 1e-6, 10.0, True)}


def _boost_space() -> Space:
    return {"n_estimators": ("int", 100, 500), "learning_rate": ("float", 0.01, 0.3, True),
            "max_depth": ("int", 2, 8), "subsample": ("float", 0.6, 1.0, False)}


def _classification() -> list[ModelSpec]:
    specs = [
        ModelSpec("Logistic Regression", lambda s, b: LogisticRegression(max_iter=2000, random_state=s, **_cw(b)),
                  {"C": ("float", 1e-3, 100.0, True)}, needs_scaling=True, supports_class_weight=True),
        ModelSpec("KNN", lambda s, b: KNeighborsClassifier(),
                  {"n_neighbors": ("int", 3, 30), "weights": ("cat", ["uniform", "distance"])}, needs_scaling=True),
        ModelSpec("SVM", lambda s, b: SVC(random_state=s, **_cw(b)),
                  {"C": ("float", 1e-2, 100.0, True), "kernel": ("cat", ["rbf", "linear"])},
                  needs_scaling=True, slow_on_large=True, supports_class_weight=True),
        ModelSpec("Decision Tree", lambda s, b: DecisionTreeClassifier(random_state=s, **_cw(b)),
                  {**_tree_space(), "criterion": ("cat", ["gini", "entropy"])}, supports_class_weight=True),
        ModelSpec("Random Forest", lambda s, b: RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=-1, **_cw(b)),
                  _forest_space(), supports_class_weight=True),
        ModelSpec("Extra Trees", lambda s, b: ExtraTreesClassifier(n_estimators=200, random_state=s, n_jobs=-1, **_cw(b)),
                  _forest_space(), supports_class_weight=True),
        ModelSpec("Gradient Boosting", lambda s, b: GradientBoostingClassifier(random_state=s), _gb_space()),
        ModelSpec("HistGradientBoosting", lambda s, b: HistGradientBoostingClassifier(random_state=s, **_cw(b)),
                  _hgb_space(), supports_class_weight=True),
    ]
    if _has("xgboost"):
        from xgboost import XGBClassifier
        specs.append(ModelSpec("XGBoost", lambda s, b: XGBClassifier(random_state=s, n_jobs=-1, verbosity=0), _boost_space()))
    if _has("lightgbm"):
        from lightgbm import LGBMClassifier
        specs.append(ModelSpec("LightGBM", lambda s, b: LGBMClassifier(random_state=s, verbose=-1, n_jobs=-1, **_cw(b)),
                               {**_boost_space(), "num_leaves": ("int", 8, 64)}, supports_class_weight=True))
    if _has("catboost"):
        from catboost import CatBoostClassifier
        specs.append(ModelSpec("CatBoost", lambda s, b: CatBoostClassifier(random_seed=s, verbose=0, allow_writing_files=False),
                               {"iterations": ("int", 100, 500), "learning_rate": ("float", 0.01, 0.3, True), "depth": ("int", 3, 8)}))
    return specs


def _regression() -> list[ModelSpec]:
    specs = [
        ModelSpec("Linear Regression", lambda s, b: LinearRegression(), {}, needs_scaling=True),
        ModelSpec("Ridge", lambda s, b: Ridge(random_state=s), {"alpha": ("float", 1e-3, 1e3, True)}, needs_scaling=True),
        ModelSpec("Lasso", lambda s, b: Lasso(random_state=s, max_iter=10000), {"alpha": ("float", 1e-4, 10.0, True)}, needs_scaling=True),
        ModelSpec("ElasticNet", lambda s, b: ElasticNet(random_state=s, max_iter=10000),
                  {"alpha": ("float", 1e-4, 10.0, True), "l1_ratio": ("float", 0.05, 0.95, False)}, needs_scaling=True),
        ModelSpec("KNN", lambda s, b: KNeighborsRegressor(),
                  {"n_neighbors": ("int", 3, 30), "weights": ("cat", ["uniform", "distance"])}, needs_scaling=True),
        ModelSpec("SVR", lambda s, b: SVR(), {"C": ("float", 1e-2, 100.0, True), "epsilon": ("float", 1e-3, 1.0, True),
                                             "kernel": ("cat", ["rbf", "linear"])}, needs_scaling=True, slow_on_large=True),
        ModelSpec("Decision Tree", lambda s, b: DecisionTreeRegressor(random_state=s), _tree_space()),
        ModelSpec("Random Forest", lambda s, b: RandomForestRegressor(n_estimators=200, random_state=s, n_jobs=-1), _forest_space()),
        ModelSpec("Extra Trees", lambda s, b: ExtraTreesRegressor(n_estimators=200, random_state=s, n_jobs=-1), _forest_space()),
        ModelSpec("Gradient Boosting", lambda s, b: GradientBoostingRegressor(random_state=s), _gb_space()),
        ModelSpec("HistGradientBoosting", lambda s, b: HistGradientBoostingRegressor(random_state=s), _hgb_space()),
    ]
    if _has("xgboost"):
        from xgboost import XGBRegressor
        specs.append(ModelSpec("XGBoost", lambda s, b: XGBRegressor(random_state=s, n_jobs=-1, verbosity=0), _boost_space()))
    if _has("lightgbm"):
        from lightgbm import LGBMRegressor
        specs.append(ModelSpec("LightGBM", lambda s, b: LGBMRegressor(random_state=s, verbose=-1, n_jobs=-1),
                               {**_boost_space(), "num_leaves": ("int", 8, 64)}))
    if _has("catboost"):
        from catboost import CatBoostRegressor
        specs.append(ModelSpec("CatBoost", lambda s, b: CatBoostRegressor(random_seed=s, verbose=0, allow_writing_files=False),
                               {"iterations": ("int", 100, 500), "learning_rate": ("float", 0.01, 0.3, True), "depth": ("int", 3, 8)}))
    return specs


def get_models(task: str) -> dict[str, ModelSpec]:
    """All models available for the task (optional libraries included only if installed)."""
    specs = _classification() if is_classification(task) else _regression()
    return {s.name: s for s in specs}


def optional_status() -> dict[str, bool]:
    return {lib: _has(lib) for lib in ("xgboost", "lightgbm", "catboost", "optuna", "shap")}


DEFAULT_SELECTION = ["Logistic Regression", "Random Forest", "HistGradientBoosting",
                     "Linear Regression", "Ridge"]
