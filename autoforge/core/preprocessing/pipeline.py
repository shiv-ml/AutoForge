"""Leakage-safe preprocessing built from sklearn Pipeline/ColumnTransformer.

Every statistic (imputation values, scaling parameters, skew masks, clipping
bounds, category vocabularies) is learned in ``fit`` and therefore only from the
training data of whichever fold/split is being fitted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler

from autoforge.core.errors import ConfigError
from autoforge.core.profiling.profile import CATEGORICAL, DATETIME, NUMERIC, infer_column_kinds


@dataclass
class PreprocessConfig:
    numeric_impute: str = "median"          # "median" | "mean"
    scaler: str = "standard"                # "standard" | "robust"; applied only to models that need it
    log_skewed: bool = True                 # log1p for non-negative columns with |skew| > threshold
    skew_threshold: float = 1.0
    clip_outliers: bool = False             # winsorise to [1st, 99th] percentile learned on train
    categorical_impute: str = "most_frequent"  # "most_frequent" | "constant"
    rare_category_share: float = 0.01       # grouped into an 'infrequent' bucket
    max_categories: int = 30                # hard cap on one-hot width per column (no feature explosion)
    date_features: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureSchema:
    """Input contract of a trained model."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical + self.datetime

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_frame(cls, X: pd.DataFrame) -> "FeatureSchema":
        kinds = infer_column_kinds(X)
        return cls(numeric=[c for c in X.columns if kinds[c] == NUMERIC],
                   categorical=[c for c in X.columns if kinds[c] == CATEGORICAL],
                   datetime=[c for c in X.columns if kinds[c] == DATETIME])


# ---------------------------------------------------------------- transformers
class DateFeatures(BaseEstimator, TransformerMixin):
    """Expand datetime columns into year/month/day/day_of_week/quarter/is_weekend."""

    PARTS = ("year", "month", "day", "day_of_week", "quarter", "is_weekend")

    def fit(self, X, y=None):
        self.columns_ = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(np.shape(X)[1])]
        self.n_features_in_ = len(self.columns_)
        return self

    def transform(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns_)
        out = []
        for c in self.columns_:
            s = pd.to_datetime(X[c], errors="coerce", utc=True)
            dow = s.dt.dayofweek.astype(float)
            out += [s.dt.year.astype(float), s.dt.month.astype(float), s.dt.day.astype(float), dow,
                    s.dt.quarter.astype(float), (dow >= 5).astype(float).where(dow.notna())]
        return np.column_stack([o.to_numpy(dtype=float) for o in out])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_{p}" for c in self.columns_ for p in self.PARTS], dtype=object)


class SkewedLog(BaseEstimator, TransformerMixin):
    """log1p on columns that were non-negative and highly skewed *in the training data*."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        skew = np.atleast_1d(sps.skew(X, axis=0, nan_policy="omit"))
        self.mask_ = (np.nanmin(X, axis=0) >= 0) & (np.abs(skew) > self.threshold)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, self.mask_] = np.log1p(np.clip(X[:, self.mask_], 0, None))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip to percentile bounds learned on the training data."""

    def __init__(self, lower: float = 1.0, upper: float = 99.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_, self.high_ = np.nanpercentile(X, [self.lower, self.upper], axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _inf_to_nan(X):
    X = np.array(X, dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan
    return X


def _to_str(X):
    df = pd.DataFrame(X)
    return df.astype(str).where(df.notna(), np.nan).to_numpy(dtype=object)


# -------------------------------------------------------------------- builders
def build_preprocessor(schema: FeatureSchema, cfg: PreprocessConfig, scale: bool) -> ColumnTransformer:
    """Create an *unfitted* ColumnTransformer for the given schema.

    ``scale`` should be False for tree-based models (scaling is unnecessary for them).
    """
    transformers = []
    if schema.numeric:
        steps = [("inf", FunctionTransformer(_inf_to_nan, feature_names_out="one-to-one")),
                 ("impute", SimpleImputer(strategy=cfg.numeric_impute))]
        if cfg.clip_outliers:
            steps.append(("clip", Winsorizer()))
        if cfg.log_skewed:
            steps.append(("log", SkewedLog(cfg.skew_threshold)))
        if scale:
            steps.append(("scale", RobustScaler() if cfg.scaler == "robust" else StandardScaler()))
        transformers.append(("num", Pipeline(steps), schema.numeric))
    if schema.categorical:
        imputer = (SimpleImputer(strategy="constant", fill_value="missing", missing_values=np.nan)
                   if cfg.categorical_impute == "constant"
                   else SimpleImputer(strategy="most_frequent", missing_values=np.nan))
        transformers.append(("cat", Pipeline([
            ("str", FunctionTransformer(_to_str, feature_names_out="one-to-one")),
            ("impute", imputer),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=cfg.rare_category_share,
                                     max_categories=cfg.max_categories, sparse_output=False)),
        ]), schema.categorical))
    if schema.datetime and cfg.date_features:
        transformers.append(("date", Pipeline([
            ("expand", DateFeatures()),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]), schema.datetime))
    if not transformers:
        raise ConfigError("No usable feature columns remain after applying the schema.")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def describe_transformations(schema: FeatureSchema, cfg: PreprocessConfig, scaled_models: bool = True) -> list[str]:
    """Human-readable list of what the pipeline does (shown in the UI and report)."""
    out = []
    if schema.numeric:
        t = f"{len(schema.numeric)} numeric columns: ±inf -> missing, {cfg.numeric_impute} imputation"
        if cfg.clip_outliers:
            t += ", 1st/99th percentile clipping"
        if cfg.log_skewed:
            t += f", log1p on non-negative columns with |skew| > {cfg.skew_threshold}"
        if scaled_models:
            t += f", {cfg.scaler} scaling (only for models that need it: linear, KNN, SVM)"
        out.append(t)
    if schema.categorical:
        out.append(f"{len(schema.categorical)} categorical columns: {cfg.categorical_impute} imputation, one-hot encoding; "
                   f"levels under {cfg.rare_category_share:.0%} grouped as 'infrequent', max {cfg.max_categories} columns each")
    if schema.datetime and cfg.date_features:
        out.append(f"{len(schema.datetime)} datetime columns -> year, month, day, day_of_week, quarter, is_weekend")
    elif schema.datetime:
        out.append(f"{len(schema.datetime)} datetime columns are ignored (date features disabled)")
    out.append("All statistics are learned on training folds only; the same fitted pipeline is saved with the model.")
    return out
