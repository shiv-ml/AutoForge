"""Dataset profiling: dtype inference, per-column statistics, DatasetProfile."""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from autoforge.config.settings import THRESHOLDS

NUMERIC, CATEGORICAL, DATETIME = "numeric", "categorical", "datetime"
_ID_NAME_HINTS = ("id", "uuid", "guid", "index", "key", "code", "number", "no")


def _looks_like_datetime(s: pd.Series) -> bool:
    """True if a string column parses as datetimes (>=90% of a sample) and is not plain numbers."""
    non_null = s.dropna()
    if non_null.empty:
        return False
    sample = non_null.astype(str).head(500)
    if sample.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.5:
        return False
    if not sample.str.contains(r"[-/:]", regex=True).mean() > 0.8:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", utc=True)
    return bool(parsed.notna().mean() >= 0.9)


def infer_column_kind(s: pd.Series) -> str:
    """Return 'numeric', 'categorical' or 'datetime' for a column."""
    if pdt.is_bool_dtype(s):
        return CATEGORICAL
    if pdt.is_datetime64_any_dtype(s):
        return DATETIME
    if pdt.is_numeric_dtype(s):
        return NUMERIC
    return DATETIME if _looks_like_datetime(s) else CATEGORICAL


def infer_column_kinds(df: pd.DataFrame) -> dict[str, str]:
    return {c: infer_column_kind(df[c]) for c in df.columns}


def dataset_hash(df: pd.DataFrame) -> str:
    """Stable content hash (columns + values) for reproducibility records."""
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]


def _iqr_bounds(x: pd.Series, k: float) -> tuple[float, float]:
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def column_stats(df: pd.DataFrame, kinds: dict[str, str] | None = None) -> pd.DataFrame:
    """Per-column statistics table (one row per column)."""
    kinds = kinds or infer_column_kinds(df)
    n = len(df)
    rows = []
    for c in df.columns:
        s = df[c]
        nn = s.notna().sum()
        nunique = int(s.nunique(dropna=True))
        row = {
            "column": c, "dtype": str(s.dtype), "kind": kinds[c],
            "missing_%": round(100 * (n - nn) / n, 2), "unique": nunique,
            "cardinality": round(nunique / nn, 4) if nn else np.nan,
        }
        if kinds[c] == NUMERIC and nn:
            x = s.dropna().astype(float)
            lo, hi = _iqr_bounds(x, 1.5)
            row.update(mean=x.mean(), median=x.median(), std=x.std(), min=x.min(), max=x.max(),
                       q25=x.quantile(.25), q75=x.quantile(.75),
                       skew=x.skew() if len(x) > 2 else np.nan,
                       outliers=int(((x < lo) | (x > hi)).sum()))
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")


def categorical_summary(s: pd.Series, top: int = 10) -> tuple[pd.DataFrame, list]:
    """Top categories with frequencies, plus the list of rare categories (< threshold share)."""
    vc = s.value_counts(dropna=True)
    total = vc.sum()
    table = pd.DataFrame({"count": vc.head(top), "share_%": (100 * vc.head(top) / total).round(2)})
    rare = vc[vc / total < THRESHOLDS.rare_category_share].index.tolist()
    return table, rare


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    missing_values: dict[str, int]
    duplicate_rows: int
    duplicate_columns: dict[str, str]          # duplicate -> original
    unique_counts: dict[str, int]
    memory_mb: float
    identifier_candidates: list[str]
    constant_columns: list[str]
    kinds: dict[str, str]
    stats: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    hash: str = ""


def _find_identifiers(df: pd.DataFrame, kinds: dict[str, str]) -> list[str]:
    ids = []
    n = len(df)
    if n < THRESHOLDS.min_rows:
        return ids
    for c in df.columns:
        s = df[c].dropna()
        if s.empty or kinds[c] == DATETIME:
            continue
        ratio = s.nunique() / len(s)
        if ratio < THRESHOLDS.id_unique_ratio:
            continue
        if pdt.is_float_dtype(s):
            continue
        name_hint = any(str(c).lower() == h or str(c).lower().endswith(("_" + h, h.upper())) or
                        str(c).lower().startswith(h + "_") for h in _ID_NAME_HINTS)
        if pdt.is_integer_dtype(s):
            consecutive = s.is_monotonic_increasing and (s.diff().dropna() == 1).mean() > 0.95
            if name_hint or consecutive:
                ids.append(c)
        else:
            ids.append(c)
    return ids


def _duplicate_columns(df: pd.DataFrame) -> dict[str, str]:
    seen: dict[int, list[str]] = {}
    dupes: dict[str, str] = {}
    for c in df.columns:
        key = int(pd.util.hash_pandas_object(df[c], index=False).sum() % (2**62))
        for other in seen.get(key, []):
            if df[c].equals(df[other]):
                dupes[c] = other
                break
        seen.setdefault(key, []).append(c)
    return dupes


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Build a :class:`DatasetProfile`. The input frame is not modified."""
    kinds = infer_column_kinds(df)
    stats = column_stats(df, kinds)
    return DatasetProfile(
        n_rows=len(df), n_cols=df.shape[1],
        numeric_columns=[c for c, k in kinds.items() if k == NUMERIC],
        categorical_columns=[c for c, k in kinds.items() if k == CATEGORICAL],
        datetime_columns=[c for c, k in kinds.items() if k == DATETIME],
        missing_values={c: int(v) for c, v in df.isna().sum().items() if v > 0},
        duplicate_rows=int(df.duplicated().sum()),
        duplicate_columns=_duplicate_columns(df),
        unique_counts={c: int(df[c].nunique(dropna=True)) for c in df.columns},
        memory_mb=round(float(df.memory_usage(deep=True).sum()) / 1e6, 3),
        identifier_candidates=_find_identifiers(df, kinds),
        constant_columns=[c for c in df.columns if df[c].nunique(dropna=True) <= 1],
        kinds=kinds, stats=stats, hash=dataset_hash(df),
    )
