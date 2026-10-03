"""Data-quality analysis. Produces findings and *suggested* exclusions; never drops data."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from autoforge.config.settings import DEFAULT_SEED, THRESHOLDS
from autoforge.core.evaluation.problem import is_classification
from autoforge.core.profiling.profile import CATEGORICAL, NUMERIC, DatasetProfile

logger = logging.getLogger(__name__)
INFO, WARNING, CRITICAL = "info", "warning", "critical"


@dataclass
class QualityIssue:
    severity: str
    code: str
    column: str | None
    message: str
    suggest_exclude: bool = False


def analyze_quality(df: pd.DataFrame, profile: DatasetProfile, target: str | None = None,
                    task: str | None = None) -> list[QualityIssue]:
    """Run all checks. ``target``/``task`` enable target-specific checks."""
    T = THRESHOLDS
    issues: list[QualityIssue] = []
    n = profile.n_rows
    stats = profile.stats

    def add(sev, code, col, msg, excl=False):
        issues.append(QualityIssue(sev, code, col, msg, excl))

    if profile.duplicate_rows:
        add(WARNING, "duplicate_rows", None,
            f"{profile.duplicate_rows} duplicate rows ({profile.duplicate_rows / n:.1%}). "
            "They were NOT removed. Duplicates can inflate cross-validation scores if copies land in both folds.")
    for dup, orig in profile.duplicate_columns.items():
        add(WARNING, "duplicate_column", dup, f"Column '{dup}' has identical values to '{orig}'; excluded as redundant.",
            dup != target and orig != target)

    for c in df.columns:
        if c == target:
            continue
        pct = stats.loc[c, "missing_%"]
        if pct >= T.missing_exclude_pct:
            add(WARNING, "missing_high", c, f"Column '{c}' is {pct:.1f}% missing; excluded because imputing most of a column is unreliable.", True)
        elif pct > T.missing_warn_pct:
            add(INFO, "missing", c, f"Column '{c}' has {pct:.1f}% missing values; they will be imputed inside the training pipeline.")
        if c in profile.constant_columns:
            add(WARNING, "constant", c, f"Column '{c}' is constant and carries no information; excluded.", True)
            continue
        s = df[c].dropna()
        if len(s) and c not in profile.duplicate_columns:
            top_share = s.value_counts(normalize=True).iloc[0]
            if top_share >= T.near_constant_share:
                add(WARNING, "near_constant", c, f"Column '{c}' is near-constant ({top_share:.1%} one value); excluded.", True)
        if c in profile.identifier_candidates:
            ratio = profile.unique_counts[c] / max(s.shape[0], 1)
            add(WARNING, "identifier", c, f"Column '{c}' was excluded because it contains {ratio:.1%} unique values and appears to be an identifier.", True)
            continue
        kind = profile.kinds[c]
        if kind == CATEGORICAL and not pdt.is_bool_dtype(df[c]):
            nun = profile.unique_counts[c]
            avg_len = s.astype(str).str.len().mean() if len(s) else 0
            if avg_len >= T.text_avg_len and nun / max(len(s), 1) > 0.5:
                add(WARNING, "free_text", c, f"Column '{c}' looks like free text (avg {avg_len:.0f} chars, mostly unique). Text modelling is not supported; excluded.", True)
            elif nun > T.high_cardinality:
                add(WARNING, "high_cardinality", c, f"Categorical column '{c}' has {nun} levels (> {T.high_cardinality}). Rare levels are grouped, but consider excluding it.")
            as_num = pd.to_numeric(s, errors="coerce")
            if len(s) and as_num.notna().mean() > 0.95:
                add(WARNING, "incompatible_dtype", c, f"Column '{c}' is stored as text but {as_num.notna().mean():.0%} of values are numeric; it will be treated as categorical. Fix the source data if it should be numeric.")
        if kind == NUMERIC and stats.loc[c, "std"] > 0:
            x = df[c].dropna().astype(float)
            q1, q3 = x.quantile(.25), x.quantile(.75)
            iqr = q3 - q1
            if iqr > 0:
                share = ((x < q1 - T.extreme_outlier_iqr * iqr) | (x > q3 + T.extreme_outlier_iqr * iqr)).mean()
                if share > T.extreme_outlier_share:
                    add(INFO, "outliers", c, f"Column '{c}' has {share:.1%} extreme outliers (>{T.extreme_outlier_iqr} IQR). Consider enabling outlier clipping in the preprocessing options.")

    if target is not None:
        issues += _target_checks(df, profile, target, task)
        if task:
            issues += _leakage_checks(df, profile, target, task, {i.column for i in issues if i.suggest_exclude})
    order = {CRITICAL: 0, WARNING: 1, INFO: 2}
    return sorted(issues, key=lambda i: order[i.severity])


def _target_checks(df, profile, target, task) -> list[QualityIssue]:
    out: list[QualityIssue] = []
    y = df[target]
    pct = y.isna().mean() * 100
    if pct > 0:
        out.append(QualityIssue(WARNING if pct < 20 else CRITICAL, "target_missing", target,
                                f"Target '{target}' has {pct:.1f}% missing values. These rows cannot be used and will be reported (not silently hidden) at training time."))
    k = y.nunique(dropna=True)
    if k < 2:
        out.append(QualityIssue(CRITICAL, "target_constant", target, f"Target '{target}' has {k} distinct value(s); nothing to learn."))
    if target in profile.identifier_candidates:
        out.append(QualityIssue(CRITICAL, "target_identifier", target, f"Target '{target}' looks like an identifier ({k} unique values); it is probably the wrong column."))
    if task and is_classification(task):
        shares = y.value_counts(normalize=True)
        if len(shares) >= 2 and shares.min() < THRESHOLDS.imbalance_minority_share:
            out.append(QualityIssue(WARNING, "imbalance", target,
                                    f"Class imbalance: smallest class '{shares.idxmin()}' is {shares.min():.1%} of rows. Accuracy can be misleading; prefer balanced accuracy, F1, PR-AUC or MCC."))
        if k > 50:
            out.append(QualityIssue(WARNING, "target_many_classes", target, f"Target has {k} classes; if it is really continuous, switch the task to regression."))
    return out


def _leakage_checks(df, profile, target, task, already_excluded, max_cols: int = 40, max_rows: int = 5000) -> list[QualityIssue]:
    """Flag features that predict the target almost perfectly on their own (single shallow tree, 3-fold CV)."""
    y = df[target].dropna()
    if y.nunique() < 2 or len(y) < 30:
        return []
    sample = df.loc[y.index]
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=DEFAULT_SEED)
    y_s = sample[target]
    cols = [c for c in df.columns if c != target and c not in already_excluded and profile.kinds[c] != "datetime"][:max_cols]
    clf = is_classification(task)
    y_enc = pd.factorize(y_s)[0] if clf else pd.to_numeric(y_s, errors="coerce")
    ok = pd.Series(y_enc).notna().to_numpy()
    out = []
    for c in cols:
        x = sample[c]
        if profile.kinds[c] == CATEGORICAL:
            if profile.unique_counts[c] > THRESHOLDS.high_cardinality:
                continue
            x = pd.factorize(x.astype(str))[0]
        x = pd.to_numeric(pd.Series(x, index=sample.index), errors="coerce").fillna(-1e9).to_numpy().reshape(-1, 1)
        model = (DecisionTreeClassifier(max_depth=4, random_state=DEFAULT_SEED) if clf
                 else DecisionTreeRegressor(max_depth=4, random_state=DEFAULT_SEED))
        folds = 3
        if clf and np.bincount(y_enc[ok]).min() < folds:
            continue
        try:
            score = cross_val_score(model, x[ok], np.asarray(y_enc)[ok], cv=folds,
                                    scoring="balanced_accuracy" if clf else "r2").mean()
        except ValueError as exc:
            logger.debug("Leakage probe skipped for %s: %s", c, exc)
            continue
        if score >= THRESHOLDS.leakage_score:
            out.append(QualityIssue(WARNING, "leakage", c,
                                    f"Potential leakage: '{c}' alone predicts the target almost perfectly "
                                    f"(CV {'balanced accuracy' if clf else 'R²'} {score:.3f}). It may be derived from the target or only known after the outcome. NOT excluded automatically; review it."))
    return out


def recommended_exclusions(issues: list[QualityIssue]) -> dict[str, str]:
    """Columns the pipeline recommends excluding, mapped to the explanation shown to the user."""
    return {i.column: i.message for i in issues if i.suggest_exclude and i.column}
