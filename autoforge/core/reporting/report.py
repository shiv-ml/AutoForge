"""Self-contained HTML experiment report (print to PDF from the browser)."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from autoforge.core.preprocessing.pipeline import describe_transformations


def _table(df: pd.DataFrame, max_rows: int = 50) -> str:
    return df.head(max_rows).to_html(index=False, escape=True, float_format=lambda v: f"{v:.4f}", border=0, classes="t")


def build_html_report(result, profile=None, issues=None, importance: pd.DataFrame | None = None,
                      error_summary: pd.DataFrame | None = None, app_name: str = "Autoforge") -> str:
    cfg, best = result.config, result.best
    e = html.escape
    lb_cols = ["rank", "model", "cv_score", "cv_std", "train_score", "test_score", "fit_time_s", "tune_trials", "best_params"]
    parts = [f"<h1>{e(app_name)} experiment report</h1><p>Experiment <b>{e(result.id)}</b> - {e(result.created_at)}</p>"]
    if profile is not None:
        parts.append(f"<h2>Dataset summary</h2><p>{profile.n_rows:,} rows x {profile.n_cols} columns; "
                     f"{len(profile.numeric_columns)} numeric, {len(profile.categorical_columns)} categorical, "
                     f"{len(profile.datetime_columns)} datetime; {profile.duplicate_rows} duplicate rows; "
                     f"dataset hash <code>{e(profile.hash)}</code>.</p>")
    if issues:
        parts.append("<h2>Data-quality findings</h2><ul>" + "".join(f"<li><b>{e(i.severity)}</b>: {e(i.message)}</li>" for i in issues) + "</ul>")
    parts.append(f"<h2>Task and evaluation strategy</h2><p>Task: <b>{e(result.task)}</b>. Target: <b>{e(cfg.target)}</b>. "
                 f"{len(cfg.features)} features. Holdout test set: {cfg.test_size:.0%} (never used for model selection). "
                 f"{result.n_folds_used}-fold {'stratified ' if result.classes else ''}cross-validation on the training split. "
                 f"Primary metric: <b>{e(cfg.primary_metric)}</b>. Seed: {cfg.seed}.</p>")
    parts.append("<h2>Preprocessing and feature engineering</h2><ul>" +
                 "".join(f"<li>{e(t)}</li>" for t in describe_transformations(result.schema, cfg.preprocess)) + "</ul>")
    parts.append(f"<h2>Models and hyperparameter search</h2><p>Models tested: {e(', '.join(cfg.models))}. Budget: {e(cfg.budget)}. "
                 "Hyperparameters were optimised with Optuna (TPE) using cross-validation on training data only.</p>")
    parts.append("<h2>Leaderboard</h2>" + _table(result.leaderboard[lb_cols]))
    parts.append(f"<h2>Best model</h2><p><b>{e(result.best_name)}</b> with parameters <code>{e(str(best.params))}</code>.</p>")
    metrics = pd.DataFrame({"cv_mean": best.cv_mean, "cv_std": best.cv_std, "train": best.train_mean, "holdout_test": best.test})
    parts.append(_table(metrics.rename_axis("metric").reset_index()))
    if importance is not None:
        parts.append("<h2>Feature importance (permutation, holdout data)</h2>" + _table(importance, 20))
    if error_summary is not None:
        parts.append("<h2>Error analysis</h2>" + _table(error_summary))
    warns = result.warnings + result.notes + [f"{n}: {d}" for n, r in result.results.items() for d in r.diagnostics]
    if warns:
        parts.append("<h2>Warnings and notes</h2><ul>" + "".join(f"<li>{e(w)}</li>" for w in warns) + "</ul>")
    parts.append("<h2>Reproducibility</h2><pre>" + e("\n".join(f"{k}: {v}" for k, v in result.versions.items())) +
                 f"\ndataset_hash: {result.dataset_hash}\nseed: {cfg.seed}</pre>")
    style = ("body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1b1f2a}"
             "table.t{border-collapse:collapse;font-size:.85rem;display:block;overflow-x:auto}"
             ".t th,.t td{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left}code,pre{background:#f3f4f8;padding:2px 4px}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{e(app_name)} report</title><style>{style}</style></head><body>{''.join(parts)}<p><small>Generated {datetime.now():%Y-%m-%d %H:%M}</small></p></body></html>"
