"""Pages 4-5: experiment configuration/training and the results workspace (leaderboard, diagnostics, errors)."""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from autoforge.app import state
from autoforge.config.settings import BUDGETS, DEFAULT_SEED
from autoforge.core.errors import AutoforgeError
from autoforge.core.evaluation.errors import error_analysis
from autoforge.core.evaluation.metrics import default_primary_metric, metric_options
from autoforge.core.evaluation.problem import TASKS, detect_problem_type, is_classification
from autoforge.core.evaluation.trainer import ExperimentConfig, run_experiment
from autoforge.core.export.artifact import build_bundle
from autoforge.core.experiments.tracker import ExperimentStore
from autoforge.core.models.registry import get_models, optional_status
from autoforge.core.optimization.tuner import OPTUNA_AVAILABLE
from autoforge.core.preprocessing.pipeline import FeatureSchema, PreprocessConfig, describe_transformations


def render_train() -> None:
    st.header("4. Train")
    if not state.require_target():
        return
    df, target = state.get("df"), state.get("target")
    prefill = state.get("prefill") or {}
    det = detect_problem_type(df[target])
    excluded = set(state.get("excluded") or [])
    all_features = [c for c in df.columns if c != target]

    st.subheader("Task")
    task_default = prefill.get("task") or state.get("task_override") or det.task
    task = st.selectbox("Problem type", TASKS, index=TASKS.index(task_default),
                        help="Auto-detected from target dtype, cardinality and distribution. Override if it is wrong.")
    state.put("task_override", task if task != det.task else None)
    if task != det.task:
        st.warning(f"You overrode the detected type ({det.task}).")
    else:
        st.caption(f"Detected **{det.task}** ({det.confidence} confidence): " + " ".join(det.reasons))

    features = st.multiselect("Features", all_features, default=[c for c in (prefill.get("features") or all_features) if c in all_features and (prefill or c not in excluded)])
    metrics = metric_options(task)
    imbalanced = False
    if is_classification(task):
        share = df[target].value_counts(normalize=True).min()
        imbalanced = share < 0.10
    primary_default = prefill.get("primary_metric") if prefill.get("primary_metric") in metrics else default_primary_metric(task, imbalanced)
    c1, c2, c3 = st.columns(3)
    primary = c1.selectbox("Primary metric (ranking & tuning)", list(metrics), index=list(metrics).index(primary_default),
                           format_func=lambda k: metrics[k].label)
    if imbalanced and primary == "accuracy":
        st.warning("Your classes are imbalanced: accuracy can look high while the minority class is ignored. Consider F1, balanced accuracy or PR-AUC.")
    budget = c2.selectbox("Search budget", list(BUDGETS), index=list(BUDGETS).index(prefill.get("budget", "Quick")))
    c2.caption(BUDGETS[budget].description + ("" if OPTUNA_AVAILABLE or budget == "Quick" else " (Optuna missing: tuning disabled)"))
    seed = int(c3.number_input("Random seed", value=int(prefill.get("seed", DEFAULT_SEED)), step=1))

    catalogue = get_models(task)
    defaults = prefill.get("models") or [m for m in ("Logistic Regression", "Linear Regression", "Ridge", "Random Forest", "HistGradientBoosting") if m in catalogue]
    models = st.multiselect("Models", list(catalogue), default=[m for m in defaults if m in catalogue])
    missing_opt = [k for k, v in optional_status().items() if not v and k in ("xgboost", "lightgbm", "catboost")]
    if missing_opt:
        st.caption(f"Optional models not installed (pip install {' '.join(missing_opt)}): they appear automatically once available.")

    d1, d2 = st.columns(2)
    n_folds = d1.slider("Cross-validation folds", 2, 10, int(prefill.get("n_folds", 5)))
    test_size = d2.slider("Holdout test size", 0.1, 0.4, float(prefill.get("test_size", 0.2)), step=0.05,
                          help="Split off first and never used for model selection or tuning.")
    pc = prefill.get("preprocess", {})
    with st.expander("Preprocessing options"):
        e1, e2, e3 = st.columns(3)
        numeric_impute = e1.selectbox("Numeric imputation", ["median", "mean"], index=["median", "mean"].index(pc.get("numeric_impute", "median")))
        scaler = e1.selectbox("Scaler (linear/KNN/SVM only)", ["standard", "robust"], index=["standard", "robust"].index(pc.get("scaler", "standard")))
        log_skewed = e2.checkbox("log1p for highly skewed non-negative columns", pc.get("log_skewed", True))
        clip = e2.checkbox("Clip outliers (1st/99th percentile)", pc.get("clip_outliers", False))
        cat_impute = e3.selectbox("Categorical imputation", ["most_frequent", "constant"], index=["most_frequent", "constant"].index(pc.get("categorical_impute", "most_frequent")))
        rare = e3.slider("Rare-category threshold", 0.0, 0.1, float(pc.get("rare_category_share", 0.01)), step=0.005)
        balanced = st.checkbox("Class-weight balancing (models that support it)", prefill.get("balanced", imbalanced)) if is_classification(task) else False
        refit = st.checkbox("Refit the winner on all data for the exported model", prefill.get("refit_full", True),
                            help="Reported metrics always come from models that never saw the holdout set.")
    pre = PreprocessConfig(numeric_impute=numeric_impute, scaler=scaler, log_skewed=log_skewed, clip_outliers=clip,
                           categorical_impute=cat_impute, rare_category_share=rare)
    if features:
        with st.expander("Transformations that will be applied", expanded=False):
            for line in describe_transformations(FeatureSchema.from_frame(df[features]), pre):
                st.write("- " + line)
    skipped = [c for c in all_features if c not in features]
    if skipped:
        st.caption(f"Not used for training: {skipped}")

    if st.button("Train models", type="primary", width="stretch"):
        cfg = ExperimentConfig(target=target, features=features, task=task, primary_metric=primary, models=models,
                               n_folds=n_folds, test_size=test_size, budget=budget, seed=seed, balanced=balanced,
                               refit_full=refit, preprocess=pre)
        bar = st.progress(0.0, text="Starting...")
        try:
            result = run_experiment(df, cfg, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
        except AutoforgeError as exc:
            bar.empty()
            st.error(str(exc))
            return
        bar.empty()
        state.put("result", result)
        state.put("bundle", build_bundle(result, {"dataset_name": state.get("dataset_name")}))
        state.put("importance", None)
        state.put("prefill", None)
        ExperimentStore().save(result)
        st.success(f"Done in {result.total_time_s:.1f}s. Best model: **{result.best_name}**. Open **Results** in the sidebar.")


def render_results() -> None:
    st.header("5. Results")
    if not state.require_result():
        return
    r = state.get("result")
    cfg, task = r.config, r.task
    metrics = metric_options(task)
    gib = metrics[cfg.primary_metric].greater_is_better
    for w in r.notes:
        st.info(w)
    for w in r.warnings:
        st.warning(w)

    best = r.best
    c = st.columns(4)
    c[0].metric("Best model", r.best_name)
    c[1].metric(f"CV {metrics[cfg.primary_metric].label}", f"{best.cv_mean[cfg.primary_metric]:.4f}", f"± {best.cv_std[cfg.primary_metric]:.4f}", delta_color="off")
    c[2].metric("Holdout test", f"{best.test[cfg.primary_metric]:.4f}")
    c[3].metric("Total time", f"{r.total_time_s:.1f}s")
    st.caption("Models are ranked by cross-validation on the training split. The holdout test column is for reporting only and was never used to choose or tune models.")

    tab_lb, tab_diag, tab_err = st.tabs(["Leaderboard", "Diagnostics", "Error analysis"])
    with tab_lb:
        lb = r.leaderboard
        extra = st.multiselect("Additional metrics", [f"{p}_{k}" for k in metrics for p in ("cv", "test")], default=[])
        base_cols = ["rank", "model", "cv_score", "cv_std", "train_score", "test_score", "fit_time_s", "inference_ms_per_1k_rows", "tune_trials", "best_params"]
        st.dataframe(lb[base_cols + extra], hide_index=True, width="stretch",
                     column_config={"cv_score": f"CV {cfg.primary_metric}", "train_score": f"train {cfg.primary_metric}", "test_score": f"test {cfg.primary_metric}"})
        plot = lb.dropna(subset=["cv_score"]).copy()
        fig = px.bar(plot, x="model", y="cv_score", error_y="cv_std", title=f"Cross-validated {cfg.primary_metric} ({'higher' if gib else 'lower'} is better)")
        st.plotly_chart(fig, width="stretch")
    with tab_diag:
        st.caption("These are diagnostic indicators from simple heuristics, not definitive conclusions.")
        shown = False
        for name, res in r.results.items():
            if res.status != "ok":
                st.error(f"{name}: {res.message}")
                continue
            if res.diagnostics:
                shown = True
                with st.expander(name, expanded=name == r.best_name):
                    for d in res.diagnostics:
                        st.write("- " + d)
        if not shown:
            st.success("No diagnostic indicators were raised.")
    with tab_err:
        names = [n for n, x in r.results.items() if x.status == "ok"]
        model_name = st.selectbox("Model", names, index=names.index(r.best_name))
        ea = error_analysis(r.results[model_name].pipeline, r.X_test, r.y_test, task, r.classes)
        if is_classification(task):
            a, b = st.columns(2)
            a.plotly_chart(px.imshow(ea["confusion"], text_auto=True, color_continuous_scale="Blues", title="Confusion matrix (holdout)"), width="stretch")
            b.write("**Per-class precision / recall**")
            b.dataframe(ea["per_class"], hide_index=True, width="stretch")
            wrong = ea["misclassified"]
            st.write(f"**Misclassified rows: {len(wrong)} of {len(ea['all'])}**" + (" (most confident mistakes first)" if ea["has_confidence"] else ""))
            st.dataframe(wrong, width="stretch")
        else:
            t = ea["all"]
            a, b = st.columns(2)
            a.plotly_chart(px.scatter(t, x="actual", y="predicted", opacity=.7, title="Predicted vs actual").add_shape(
                type="line", x0=t.actual.min(), y0=t.actual.min(), x1=t.actual.max(), y1=t.actual.max(), line=dict(dash="dash")), width="stretch")
            b.plotly_chart(px.scatter(t, x="predicted", y="residual", opacity=.7, title="Residuals vs predicted"), width="stretch")
            st.plotly_chart(px.histogram(t, x="abs_error", nbins=40, title="Absolute error distribution"), width="stretch")
            st.write("**Largest errors**")
            st.dataframe(ea["largest"].head(25), width="stretch")
        st.session_state["error_summary"] = ea["per_class"] if is_classification(task) else ea["all"][["abs_error"]].describe().T.reset_index()
