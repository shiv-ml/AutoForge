"""Page 6: global and local explanations on holdout data."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from autoforge.app import state
from autoforge.core.evaluation.problem import is_classification
from autoforge.core.explainability.importance import global_importance, local_explanation, native_importance


def render() -> None:
    st.header("6. Explain")
    if not state.require_result():
        return
    r = state.get("result")
    pipe = r.best.pipeline
    classes_idx = list(range(len(r.classes))) if r.classes else None
    st.caption(f"Explaining **{r.best_name}** (the version trained on the training split) using the held-out test rows. "
               "All importances are attributed to your original input columns.")

    if state.get("importance") is None:
        with st.spinner("Computing permutation importance..."):
            state.put("importance", global_importance(pipe, r.X_test, r.y_test, r.task, r.config.primary_metric, classes_idx, r.config.seed))
    imp = state.get("importance")
    st.subheader("Global feature importance (permutation)")
    st.caption(f"Drop in held-out {r.config.primary_metric} when a column is shuffled. Values near or below zero mean the model does not rely on it.")
    top = imp.head(20).iloc[::-1]
    st.plotly_chart(px.bar(top, x="importance", y="feature", error_x="std", orientation="h"), width="stretch")
    native = native_importance(pipe, r.schema.columns)
    if native is not None:
        with st.expander("Native model importance (aggregated to source columns)"):
            st.caption("Computed from the model's own weights/split gains; can be biased toward high-cardinality or large-scale features. Prefer permutation importance.")
            st.plotly_chart(px.bar(native.head(20).iloc[::-1], x="importance", y="feature", orientation="h"), width="stretch")

    st.subheader("Explain one prediction")
    pos = st.number_input("Holdout row number", 0, len(r.X_test) - 1, 0)
    row = r.X_test.iloc[[int(pos)]]
    expl, base, cls = local_explanation(pipe, row, r.X_train, r.task, seed=r.config.seed)
    actual = r.y_test[int(pos)]
    if is_classification(r.task):
        st.write(f"Predicted **{r.classes[cls]}** with probability/score **{base:.3f}** (actual: {r.classes[int(actual)]}).")
    else:
        st.write(f"Predicted **{base:.3f}** (actual: {actual:.3f}).")
    st.caption("Impact = change in the prediction compared with replacing this feature by typical values from the training data (occlusion what-if; not SHAP).")
    st.plotly_chart(px.bar(expl.head(15).iloc[::-1], x="impact", y="feature", orientation="h", color="impact",
                           color_continuous_scale="RdBu", color_continuous_midpoint=0, hover_data=["value"]), width="stretch")
    st.dataframe(row, width="stretch")
