"""Page 2: automatic profiling and data-quality analysis."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from autoforge.app import state
from autoforge.app.components.ui import show_issues
from autoforge.core.evaluation.problem import detect_problem_type
from autoforge.core.profiling.profile import NUMERIC, categorical_summary, profile_dataset
from autoforge.core.profiling.quality import analyze_quality, recommended_exclusions


@st.cache_data(show_spinner="Profiling dataset...")
def cached_profile(_df: pd.DataFrame, data_hash: str):
    return profile_dataset(_df)


@st.cache_data(show_spinner="Running data-quality checks...")
def cached_quality(_df: pd.DataFrame, data_hash: str, target: str, task: str):
    return analyze_quality(_df, profile_dataset(_df), target, task)


def render() -> None:
    st.header("2. Profile & Quality")
    if not state.require_target():
        return
    df, target, h = state.get("df"), state.get("target"), state.get("data_hash")
    prof = cached_profile(df, h)
    task = state.get("task_override") or detect_problem_type(df[target]).task

    c = st.columns(5)
    c[0].metric("Rows", f"{prof.n_rows:,}")
    c[1].metric("Columns", prof.n_cols)
    c[2].metric("Duplicate rows", prof.duplicate_rows)
    c[3].metric("Cells missing", f"{sum(prof.missing_values.values()):,}")
    c[4].metric("Memory", f"{prof.memory_mb:.2f} MB")
    st.caption(f"{len(prof.numeric_columns)} numeric - {len(prof.categorical_columns)} categorical - "
               f"{len(prof.datetime_columns)} datetime - {len(prof.identifier_candidates)} suspected identifiers")

    tab_q, tab_cols, tab_detail = st.tabs(["Data quality", "Column statistics", "Column detail"])
    with tab_q:
        issues = cached_quality(df, h, target, task)
        show_issues(issues)
        recs = recommended_exclusions(issues)
        if state.get("excluded") is None:
            state.put("excluded", list(recs))
        st.subheader("Columns excluded from training")
        st.caption("Pre-selected using the findings above. Nothing is removed from your data; exclusions only affect training and you can change them.")
        features = [x for x in df.columns if x != target]
        excluded = st.multiselect("Excluded columns", features, default=[x for x in state.get("excluded") if x in features])
        state.put("excluded", excluded)
        if excluded:
            st.dataframe(pd.DataFrame({"column": excluded, "reason": [recs.get(x, "Excluded manually.") for x in excluded]}),
                         hide_index=True, width="stretch")
    with tab_cols:
        st.dataframe(prof.stats, width="stretch")
    with tab_detail:
        col = st.selectbox("Column", df.columns.tolist())
        if prof.kinds[col] == NUMERIC:
            left, right = st.columns(2)
            left.plotly_chart(px.histogram(df, x=col, nbins=40, title=f"Distribution of {col}"), width="stretch")
            right.plotly_chart(px.box(df, y=col, title=f"Box plot of {col}"), width="stretch")
            st.write(df[col].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_frame().T)
        else:
            top, rare = categorical_summary(df[col])
            st.dataframe(top, width="stretch")
            st.plotly_chart(px.bar(top.reset_index(names=col), x=col, y="count", title=f"Top categories of {col}"), width="stretch")
            st.caption(f"{prof.unique_counts[col]} distinct values; {len(rare)} rare categories (< 1% each).")
