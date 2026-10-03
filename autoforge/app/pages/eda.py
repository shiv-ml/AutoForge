"""Page 3: focused exploratory data analysis (limited auto-plots + user-chosen extras)."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from autoforge.app import state
from autoforge.app.pages.profiling import cached_profile
from autoforge.core.evaluation.problem import detect_problem_type, is_classification
from autoforge.core.profiling.profile import CATEGORICAL, NUMERIC

MAX_AUTO = 6


def render() -> None:
    st.header("3. Explore")
    if not state.require_target():
        return
    df, target = state.get("df"), state.get("target")
    prof = cached_profile(df, state.get("data_hash"))
    task = state.get("task_override") or detect_problem_type(df[target]).task
    excluded = set(state.get("excluded") or [])
    nums = [c for c in prof.numeric_columns if c != target and c not in excluded]
    cats = [c for c in prof.categorical_columns if c != target and c not in excluded and prof.unique_counts[c] <= 30]
    plot_df = df.sample(min(len(df), 5000), random_state=0) if len(df) > 5000 else df
    if len(df) > 5000:
        st.caption("Plots use a random sample of 5,000 rows for speed.")

    tab_t, tab_n, tab_c, tab_r = st.tabs(["Target", "Numeric", "Categorical", "Relationships"])
    with tab_t:
        if is_classification(task):
            vc = df[target].value_counts().reset_index()
            vc.columns = [target, "count"]
            st.plotly_chart(px.bar(vc, x=target, y="count", title="Class balance"), width="stretch")
        else:
            st.plotly_chart(px.histogram(df, x=target, nbins=40, title="Target distribution"), width="stretch")
    with tab_n:
        default = sorted(nums, key=lambda c: -abs(prof.stats.loc[c, "skew"]) if prof.stats.loc[c, "skew"] == prof.stats.loc[c, "skew"] else 0)[:MAX_AUTO]
        chosen = st.multiselect("Numeric columns (most skewed shown by default)", nums, default=default)
        for col in chosen:
            a, b = st.columns(2)
            a.plotly_chart(px.histogram(plot_df, x=col, nbins=40, title=col), width="stretch")
            b.plotly_chart(px.box(plot_df, y=col, title=col), width="stretch")
    with tab_c:
        chosen = st.multiselect("Categorical columns", cats, default=cats[:MAX_AUTO])
        for col in chosen:
            vc = df[col].value_counts().head(15).reset_index()
            vc.columns = [col, "count"]
            if is_classification(task):
                st.plotly_chart(px.histogram(plot_df, x=col, color=target, barmode="group", title=f"{col} by {target}"), width="stretch")
            else:
                st.plotly_chart(px.box(plot_df, x=col, y=target, title=f"{target} by {col}"), width="stretch")
    with tab_r:
        if len(nums) >= 2:
            corr_cols = nums[:25] + ([target] if target in prof.numeric_columns else [])
            st.plotly_chart(px.imshow(df[corr_cols].corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                      title="Correlation matrix (Pearson)"), width="stretch")
            x = st.selectbox("Scatter X", nums)
            y = st.selectbox("Scatter Y", [c for c in nums if c != x] or nums)
            color = target if is_classification(task) else None
            st.plotly_chart(px.scatter(plot_df, x=x, y=y, color=plot_df[color].astype(str) if color else None, opacity=.7), width="stretch")
        else:
            st.info("Need at least two numeric columns for correlation and scatter plots.")
        if nums and not is_classification(task) and target in prof.numeric_columns:
            top = df[nums].corrwith(df[target]).abs().sort_values(ascending=False).head(MAX_AUTO).index.tolist()
            st.subheader("Numeric features vs target (strongest correlations)")
            for col in top:
                st.plotly_chart(px.scatter(plot_df, x=col, y=target, trendline=None, opacity=.6, title=f"{col} vs {target}"), width="stretch")
