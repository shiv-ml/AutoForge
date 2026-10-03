"""Reusable UI pieces."""
from __future__ import annotations

import streamlit as st

from autoforge.config.settings import APP_NAME, APP_TAGLINE
from autoforge.core.profiling.quality import CRITICAL, INFO, QualityIssue

CSS = """
<style>
.block-container { padding-top: 2rem; }
.af-title { font-size:2.9em; font-weight:900; color:#B6C7FF; letter-spacing:.04em; text-align:center; margin-bottom:.1em; }
.af-sub { font-size:1.4em; color:#ffffff; text-align:center; margin-bottom:.6em; }
.af-desc { text-align:center; font-size:1.1em; color:#b0bacf; margin-bottom:1.4em; }
.af-bullets { color:#b0bacf; font-size:1.08em; margin:0 auto 1.6em auto; max-width:560px; text-align:left; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def landing() -> bool:
    """Render the landing page; returns True when the user clicks Start."""
    st.markdown(f"<div class='af-title'>{APP_NAME}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='af-sub'>{APP_TAGLINE}</div>", unsafe_allow_html=True)
    st.markdown("<div class='af-desc'>Upload a CSV, review data quality, and compare cross-validated models "
                "with leakage-safe preprocessing, then explain, export and reuse the winning pipeline.</div>", unsafe_allow_html=True)
    st.markdown("""<ul class='af-bullets'>
      <li>Automatic profiling, data-quality checks and leakage warnings</li>
      <li>sklearn pipelines: preprocessing is fitted on training folds only</li>
      <li>Cross-validated leaderboard, optional Optuna tuning, untouched holdout test set</li>
      <li>Explainability, error analysis, model export and predictions on new data</li></ul>""", unsafe_allow_html=True)
    return st.columns([5, 2, 5])[1].button("Start", width="stretch")


def show_issues(issues: list[QualityIssue]) -> None:
    if not issues:
        st.success("No data-quality problems detected.")
        return
    for i in issues:
        text = f"**{i.code.replace('_', ' ')}** - {i.message}"
        (st.error if i.severity == CRITICAL else st.info if i.severity == INFO else st.warning)(text)
