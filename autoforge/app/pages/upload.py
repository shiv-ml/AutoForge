"""Page 1: dataset ingestion and target selection."""
from __future__ import annotations

import streamlit as st

from autoforge.app import state
from autoforge.config.settings import DATASET_DIR
from autoforge.core.data.ingestion import load_csv
from autoforge.core.errors import DataLoadError
from autoforge.core.evaluation.problem import detect_problem_type
from autoforge.core.profiling.profile import dataset_hash


def render() -> None:
    st.header("1. Data")
    builtin = sorted(p.name for p in DATASET_DIR.glob("*.csv")) if DATASET_DIR.exists() else []
    options = (["Sample dataset"] if builtin else []) + ["Upload CSV"]
    source = st.radio("Dataset source", options, horizontal=True)
    result, name = None, None
    try:
        if source == "Sample dataset":
            name = st.selectbox("Available datasets", builtin)
            result = load_csv(DATASET_DIR / name)
        else:
            up = st.file_uploader("Upload a CSV file", type=["csv", "txt"])
            if up is not None:
                name = up.name
                result = load_csv(up)
    except DataLoadError as exc:
        st.error(f"Could not load the dataset: {exc}")
        return
    if result is not None:
        state.set_dataset(result.df, name, result.warnings, dataset_hash(result.df))
    if not state.has_data():
        st.info("Choose a sample dataset or upload a CSV to begin.")
        return

    df = state.get("df")
    st.success(f"Loaded **{state.get('dataset_name')}**: {df.shape[0]:,} rows x {df.shape[1]} columns. Your original data is never modified.")
    for w in state.get("load_warnings", []):
        st.warning(w)
    st.dataframe(df.head(20), width="stretch")

    cols = df.columns.tolist()
    current = state.get("target")
    target = st.selectbox("Target column (what you want to predict)", cols,
                          index=cols.index(current) if current in cols else len(cols) - 1)
    if target != current:
        state.put("excluded", None)
        state.put("task_override", None)
        state.put("result", None)
        state.put("bundle", None)
    state.put("target", target)
    det = detect_problem_type(df[target])
    st.caption(f"Detected task: **{det.task}** ({det.confidence} confidence). " + " ".join(det.reasons) +
               " You can override this on the Train page.")
    st.info("Next: open **Profile & Quality** in the sidebar.")
