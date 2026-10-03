"""Page 9: experiment history, comparison and reproduction."""
from __future__ import annotations

import json

import streamlit as st

from autoforge.app import state
from autoforge.core.experiments.tracker import ExperimentStore


def render() -> None:
    st.header("9. Experiments")
    store = ExperimentStore()
    summary = store.summary()
    if summary.empty:
        st.info("No experiments yet. Every training run is saved automatically.")
        return
    st.dataframe(summary, hide_index=True, width="stretch")
    ids = summary["id"].tolist()
    picked = st.multiselect("Compare experiments", ids, default=ids[:2])
    if len(picked) >= 2:
        recs = [store.get(i) for i in picked]
        rows = {}
        for r in recs:
            flat = {"task": r["task"], "target": r["target"], "budget": r["budget"], "seed": r["seed"], "folds": r["n_folds"],
                    "best_model": r["best_model"], "n_features": r["n_features"], "dataset_hash": r["dataset_hash"],
                    "models": ", ".join(r["models_tested"])}
            flat.update({f"cv_{k}": v for k, v in r["final_metrics"]["cv"].items()})
            flat.update({f"test_{k}": v for k, v in r["final_metrics"]["holdout_test"].items()})
            rows[r["id"]] = flat
        import pandas as pd
        st.dataframe(pd.DataFrame(rows).astype(str), width="stretch")

    st.subheader("Reproduce an experiment")
    choice = st.selectbox("Experiment", ids)
    rec = store.get(choice)
    st.json(rec["config"], expanded=False)
    cur = state.get("data_hash")
    if cur != rec["dataset_hash"]:
        st.warning(f"The currently loaded dataset (hash {cur}) differs from the one used in this experiment ({rec['dataset_hash']}). Load the same file to reproduce results exactly.")
    else:
        st.success("Current dataset hash matches this experiment.")
    if st.button("Load this configuration into the Train page"):
        state.put("prefill", rec["config"])
        st.success("Configuration loaded. Open the Train page.")
    st.download_button("Download experiment record (JSON)", json.dumps(rec, indent=2), f"experiment_{choice}.json", "application/json")
