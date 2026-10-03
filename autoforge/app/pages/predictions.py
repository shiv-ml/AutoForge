"""Pages 7-8: predictions on new data and export of artifacts / reports."""
from __future__ import annotations

import streamlit as st

from autoforge.app import state
from autoforge.config.settings import APP_NAME
from autoforge.core.data.ingestion import load_csv
from autoforge.core.errors import AutoforgeError
from autoforge.core.export.artifact import bundle_to_bytes, load_bundle, metadata_to_json
from autoforge.core.reporting.report import build_html_report


def render_predict() -> None:
    st.header("7. Predict")
    bundle = state.get("bundle")
    with st.expander("Or load a previously exported model.joblib"):
        st.caption("Only load files you created or trust: joblib files can execute code when opened.")
        up_model = st.file_uploader("model.joblib", type=["joblib"], key="model_upload")
        if up_model is not None:
            try:
                bundle = load_bundle(up_model)
                st.success(f"Loaded {bundle.metadata.get('model')} for target '{bundle.target}'.")
            except (AutoforgeError, OSError, ValueError, KeyError, AttributeError, ImportError) as exc:
                st.error(f"Could not load model: {exc}")
                return
    if bundle is None:
        st.info("Train a model first (or load an exported one).")
        return
    st.write(f"Model: **{bundle.metadata['model']}** - task: {bundle.task} - target: `{bundle.target}`")
    with st.expander("Expected input columns"):
        st.json(bundle.schema.to_dict())
    up = st.file_uploader("Upload new data (CSV) without the target column", type=["csv", "txt"], key="predict_upload")
    if up is None:
        return
    try:
        new = load_csv(up).df
        out, rep = bundle.predict(new)
    except AutoforgeError as exc:
        st.error(str(exc))
        return
    for m in rep.messages:
        st.warning(m)
    st.success(f"Predicted {len(out):,} rows.")
    st.dataframe(out, width="stretch")
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")


def render_export() -> None:
    st.header("8. Export")
    if not state.require_result():
        return
    r, bundle = state.get("result"), state.get("bundle")
    st.write(f"Experiment `{r.id}` - best model **{r.best_name}**.")
    st.caption("The model file contains preprocessing + model together with the feature schema, training config, metrics and library versions. "
               "Loading it later requires the `autoforge` package on the Python path and compatible library versions.")
    a, b, c, d = st.columns(4)
    a.download_button("Model (joblib)", bundle_to_bytes(bundle), f"{r.id}_model.joblib", width="stretch")
    b.download_button("Metadata (JSON)", metadata_to_json(bundle), f"{r.id}_metadata.json", "application/json", width="stretch")
    import json
    d.download_button("Config (JSON)", json.dumps(r.config.to_dict(), indent=2), f"{r.id}_config.json", "application/json", width="stretch")
    from autoforge.app.pages.profiling import cached_profile, cached_quality
    df = state.get("df")
    profile = cached_profile(df, state.get("data_hash"))
    issues = cached_quality(df, state.get("data_hash"), r.config.target, r.task)
    report = build_html_report(r, profile, issues, state.get("importance"), st.session_state.get("error_summary"), APP_NAME)
    c.download_button("Report (HTML)", report, f"{r.id}_report.html", "text/html", width="stretch")
    st.caption("Open the HTML report in a browser and use Print -> Save as PDF for a PDF copy. Tip: visit the Explain and Results pages first so importance and error analysis are included.")
