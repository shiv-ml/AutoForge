"""Autoforge - entry point. Run with:  streamlit run app.py"""
import logging

import streamlit as st

from autoforge.app import state
from autoforge.app.components.ui import inject_css, landing
from autoforge.app.pages import automl, eda, experiments, explainability, predictions, profiling, upload
from autoforge.config.settings import APP_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
st.set_page_config(page_title=APP_NAME, layout="wide")
inject_css()

PAGES = {
    "1. Data": upload.render,
    "2. Profile & Quality": profiling.render,
    "3. Explore": eda.render,
    "4. Train": automl.render_train,
    "5. Results": automl.render_results,
    "6. Explain": explainability.render,
    "7. Predict": predictions.render_predict,
    "8. Export": predictions.render_export,
    "9. Experiments": experiments.render,
}

if not state.get("started"):
    if landing():
        state.put("started", True)
        st.rerun()
else:
    st.sidebar.title(APP_NAME)
    page = st.sidebar.radio("Workflow", list(PAGES), label_visibility="collapsed")
    if state.has_data():
        st.sidebar.caption(f"Dataset: {state.get('dataset_name')}\n\nTarget: {state.get('target')}")
    PAGES[page]()
