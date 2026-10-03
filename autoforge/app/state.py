"""Thin typed wrapper around st.session_state so pages do not scatter magic keys."""
from __future__ import annotations

import pandas as pd
import streamlit as st

_DATASET_KEYS = ("target", "excluded", "task_override", "result", "bundle", "importance", "prefill", "report_ctx")


def get(key: str, default=None):
    return st.session_state.get(key, default)


def put(key: str, value) -> None:
    st.session_state[key] = value


def set_dataset(df: pd.DataFrame, name: str, warnings: list[str], data_hash: str) -> None:
    """Store a dataset; if it changed, discard everything derived from the previous one."""
    if get("data_hash") != data_hash:
        for k in _DATASET_KEYS:
            st.session_state.pop(k, None)
    put("df", df)
    put("dataset_name", name)
    put("load_warnings", warnings)
    put("data_hash", data_hash)


def has_data() -> bool:
    return isinstance(get("df"), pd.DataFrame)


def require_data() -> bool:
    if not has_data():
        st.info("Load a dataset on the **Data** page first.")
        return False
    return True


def require_target() -> bool:
    if not require_data():
        return False
    if get("target") is None:
        st.info("Choose a target column on the **Data** page first.")
        return False
    return True


def require_result() -> bool:
    if get("result") is None:
        st.info("Train models on the **Train** page first.")
        return False
    return True
