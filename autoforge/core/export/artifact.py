"""Portable model artifact (preprocessing + model + metadata), schema validation and prediction."""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from autoforge.core.errors import SchemaError
from autoforge.core.evaluation.problem import is_classification
from autoforge.core.preprocessing.pipeline import FeatureSchema
from autoforge.core.utils import library_versions

logger = logging.getLogger(__name__)
FORMAT_VERSION = 1


@dataclass
class SchemaReport:
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    coerced_to_missing: dict[str, int] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


@dataclass
class ModelBundle:
    pipeline: object
    task: str
    target: str
    schema: FeatureSchema
    classes: list | None
    metadata: dict

    # -------- schema handling
    def validate_input(self, df: pd.DataFrame, max_missing_share: float = 0.5) -> tuple[pd.DataFrame, SchemaReport]:
        """Align new data to the training schema without altering the caller's frame.

        Missing columns are added as NaN (the saved imputers handle them) unless more than
        ``max_missing_share`` of the required columns are absent; extra columns are ignored and reported.
        """
        rep = SchemaReport()
        required = self.schema.columns
        rep.missing = [c for c in required if c not in df.columns]
        rep.extra = [c for c in df.columns if c not in required]
        if len(rep.missing) / max(len(required), 1) > max_missing_share:
            raise SchemaError(f"{len(rep.missing)} of {len(required)} required columns are missing "
                              f"(e.g. {rep.missing[:5]}). This file does not look like the training data.")
        out = df.reindex(columns=required).copy()
        if rep.missing:
            rep.messages.append(f"Missing columns filled with NaN and imputed by the saved pipeline: {rep.missing}")
        if rep.extra:
            rep.messages.append(f"Ignored columns not used by the model: {rep.extra}")
        for c in self.schema.numeric:
            if c in rep.missing:
                continue
            before = int(out[c].isna().sum())
            out[c] = pd.to_numeric(out[c], errors="coerce")
            newly = int(out[c].isna().sum()) - before
            if newly > 0:
                rep.coerced_to_missing[c] = newly
                rep.messages.append(f"Column '{c}': {newly} non-numeric values were treated as missing.")
        return out, rep

    def predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, SchemaReport]:
        """Return a frame with the original columns plus predictions (and probabilities when available)."""
        X, rep = self.validate_input(df)
        pred = self.pipeline.predict(X)
        out = df.copy().reset_index(drop=True)
        if is_classification(self.task):
            labels = np.asarray(self.classes, dtype=object)
            out[f"predicted_{self.target}"] = labels[np.asarray(pred, dtype=int)]
            if hasattr(self.pipeline, "predict_proba"):
                proba = self.pipeline.predict_proba(X)
                for i, c in enumerate(self.classes):
                    out[f"proba_{c}"] = proba[:, i]
                out["confidence"] = proba.max(axis=1)
        else:
            out[f"predicted_{self.target}"] = pred
        return out, rep


def build_bundle(result, extra_metadata: dict | None = None) -> ModelBundle:
    """Create a :class:`ModelBundle` from an ExperimentResult (final pipeline + full metadata)."""
    best = result.best
    cfg = result.config
    meta = {
        "format_version": FORMAT_VERSION, "experiment_id": result.id, "created_at": result.created_at,
        "task": result.task, "target": cfg.target, "model": result.best_name, "classes": result.classes,
        "hyperparameters": best.params, "primary_metric": cfg.primary_metric,
        "cv_metrics": best.cv_mean, "holdout_test_metrics": best.test,
        "training_config": cfg.to_dict(), "feature_schema": result.schema.to_dict(),
        "dataset_hash": result.dataset_hash, "n_rows": result.n_rows, "refit_on_all_data": cfg.refit_full,
        "library_versions": library_versions(),
    }
    meta.update(extra_metadata or {})
    return ModelBundle(result.final_pipeline, result.task, cfg.target, result.schema, result.classes, meta)


def save_bundle(bundle: ModelBundle, directory: str | Path) -> tuple[Path, Path]:
    """Write ``model.joblib`` and ``metadata.json`` into ``directory``."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = d / "model.joblib", d / "metadata.json"
    joblib.dump(bundle, model_path)
    meta_path.write_text(json.dumps(bundle.metadata, indent=2, default=str))
    return model_path, meta_path


def load_bundle(path: str | Path) -> ModelBundle:
    """Load a bundle. Only load files you trust: joblib uses pickle internally."""
    obj = joblib.load(path)
    if not isinstance(obj, ModelBundle):
        raise SchemaError("File is not an Autoforge model bundle.")
    return obj


def bundle_to_bytes(bundle: ModelBundle) -> bytes:
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    return buf.getvalue()


def metadata_to_json(bundle: ModelBundle) -> str:
    return json.dumps(bundle.metadata, indent=2, default=str)
