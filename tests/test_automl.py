import numpy as np
import pandas as pd
import pytest

from autoforge.core.errors import ConfigError, SchemaError
from autoforge.core.evaluation.errors import error_analysis
from autoforge.core.evaluation.trainer import ExperimentConfig, prepare_dataset, run_experiment
from autoforge.core.experiments.tracker import ExperimentStore
from autoforge.core.explainability.importance import global_importance, local_explanation
from autoforge.core.export.artifact import build_bundle, load_bundle, save_bundle
from autoforge.core.reporting.report import build_html_report

CLF_MODELS = ["Logistic Regression", "Random Forest", "Decision Tree"]


def clf_cfg(df, **kw):
    base = dict(target="y", features=[c for c in df.columns if c != "y"], task="binary", primary_metric="f1",
                models=CLF_MODELS, n_folds=3)
    base.update(kw)
    return ExperimentConfig(**base)


@pytest.fixture(scope="module")
def clf_result(clf_df):
    return run_experiment(clf_df, clf_cfg(clf_df))


def test_classification_pipeline(clf_result):
    r = clf_result
    lb = r.leaderboard
    assert list(lb["model"].iloc[:3].isin(CLF_MODELS)) == [True] * 3 and lb["model"].iloc[-1] == "Baseline (dummy)"
    assert lb["cv_score"].iloc[0] >= lb["cv_score"].iloc[1]                          # ranked by primary metric
    assert r.best.cv_mean["f1"] > r.baseline.cv_mean["f1"]
    assert r.best.test["f1"] > 0.6


def test_ranking_follows_selected_metric_direction(clf_df):
    r = run_experiment(clf_df, clf_cfg(clf_df, primary_metric="log_loss", models=["Logistic Regression", "Random Forest"]))
    scores = r.leaderboard["cv_score"].dropna().iloc[:2]
    assert scores.iloc[0] <= scores.iloc[1]                                         # lower log loss is better


def test_regression_pipeline(reg_df):
    cfg = ExperimentConfig("y", ["x1", "x2", "cat"], "regression", "rmse", ["Ridge", "Random Forest"], n_folds=3)
    r = run_experiment(reg_df, cfg)
    assert r.best.test["r2"] > 0.8 and r.best.cv_mean["rmse"] < r.baseline.cv_mean["rmse"]


def test_original_dataset_not_modified(clf_df):
    before = clf_df.copy()
    run_experiment(clf_df, clf_cfg(clf_df, models=["Decision Tree"]))
    pd.testing.assert_frame_equal(before, clf_df)


def test_reproducible_with_seed(clf_df):
    a = run_experiment(clf_df, clf_cfg(clf_df, models=["Random Forest"], seed=3)).leaderboard["cv_score"].iloc[0]
    b = run_experiment(clf_df, clf_cfg(clf_df, models=["Random Forest"], seed=3)).leaderboard["cv_score"].iloc[0]
    assert a == b


def test_missing_target_rows_are_reported_not_hidden(clf_df):
    df = clf_df.copy()
    df.loc[:4, "y"] = None
    prep = prepare_dataset(df, clf_cfg(df))
    assert len(prep.X) == len(df) - 5 and "5 rows were removed" in prep.notes[0]


def test_config_validation(clf_df):
    with pytest.raises(ConfigError):
        prepare_dataset(clf_df, clf_cfg(clf_df, features=["x1", "y"]))
    with pytest.raises(ConfigError):
        prepare_dataset(clf_df, clf_cfg(clf_df, primary_metric="rmse"))
    with pytest.raises(ConfigError):
        prepare_dataset(clf_df, clf_cfg(clf_df, models=["Nope"]))
    bad = clf_df.copy()
    bad.loc[0, "y"] = "singleton"
    with pytest.raises(ConfigError, match="fewer than 2"):
        prepare_dataset(bad, clf_cfg(bad))
    with pytest.raises(ConfigError):
        prepare_dataset(clf_df.assign(y=clf_df["x2"].astype(str) + "x"), ExperimentConfig("y", ["x1"], "regression", "rmse", ["Ridge"]))


def test_tuning_budget(clf_df):
    r = run_experiment(clf_df, clf_cfg(clf_df, models=["Logistic Regression"], budget="Standard"))
    assert r.best.tune_trials > 0 and "C" in r.best.params


def test_export_import_and_predict_new_data(clf_result, clf_df, tmp_path):
    bundle = build_bundle(clf_result)
    model_path, meta_path = save_bundle(bundle, tmp_path)
    loaded = load_bundle(model_path)
    assert meta_path.exists() and "library_versions" in loaded.metadata and loaded.metadata["feature_schema"]["numeric"]
    new = clf_df.drop(columns="y").iloc[:10]
    out, rep = loaded.predict(new)
    assert set(out[f"predicted_y"]) <= {"yes", "no"} and {"proba_yes", "proba_no"} <= set(out.columns) and not rep.messages
    ref, _ = bundle.predict(new)
    assert (ref["predicted_y"] == out["predicted_y"]).all()


def test_prediction_handles_missing_extra_and_bad_columns(clf_result, clf_df):
    bundle = build_bundle(clf_result)
    new = clf_df.drop(columns=["y", "x2"]).iloc[:5].assign(junk=1)
    new["x1"] = new["x1"].astype(object)
    new.loc[new.index[0], "x1"] = "oops"
    out, rep = bundle.predict(new)
    assert len(out) == 5 and rep.missing == ["x2"] and rep.extra == ["junk"] and rep.coerced_to_missing["x1"] == 1
    with pytest.raises(SchemaError):
        bundle.predict(pd.DataFrame({"unrelated": [1, 2]}))


def test_explainability_and_error_analysis(clf_result):
    r = clf_result
    imp = global_importance(r.best.pipeline, r.X_test, r.y_test, r.task, "f1", [0, 1], n_repeats=3)
    assert set(imp["feature"]) == set(r.schema.columns)                              # mapped to source columns
    expl, base, cls = local_explanation(r.best.pipeline, r.X_test.iloc[[0]], r.X_train, r.task)
    assert len(expl) == len(r.schema.columns) and 0 <= base <= 1
    ea = error_analysis(r.best.pipeline, r.X_test, r.y_test, r.task, r.classes)
    assert ea["confusion"].values.sum() == len(r.X_test) and "confidence" in ea["misclassified"]


def test_experiment_store_and_report(clf_result, tmp_path):
    store = ExperimentStore(tmp_path)
    store.save(clf_result)
    rec = store.get(clf_result.id)
    assert rec["dataset_hash"] == clf_result.dataset_hash and rec["seed"] == clf_result.config.seed and "scikit-learn" in rec["versions"]
    assert ExperimentConfig.from_dict(rec["config"]).to_dict() == clf_result.config.to_dict()
    assert len(store.summary()) == 1
    html = build_html_report(clf_result)
    assert "Leaderboard" in html and "Reproducibility" in html


def test_sample_datasets_still_work():
    from pathlib import Path
    from autoforge.core.data.ingestion import load_csv
    root = Path(__file__).resolve().parents[1] / "datasets"
    df = load_csv(root / "diabetes.csv").df
    r = run_experiment(df, ExperimentConfig("target", [c for c in df.columns if c != "target"], "regression", "rmse", ["Ridge"], n_folds=3))
    assert r.best_name == "Ridge"
