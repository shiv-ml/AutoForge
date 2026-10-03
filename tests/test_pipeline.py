import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from autoforge.core.evaluation.cv import cross_validate, make_splitter
from autoforge.core.evaluation.metrics import compute_metrics
from autoforge.core.preprocessing.pipeline import FeatureSchema, PreprocessConfig, SkewedLog, build_preprocessor


def test_preprocessor_handles_missing_dates_unseen(clf_df):
    X = clf_df.drop(columns="y")
    schema = FeatureSchema.from_frame(X)
    assert schema.datetime == ["when"]
    pre = build_preprocessor(schema, PreprocessConfig(), scale=True).fit(X.iloc[:200])
    new = X.iloc[200:].copy()
    new.loc[new.index[0], "cat"] = "never_seen"
    out = pre.transform(new)
    assert not np.isnan(out).any()
    assert any(n.startswith("when_") for n in pre.get_feature_names_out())


def test_no_leakage_statistics_come_from_train_only():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"a": [1000.0]})
    pre = build_preprocessor(FeatureSchema(numeric=["a"]), PreprocessConfig(log_skewed=False), scale=True).fit(train)
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert scaler.mean_[0] == 2.5            # unaffected by the test row
    imp_pre = build_preprocessor(FeatureSchema(numeric=["a"]), PreprocessConfig(log_skewed=False), scale=False).fit(
        pd.DataFrame({"a": [1.0, np.nan, 3.0]}))
    assert imp_pre.transform(pd.DataFrame({"a": [np.nan]}))[0, 0] == 2.0


def test_scaling_only_when_requested():
    schema = FeatureSchema(numeric=["a"])
    tree = build_preprocessor(schema, PreprocessConfig(), scale=False)
    lin = build_preprocessor(schema, PreprocessConfig(), scale=True)
    assert "scale" not in tree.transformers[0][1].named_steps and "scale" in lin.transformers[0][1].named_steps


def test_skewed_log_mask_learned_on_train():
    t = SkewedLog().fit(np.array([[1.0, -5.0], [2.0, 1.0], [1000.0, 3.0], [2.0, 2.0], [3.0, 2.0]]))
    assert t.mask_.tolist() == [True, False]


def test_one_hot_is_capped():
    X = pd.DataFrame({"c": [f"v{i % 200}" for i in range(2000)]})
    pre = build_preprocessor(FeatureSchema(categorical=["c"]), PreprocessConfig(), False).fit(X)
    assert pre.transform(X).shape[1] <= 30


def test_cv_refits_preprocessing_each_fold(clf_df):
    X = clf_df.drop(columns="y")
    y = (clf_df["y"] == "yes").astype(int).to_numpy()
    pipe = Pipeline([("prep", build_preprocessor(FeatureSchema.from_frame(X), PreprocessConfig(), True)),
                     ("model", LogisticRegression(max_iter=500))])
    cv = cross_validate(pipe, X, y, "binary", make_splitter("binary", 4, 0), [0, 1])
    assert len(cv.val["accuracy"]) == 4 and cv.mean("accuracy") > 0.7
    assert not hasattr(pipe.named_steps["prep"], "transformers_")       # original (unfitted) pipeline untouched


def test_cv_is_deterministic(clf_df):
    X = clf_df.drop(columns="y")
    y = (clf_df["y"] == "yes").astype(int).to_numpy()
    mk = lambda: Pipeline([("prep", build_preprocessor(FeatureSchema.from_frame(X), PreprocessConfig(), True)), ("model", LogisticRegression(max_iter=500))])
    a = cross_validate(mk(), X, y, "binary", make_splitter("binary", 3, 7), [0, 1]).val["f1"]
    b = cross_validate(mk(), X, y, "binary", make_splitter("binary", 3, 7), [0, 1]).val["f1"]
    assert a == b


def test_metrics_classification_and_regression():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics("binary", y, np.array([0, 1, 1, 1]), np.array([[.9, .1], [.4, .6], [.2, .8], [.1, .9]]), "proba", [0, 1])
    assert m["accuracy"] == 0.75 and m["roc_auc"] == 1.0 and not np.isnan(m["log_loss"])
    assert np.isnan(compute_metrics("binary", y, y, None, None, [0, 1])["roc_auc"])
    r = compute_metrics("regression", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 3.0]))
    assert np.isnan(r["mape"]) and np.isclose(r["mae"], 1 / 3)       # MAPE undefined when a true value is 0
    assert np.isnan(compute_metrics("binary", np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([.9, .8, .7]), "decision", [0, 1])["roc_auc"])
