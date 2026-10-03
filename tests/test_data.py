import numpy as np
import pandas as pd
import pytest

from autoforge.core.data.ingestion import load_csv
from autoforge.core.errors import DataLoadError
from autoforge.core.evaluation.problem import BINARY, MULTICLASS, REGRESSION, detect_problem_type
from autoforge.core.profiling.profile import CATEGORICAL, DATETIME, NUMERIC, infer_column_kinds, profile_dataset
from autoforge.core.profiling.quality import analyze_quality, recommended_exclusions


def csv_bytes(df): return df.to_csv(index=False).encode()


def test_load_ok(clf_df):
    assert load_csv(csv_bytes(clf_df)).df.shape == clf_df.shape


@pytest.mark.parametrize("raw", [b"", b"   \n  ", b"a,b\n", b"\x00\x01\x02binary"])
def test_empty_or_unusable(raw):
    with pytest.raises(DataLoadError):
        load_csv(raw)


def test_malformed_csv():
    with pytest.raises(DataLoadError):
        load_csv(b"a,b\n1,2\n1,2,3,4,5\n6,7\n\"unterminated,3\n")


def test_single_column_rejected():
    with pytest.raises(DataLoadError):
        load_csv(b"a\n1\n2\n")


def test_encoding_fallback_and_duplicate_header():
    res = load_csv("name,a,a\ncaf\xe9,1,2\n".encode("latin-1"))
    assert res.encoding != "utf-8" and any("Duplicate column" in w for w in res.warnings)


def test_semicolon_delimiter():
    assert load_csv(b"a;b\n1;2\n3;4\n").df.shape == (2, 2)


def test_dtype_inference(clf_df):
    k = infer_column_kinds(clf_df)
    assert k["x1"] == NUMERIC and k["cat"] == CATEGORICAL and k["when"] == DATETIME


def test_profile_does_not_modify_and_counts(clf_df):
    before = clf_df.copy()
    p = profile_dataset(clf_df)
    pd.testing.assert_frame_equal(before, clf_df)
    assert p.missing_values["x1"] == 20 and p.n_rows == 300


def test_quality_detects_problems(clf_df):
    df = clf_df.copy()
    df["cust_id"] = [f"C{i}" for i in range(len(df))]
    df["const"] = 1
    df["leak"] = np.where(df["y"] == "yes", 5.0, 1.0)
    df["dup"] = df["x2"]
    issues = analyze_quality(df, profile_dataset(df), "y", BINARY)
    codes = {(i.code, i.column) for i in issues}
    assert ("identifier", "cust_id") in codes and ("constant", "const") in codes
    assert ("leakage", "leak") in codes and ("duplicate_column", "dup") in codes
    rec = recommended_exclusions(issues)
    assert "cust_id" in rec and "98" not in rec["const"] and "leak" not in rec   # leakage is warned, not auto-excluded


def test_imbalance_warning():
    df = pd.DataFrame({"x": np.arange(200.0), "y": [1] * 10 + [0] * 190})
    assert any(i.code == "imbalance" for i in analyze_quality(df, profile_dataset(df), "y", BINARY))


def test_problem_detection():
    assert detect_problem_type(pd.Series(["a", "b"] * 20)).task == BINARY
    assert detect_problem_type(pd.Series(np.random.RandomState(0).normal(size=200))).task == REGRESSION
    assert detect_problem_type(pd.Series([0, 1, 2] * 100)).task == MULTICLASS
    assert detect_problem_type(pd.Series(np.arange(500))).task == REGRESSION
    assert detect_problem_type(pd.Series(["a", "b", "c"] * 10)).task == MULTICLASS
