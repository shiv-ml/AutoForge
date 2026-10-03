# Autoforge

Leakage-safe tabular AutoML (classification and regression) with a Streamlit UI.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q          # 35 tests
```
Optional: `pip install xgboost lightgbm catboost` adds those models automatically.

## Structure
```
app.py                          entry point (navigation only)
autoforge/
  config/settings.py            app name, thresholds, Quick/Standard/Thorough budgets
  core/                         no Streamlit imports - pure ML/business logic
    data/ingestion.py           CSV loading: encoding, delimiter, empty/malformed/binary detection
    profiling/profile.py        dtype inference, column stats, DatasetProfile, dataset hash
    profiling/quality.py        quality findings, suggested exclusions (with explanations), leakage probe
    preprocessing/pipeline.py   ColumnTransformer pipelines + datetime/log/winsorize transformers
    models/registry.py          model catalogue + search spaces (optional libs fail gracefully)
    optimization/tuner.py       Optuna search on training data only
    evaluation/                 problem detection, metrics, CV, diagnostics, trainer, error analysis
    explainability/importance.py permutation + native importance, local what-if explanation
    export/artifact.py          ModelBundle (pipeline+schema+metadata), validation, predict, joblib I/O
    experiments/tracker.py      local JSON experiment store
    reporting/report.py         HTML report
  app/                          Streamlit pages: upload, profiling, eda, automl, explainability, predictions, experiments
tests/                          pytest suite
datasets/                       sample CSVs (breast_cancer, diabetes)
```

## ML methodology
1. Rows with a missing target are removed and **reported**; nothing else is dropped silently. Your DataFrame is never mutated.
2. A holdout test set is split off first (stratified for classification). It is only used for reporting.
3. Preprocessing (imputation, scaling, log/clipping, one-hot with rare-level grouping, date expansion) is inside an sklearn `Pipeline`, so it is refit on each CV training fold. Scaling is applied only to models that need it (linear, KNN, SVM).
4. Models are ranked by the **CV mean of the user-chosen primary metric** (direction-aware), never by the test set.
5. Optuna tunes on training data with different fold seeds than the reported CV; tuned CV scores are flagged as slightly optimistic and the holdout score is the unbiased estimate.
6. A dummy baseline is evaluated the same way; over/underfitting/instability are shown as diagnostic indicators.
7. The exported artifact is the complete pipeline (optionally refit on all data with the winner's hyperparameters).

## Known limitations
- Training runs synchronously inside the Streamlit session (progress bar, but the UI blocks until done); large data + Thorough budget can be slow. No SHAP (permutation + occlusion explanations only). PDF export = print the HTML report. Local JSON tracking instead of MLflow. No polynomial/interaction features (deliberately, to avoid feature explosion). `joblib` bundles need the `autoforge` package and compatible library versions, and must only be loaded from trusted sources.

## Ideas for next steps
Background job runner, SHAP, nested CV, MLflow, time-series split strategy, calibration and threshold tuning, target-encoding for high-cardinality categoricals.
