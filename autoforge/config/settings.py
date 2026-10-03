"""Central configuration. Change ``APP_NAME`` here to rename the whole application."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

APP_NAME = "Autoforge"
APP_TAGLINE = "Leakage-safe tabular AutoML, from CSV to deployable pipeline"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"


@dataclass(frozen=True)
class Thresholds:
    """Heuristic thresholds used by validation and data-quality checks."""

    id_unique_ratio: float = 0.95
    high_cardinality: int = 50
    near_constant_share: float = 0.98
    missing_warn_pct: float = 0.0
    missing_exclude_pct: float = 50.0
    extreme_outlier_iqr: float = 3.0
    extreme_outlier_share: float = 0.01
    imbalance_minority_share: float = 0.10
    leakage_score: float = 0.98
    rare_category_share: float = 0.01
    min_rows: int = 20
    slow_model_row_limit: int = 20_000
    text_avg_len: int = 40


@dataclass(frozen=True)
class Budget:
    """Computational budget for a training run."""

    n_trials: int          # Optuna trials per model (0 = no tuning)
    timeout_s: int         # wall-clock limit per model tuning study
    description: str


BUDGETS: dict[str, Budget] = {
    "Quick": Budget(0, 0, "Default hyperparameters, cross-validated. Fastest."),
    "Standard": Budget(15, 45, "15 Optuna trials per model (max 45 s each)."),
    "Thorough": Budget(40, 120, "40 Optuna trials per model (max 120 s each)."),
}

THRESHOLDS = Thresholds()
DEFAULT_SEED = 42
MAX_UPLOAD_MB = 200
