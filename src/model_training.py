"""
Model Training — Breast Cancer Diagnosis
==========================================
Train, tune, save, and load all classification models.

Supported Models:
    - Logistic Regression
    - Random Forest
    - SVM (with probability estimates)
    - K-Nearest Neighbors
    - XGBoost
    - MLP (Multi-Layer Perceptron)

Each model is tuned via GridSearchCV with StratifiedKFold cross-validation
and scored on the metric specified in MODEL_CONFIG['scoring_metric'].
"""

import os
import sys

# ── Project root on sys.path ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.utils import get_logger, timer, save_json
from config.config import (
    MODEL_CONFIG,
    EVAL_CONFIG,
    PLOT_CONFIG,
    MODELS_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_directories,
)

# Graceful XGBoost import (PITFALL-006)
# Catches both ImportError and XGBoostError (e.g., missing libomp on macOS)
try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False

logger = get_logger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────────
_RANDOM_STATE = MODEL_CONFIG["random_state"]
_CV_FOLDS = MODEL_CONFIG["cv_folds"]
_SCORING = MODEL_CONFIG["scoring_metric"]
_N_JOBS = MODEL_CONFIG["n_jobs"]
_PARAM_GRIDS = MODEL_CONFIG["models"]


# ─── Public API ─────────────────────────────────────────────────────────────────


def get_models() -> dict:
    """
    Build the catalogue of models and their corresponding parameter grids.

    Returns:
        dict[str, tuple]: Mapping of model name → (model_instance, param_grid).
            Example::

                {
                    "Logistic Regression": (LogisticRegression(...), {...}),
                    "Random Forest":       (RandomForestClassifier(...), {...}),
                    ...
                }

    Notes:
        * ``random_state`` is injected into every constructor that accepts it
          (PITFALL-007).
        * ``class_weight='balanced'`` is set for models that support it
          (PITFALL-002).
        * XGBoost is silently skipped when the package is not installed.
    """
    models = {}

    # 1. Logistic Regression
    models["Logistic Regression"] = (
        LogisticRegression(
            random_state=_RANDOM_STATE,
            class_weight="balanced",
        ),
        _PARAM_GRIDS["Logistic Regression"],
    )

    # 2. Random Forest
    models["Random Forest"] = (
        RandomForestClassifier(
            random_state=_RANDOM_STATE,
            class_weight="balanced",
        ),
        _PARAM_GRIDS["Random Forest"],
    )

    # 3. SVM (probability=True for ROC-AUC)
    models["SVM"] = (
        SVC(
            random_state=_RANDOM_STATE,
            probability=True,
            class_weight="balanced",
        ),
        _PARAM_GRIDS["SVM"],
    )

    # 4. KNN (no random_state, no class_weight)
    models["KNN"] = (
        KNeighborsClassifier(),
        _PARAM_GRIDS["KNN"],
    )

    # 5. XGBoost
    if _HAS_XGBOOST:
        models["XGBoost"] = (
            XGBClassifier(
                random_state=_RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
            _PARAM_GRIDS["XGBoost"],
        )
    else:
        logger.warning(
            "⚠️  XGBoost not installed — skipping XGBClassifier. "
            "Install with: pip install xgboost"
        )

    # 6. MLP
    models["MLP"] = (
        MLPClassifier(
            random_state=_RANDOM_STATE,
        ),
        _PARAM_GRIDS["MLP"],
    )

    logger.info(f"Model catalogue built: {list(models.keys())}")
    return models


def train_single_model(
    name: str,
    model,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple:
    """
    Tune a single model with GridSearchCV + StratifiedKFold.

    Args:
        name:       Human-readable model name.
        model:      Unfitted sklearn-compatible estimator.
        param_grid: Dictionary of hyperparameter search spaces.
        X_train:    Training feature matrix.
        y_train:    Training target vector.

    Returns:
        tuple: (best_model, best_params, cv_results)
            - best_model:  Refitted estimator with the best hyperparameters.
            - best_params: dict of winning hyperparameters.
            - cv_results:  Full ``cv_results_`` dict from GridSearchCV.
    """
    logger.info(f"{'─' * 50}")
    logger.info(f"Training: {name}")
    logger.info(f"  Param grid : {param_grid}")

    cv = StratifiedKFold(
        n_splits=_CV_FOLDS,
        shuffle=True,
        random_state=_RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=_SCORING,
        n_jobs=_N_JOBS,
        verbose=0,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_results = grid_search.cv_results_

    logger.info(f"  Best {_SCORING}: {grid_search.best_score_:.4f}")
    logger.info(f"  Best params: {best_params}")

    return best_model, best_params, cv_results


@timer
def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """
    Train and tune every model in the catalogue.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        dict[str, dict]: Mapping of model name →
            ``{"model": fitted_estimator, "best_params": dict, "cv_results": dict}``
    """
    ensure_directories()
    catalogue = get_models()
    trained_models: dict = {}

    logger.info(f"Training {len(catalogue)} models with {_CV_FOLDS}-fold "
                f"stratified CV (scoring={_SCORING})")

    for idx, (name, (model, param_grid)) in enumerate(catalogue.items(), 1):
        logger.info(f"\n[{idx}/{len(catalogue)}] ─── {name} ───")
        best_model, best_params, cv_results = train_single_model(
            name, model, param_grid, X_train, y_train
        )
        trained_models[name] = {
            "model": best_model,
            "best_params": best_params,
            "cv_results": cv_results,
        }

    logger.info(f"\n✅ All {len(trained_models)} models trained successfully.")

    # Quick summary
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║            Training Summary                     ║")
    logger.info("╠══════════════════════════════════════════════════╣")
    for name, info in trained_models.items():
        # Extract mean CV score of best candidate
        best_idx = info["cv_results"]["params"].index(info["best_params"])
        mean_score = info["cv_results"][f"mean_test_score"][best_idx]
        logger.info(f"║  {name:<25s}  CV {_SCORING}: {mean_score:.4f}  ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    return trained_models


def save_models(trained_models: dict) -> dict:
    """
    Persist every trained model to disk using joblib.

    Args:
        trained_models: Output of :func:`train_all_models`.

    Returns:
        dict[str, str]: Mapping of model name → saved file path.
    """
    ensure_directories()
    saved_paths: dict = {}

    for name, info in trained_models.items():
        safe_name = name.lower().replace(" ", "_")
        filepath = os.path.join(MODELS_DIR, f"{safe_name}_best.joblib")
        joblib.dump(info["model"], filepath)
        saved_paths[name] = filepath
        logger.info(f"💾 Saved {name} → {filepath}")

    # Save a manifest so we know which models exist
    manifest_path = os.path.join(MODELS_DIR, "model_manifest.json")
    save_json(
        {name: {"path": path, "params": trained_models[name]["best_params"]}
         for name, path in saved_paths.items()},
        manifest_path,
    )
    logger.info(f"📋 Model manifest saved → {manifest_path}")

    return saved_paths


def load_model(model_name: str):
    """
    Load a previously saved model from ``MODELS_DIR``.

    Args:
        model_name: Human-readable model name
                    (e.g. ``"Random Forest"``).

    Returns:
        Fitted sklearn-compatible estimator.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    safe_name = model_name.lower().replace(" ", "_")
    filepath = os.path.join(MODELS_DIR, f"{safe_name}_best.joblib")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Model file not found: {filepath}. "
            f"Train and save models first."
        )

    model = joblib.load(filepath)
    logger.info(f"📂 Loaded {model_name} from {filepath}")
    return model


# ─── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from config.config import DATA_CONFIG

    ensure_directories()
    logger.info("=" * 60)
    logger.info("  Model Training — standalone run")
    logger.info("=" * 60)

    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=DATA_CONFIG["test_size"],
        random_state=DATA_CONFIG["random_state"],
        stratify=y,
    )
    logger.info(f"Data: X_train={X_train.shape}, X_test={X_test.shape}")

    # Train
    trained = train_all_models(X_train, y_train)

    # Save
    paths = save_models(trained)
    logger.info(f"Saved models: {list(paths.keys())}")

    # Reload quick check
    for name in trained:
        reloaded = load_model(name)
        score = reloaded.score(X_test, y_test)
        logger.info(f"  {name} reload check — test accuracy: {score:.4f}")

    logger.info("Done.")
