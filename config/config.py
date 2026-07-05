"""
Centralized Configuration — Breast Cancer Diagnosis
=====================================================
ALL paths, hyperparameters, and constants live here.
No magic numbers anywhere else in the codebase.
"""

import os
import yaml

# ─── Project Root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Directory Paths ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
RESULTS_DIR = os.path.join(REPORTS_DIR, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ─── Data Configuration ────────────────────────────────────────────────────────
DATA_CONFIG = {
    "test_size": 0.2,              # 80/20 train/test split
    "random_state": 42,            # Reproducibility seed
    "target_column": "target",     # Target column name
    "class_names": ["Malignant", "Benign"],  # 0 = Malignant, 1 = Benign
    "expected_n_samples": 569,
    "expected_n_features": 30,
}

# ─── Feature Engineering Configuration ──────────────────────────────────────────
FEATURE_CONFIG = {
    "scaler_type": "standard",     # "standard" or "minmax"
    "variance_threshold": 0.01,    # Remove near-zero variance features
    "correlation_threshold": 0.95, # Drop one of highly correlated feature pairs
    "pca_n_components": 10,        # Number of PCA components for analysis
    "mutual_info_top_k": 15,       # Top K features by mutual information
}

# ─── Model Configuration ───────────────────────────────────────────────────────
MODEL_CONFIG = {
    "cv_folds": 10,                # Stratified K-Fold cross validation
    "random_state": 42,            # Reproducibility seed
    "scoring_metric": "f1",        # Primary metric for model selection
    "n_jobs": -1,                  # Use all CPU cores

    "models": {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [1000],
            "class_weight": ["balanced"],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "class_weight": ["balanced"],
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
            "class_weight": ["balanced"],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
        },
        "MLP": {
            "hidden_layer_sizes": [(64, 32), (128, 64), (100,)],
            "activation": ["relu"],
            "alpha": [0.0001, 0.001, 0.01],
            "max_iter": [500],
            "early_stopping": [True],
        },
    },
}

# ─── Evaluation Configuration ──────────────────────────────────────────────────
EVAL_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "confusion_matrix_cmap": "Blues",
    "roc_curve_figsize": (10, 8),
    "comparison_figsize": (14, 8),
    "top_features_count": 10,
}

# ─── Logging Configuration ─────────────────────────────────────────────────────
LOG_CONFIG = {
    "log_file": os.path.join(LOGS_DIR, "pipeline.log"),
    "log_level": "INFO",
    "log_format": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# ─── Plot Style Configuration ──────────────────────────────────────────────────
PLOT_CONFIG = {
    "style": "seaborn-v0_8-whitegrid",
    "figsize_default": (10, 6),
    "figsize_large": (14, 10),
    "dpi": 150,
    "color_palette": "husl",
    "font_size": 12,
    "title_size": 14,
}
# ─── Deep Learning Configuration (Phase 2) ──────────────────────────────────────
DL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "deep_learning.yaml")
if os.path.exists(DL_CONFIG_PATH):
    try:
        with open(DL_CONFIG_PATH, "r") as f:
            DL_CONFIG = yaml.safe_load(f)
    except Exception:
        DL_CONFIG = {}
else:
    DL_CONFIG = {}

# Set defaults if keys are missing
DL_CONFIG.setdefault("backbone", "efficientnet_b0")
DL_CONFIG.setdefault("epochs", 10)
DL_CONFIG.setdefault("batch_size", 32)
DL_CONFIG.setdefault("learning_rate", 0.001)
DL_CONFIG.setdefault("device", "auto")
DL_CONFIG.setdefault("patience", 3)  # Early stopping patience
DL_CONFIG.setdefault("test_size", 0.15)
DL_CONFIG.setdefault("val_size", 0.15)



def ensure_directories():
    """Create all required directories if they don't exist."""
    dirs = [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
        FIGURES_DIR, RESULTS_DIR, LOGS_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
