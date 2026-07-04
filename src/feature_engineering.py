"""
Feature Engineering — Breast Cancer Diagnosis
===============================================
Scaling, selection, and dimensionality-reduction utilities.

**PITFALL-001 — Data Leakage Prevention**
Every transformer is fitted on the TRAINING split **only** and then used
to transform both training *and* test splits.  No test-set statistics
ever leak into the fitting step.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be before pyplot import

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.decomposition import PCA

# ─── Project path bootstrap ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, timer, save_plot
from config.config import (
    DATA_CONFIG,
    FEATURE_CONFIG,
    PLOT_CONFIG,
    FIGURES_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    ensure_directories,
)

logger = get_logger(__name__)

# ─── Apply global plot style ───────────────────────────────────────────────────
plt.style.use(PLOT_CONFIG["style"])
plt.rcParams.update({
    "figure.figsize": PLOT_CONFIG["figsize_default"],
    "figure.dpi": PLOT_CONFIG["dpi"],
    "font.size": PLOT_CONFIG["font_size"],
    "axes.titlesize": PLOT_CONFIG["title_size"],
    "savefig.dpi": PLOT_CONFIG["dpi"],
    "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Feature Scaling
# ═══════════════════════════════════════════════════════════════════════════════

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Scale features using the method specified in FEATURE_CONFIG.

    **PITFALL-001**: The scaler is fitted on *X_train* **only**.
    *X_test* is transformed using the same fitted scaler.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.

    Returns:
        Tuple of (scaled_X_train, scaled_X_test, fitted_scaler).
    """
    scaler_type = FEATURE_CONFIG["scaler_type"]
    logger.info(f"Scaling features — method: {scaler_type}")

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}. Use 'standard' or 'minmax'.")

    # Fit on train ONLY, transform both (PITFALL-001)
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    logger.info(f"Scaler fit on training set ({X_train.shape[0]} samples)")
    logger.info(f"Train scaled range: [{X_train_scaled.min().min():.3f}, {X_train_scaled.max().max():.3f}]")
    logger.info(f"Test  scaled range: [{X_test_scaled.min().min():.3f}, {X_test_scaled.max().max():.3f}]")

    return X_train_scaled, X_test_scaled, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Variance Threshold Selection
# ═══════════════════════════════════════════════════════════════════════════════

def select_features_variance(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, VarianceThreshold]:
    """Remove features with near-zero variance.

    The selector is fitted on *X_train* only.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        threshold: Variance threshold. Defaults to ``FEATURE_CONFIG['variance_threshold']``.

    Returns:
        Tuple of (filtered_X_train, filtered_X_test, fitted_selector).
    """
    if threshold is None:
        threshold = FEATURE_CONFIG["variance_threshold"]

    logger.info(f"Variance threshold selection — threshold: {threshold}")
    selector = VarianceThreshold(threshold=threshold)

    # Fit on train ONLY (PITFALL-001)
    selector.fit(X_train)
    kept_mask = selector.get_support()
    kept_features = X_train.columns[kept_mask].tolist()
    removed_features = X_train.columns[~kept_mask].tolist()

    X_train_filtered = pd.DataFrame(
        selector.transform(X_train),
        columns=kept_features,
        index=X_train.index,
    )
    X_test_filtered = pd.DataFrame(
        selector.transform(X_test),
        columns=kept_features,
        index=X_test.index,
    )

    logger.info(f"Features kept: {len(kept_features)} / {X_train.shape[1]}")
    if removed_features:
        logger.info(f"Features removed (low variance): {removed_features}")
    else:
        logger.info("No features removed — all exceed variance threshold")

    return X_train_filtered, X_test_filtered, selector


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Mutual Information Selection
# ═══════════════════════════════════════════════════════════════════════════════

def select_features_mutual_info(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = None,
) -> Tuple[List[str], np.ndarray]:
    """Rank features by mutual information with the target.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        top_k: Number of top features to return. Defaults to
            ``FEATURE_CONFIG['mutual_info_top_k']``.

    Returns:
        Tuple of (top_k_feature_names, corresponding_mi_scores).
    """
    if top_k is None:
        top_k = FEATURE_CONFIG["mutual_info_top_k"]

    logger.info(f"Computing mutual information scores — top_k: {top_k}")

    mi_scores = mutual_info_classif(
        X_train,
        y_train,
        random_state=DATA_CONFIG["random_state"],
    )

    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
    top_features = mi_series.head(top_k).index.tolist()
    top_scores = mi_series.head(top_k).values

    logger.info("Mutual information ranking:")
    for feat, score in zip(top_features, top_scores):
        logger.info(f"  {feat}: {score:.4f}")

    # Plot MI scores
    plot_feature_importance(
        feature_names=top_features,
        scores=top_scores,
        title="Mutual Information Scores — Top Features",
        filename="mutual_info_scores.png",
    )

    return top_features, top_scores


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Remove Correlated Features
# ═══════════════════════════════════════════════════════════════════════════════

def remove_correlated_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Drop one feature from each highly correlated pair.

    For every pair with |r| > *threshold*, the feature that appears later
    in the column order is dropped.  Correlation is computed on *X_train*
    only to prevent leakage.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        threshold: Correlation threshold. Defaults to
            ``FEATURE_CONFIG['correlation_threshold']``.

    Returns:
        Tuple of (filtered_X_train, filtered_X_test, list_of_dropped_columns).
    """
    if threshold is None:
        threshold = FEATURE_CONFIG["correlation_threshold"]

    logger.info(f"Removing correlated features — threshold: {threshold}")

    # Compute correlation on training data ONLY (PITFALL-001)
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    # Identify columns to drop
    to_drop = set()
    for col in upper.columns:
        high_corr_cols = upper.index[upper[col] > threshold].tolist()
        if high_corr_cols:
            to_drop.add(col)

    to_drop = sorted(to_drop)
    logger.info(f"Dropping {len(to_drop)} correlated features: {to_drop}")

    X_train_filtered = X_train.drop(columns=to_drop)
    X_test_filtered = X_test.drop(columns=to_drop)

    logger.info(f"Features remaining: {X_train_filtered.shape[1]} / {X_train.shape[1]}")

    return X_train_filtered, X_test_filtered, to_drop


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PCA
# ═══════════════════════════════════════════════════════════════════════════════

def apply_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """Fit PCA on training data and transform both splits.

    A cumulative variance-explained curve is saved to FIGURES_DIR.

    Args:
        X_train: Training feature matrix (ideally already scaled).
        X_test: Test feature matrix.
        n_components: Number of principal components. Defaults to
            ``FEATURE_CONFIG['pca_n_components']``.

    Returns:
        Tuple of (X_train_pca, X_test_pca, fitted_PCA).
    """
    if n_components is None:
        n_components = FEATURE_CONFIG["pca_n_components"]

    n_components = min(n_components, X_train.shape[1], X_train.shape[0])
    logger.info(f"Applying PCA — n_components: {n_components}")

    pca = PCA(n_components=n_components, random_state=DATA_CONFIG["random_state"])

    # Fit on train ONLY (PITFALL-001)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    pc_cols = [f"PC{i+1}" for i in range(n_components)]

    X_train_pca = pd.DataFrame(X_train_pca, columns=pc_cols, index=X_train.index)
    X_test_pca = pd.DataFrame(X_test_pca, columns=pc_cols, index=X_test.index)

    # Log variance explained
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    logger.info(f"Variance explained by {n_components} components: {cumulative_var[-1]:.4f}")
    for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, cumulative_var)):
        logger.info(f"  PC{i+1}: {var:.4f} (cumulative: {cum:.4f})")

    # ── Plot cumulative variance explained ──
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize_default"])
    components = np.arange(1, n_components + 1)

    ax.bar(
        components,
        pca.explained_variance_ratio_,
        alpha=0.6,
        color=sns.color_palette(PLOT_CONFIG["color_palette"])[0],
        label="Individual",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.step(
        components,
        cumulative_var,
        where="mid",
        color=sns.color_palette(PLOT_CONFIG["color_palette"])[1],
        linewidth=2,
        label="Cumulative",
    )
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label="95% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title("PCA — Variance Explained", fontsize=PLOT_CONFIG["title_size"])
    ax.set_xticks(components)
    ax.legend(loc="center right")

    save_plot(fig, "pca_variance_explained.png", FIGURES_DIR)
    plt.close("all")

    return X_train_pca, X_test_pca, pca


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Feature Importance Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    feature_names: list,
    scores: np.ndarray,
    title: str,
    filename: str,
) -> str:
    """Horizontal bar chart of feature importance / scores.

    Args:
        feature_names: List of feature names.
        scores: Corresponding importance scores.
        title: Plot title.
        filename: Output filename (saved to FIGURES_DIR).

    Returns:
        Filepath of the saved plot.
    """
    logger.info(f"Plotting feature importance → {filename}")

    # Sort ascending so highest bar is on top
    sorted_idx = np.argsort(scores)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_scores = scores[sorted_idx]

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize_default"])
    bars = ax.barh(
        range(len(sorted_names)),
        sorted_scores,
        color=sns.color_palette(PLOT_CONFIG["color_palette"], n_colors=len(sorted_names)),
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Score")
    ax.set_title(title, fontsize=PLOT_CONFIG["title_size"])

    # Annotate bars
    for bar, score in zip(bars, sorted_scores):
        ax.text(
            bar.get_width() + max(sorted_scores) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            fontsize=PLOT_CONFIG["font_size"] - 2,
        )

    fig.tight_layout()
    filepath = save_plot(fig, filename, FIGURES_DIR)
    plt.close("all")
    return filepath


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Master Feature Engineering Runner
# ═══════════════════════════════════════════════════════════════════════════════

@timer
def run_feature_engineering(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Any]:
    """Execute the full feature engineering pipeline.

    Steps:
        1. Scale features (fit on train only).
        2. Remove near-zero variance features.
        3. Remove highly correlated features.
        4. Rank features by mutual information.
        5. Apply PCA for dimensionality reduction analysis.

    All fitted transformers are returned so they can be reused on
    unseen data.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target labels.

    Returns:
        Dictionary containing processed data and fitted transformers::

            {
                "X_train_scaled", "X_test_scaled", "scaler",
                "X_train_selected", "X_test_selected", "variance_selector",
                "X_train_decorr", "X_test_decorr", "dropped_corr_cols",
                "mi_top_features", "mi_scores",
                "X_train_pca", "X_test_pca", "pca",
            }
    """
    ensure_directories()
    logger.info(
        f"Starting feature engineering — "
        f"train: {X_train.shape}, test: {X_test.shape}"
    )
    results: Dict[str, Any] = {}

    # 1 — Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    results["X_train_scaled"] = X_train_scaled
    results["X_test_scaled"] = X_test_scaled
    results["scaler"] = scaler

    # Save scaler for API/inference use
    import joblib
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info(f"💾 Saved scaler → {scaler_path}")

    # 2 — Variance threshold
    X_train_var, X_test_var, var_selector = select_features_variance(
        X_train_scaled, X_test_scaled
    )
    results["X_train_selected"] = X_train_var
    results["X_test_selected"] = X_test_var
    results["variance_selector"] = var_selector

    # 3 — Remove correlated features
    X_train_decorr, X_test_decorr, dropped_cols = remove_correlated_features(
        X_train_var, X_test_var
    )
    results["X_train_decorr"] = X_train_decorr
    results["X_test_decorr"] = X_test_decorr
    results["dropped_corr_cols"] = dropped_cols

    # 4 — Mutual information ranking (uses scaled, variance-filtered data)
    mi_features, mi_scores = select_features_mutual_info(X_train_var, y_train)
    results["mi_top_features"] = mi_features
    results["mi_scores"] = mi_scores

    # 5 — PCA (on scaled data for maximum variance capture)
    X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled)
    results["X_train_pca"] = X_train_pca
    results["X_test_pca"] = X_test_pca
    results["pca"] = pca

    logger.info("Feature engineering complete")
    logger.info(f"  Scaled features      : {X_train_scaled.shape[1]}")
    logger.info(f"  After variance filter : {X_train_var.shape[1]}")
    logger.info(f"  After decorrelation   : {X_train_decorr.shape[1]}")
    logger.info(f"  PCA components        : {X_train_pca.shape[1]}")
    logger.info(f"  MI top features       : {len(mi_features)}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    ensure_directories()
    logger.info("Loading Wisconsin Breast Cancer dataset …")

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[DATA_CONFIG["target_column"]] = data.target

    target_col = DATA_CONFIG["target_column"]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=DATA_CONFIG["test_size"],
        random_state=DATA_CONFIG["random_state"],
        stratify=y,
    )

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    results = run_feature_engineering(X_train, X_test, y_train)

    logger.info("═" * 60)
    logger.info("Feature engineering results:")
    for key, val in results.items():
        if isinstance(val, pd.DataFrame):
            logger.info(f"  {key}: DataFrame {val.shape}")
        elif isinstance(val, np.ndarray):
            logger.info(f"  {key}: ndarray {val.shape}")
        elif isinstance(val, list):
            logger.info(f"  {key}: {val}")
        else:
            logger.info(f"  {key}: {type(val).__name__}")
    logger.info("═" * 60)
