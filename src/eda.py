"""
Exploratory Data Analysis — Breast Cancer Diagnosis
=====================================================
Generates visualisations and statistical summaries for the
Wisconsin Breast Cancer dataset.

Every plot is saved to FIGURES_DIR automatically; nothing is
displayed interactively (Agg backend).
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be before pyplot import

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ─── Project path bootstrap ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, timer, save_plot
from config.config import (
    DATA_CONFIG,
    FEATURE_CONFIG,
    PLOT_CONFIG,
    FIGURES_DIR,
    RESULTS_DIR,
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
# 1. Class Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(df: pd.DataFrame, target_col: str) -> str:
    """Plot a bar chart of class counts (Malignant vs Benign).

    Args:
        df: Full dataset including the target column.
        target_col: Name of the target column.

    Returns:
        Filepath of the saved plot.
    """
    logger.info("Plotting class distribution …")
    class_names = DATA_CONFIG["class_names"]
    counts = df[target_col].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize_default"])
    bars = ax.bar(
        [class_names[i] for i in counts.index],
        counts.values,
        color=sns.color_palette(PLOT_CONFIG["color_palette"], n_colors=2),
        edgecolor="black",
        linewidth=0.8,
    )

    # Annotate bars with counts and percentages
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=PLOT_CONFIG["font_size"],
        )

    ax.set_title("Class Distribution — Breast Cancer Diagnosis", fontsize=PLOT_CONFIG["title_size"])
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)

    filepath = save_plot(fig, "class_distribution.png", FIGURES_DIR)
    plt.close("all")
    logger.info(f"Class distribution: {dict(zip(class_names, counts.values))}")
    return filepath


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Plot the full correlation matrix as a heatmap.

    Highly correlated pairs (|r| > 0.95) are logged for inspection.

    Args:
        df: DataFrame with numeric features only (no target).

    Returns:
        Filepath of the saved plot.
    """
    logger.info("Plotting correlation heatmap …")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    # Mask upper triangle for cleaner visual
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize_large"])
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        annot=False,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=PLOT_CONFIG["title_size"])

    filepath = save_plot(fig, "correlation_heatmap.png", FIGURES_DIR)
    plt.close("all")

    # Log highly correlated pairs
    high_corr = identify_correlated_features(numeric_df, threshold=0.95)
    if high_corr:
        logger.info(f"Found {len(high_corr)} feature pairs with |r| > 0.95")
    return filepath


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Feature Distributions (Violin Plots)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_distributions(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 10,
) -> str:
    """Violin plots of the *top_n* most discriminative features by class.

    Discriminativeness is measured via absolute t-statistic between
    Malignant (0) and Benign (1) groups.

    Args:
        df: Full dataset including target.
        target_col: Target column name.
        top_n: Number of top features to plot.

    Returns:
        Filepath of the saved plot.
    """
    logger.info(f"Plotting violin distributions for top {top_n} features …")
    class_names = DATA_CONFIG["class_names"]
    feature_cols = [c for c in df.columns if c != target_col]

    # Rank features by absolute t-statistic
    t_scores = {}
    group0 = df[df[target_col] == 0]
    group1 = df[df[target_col] == 1]
    for col in feature_cols:
        t_stat, _ = stats.ttest_ind(group0[col], group1[col], equal_var=False)
        t_scores[col] = abs(t_stat)

    top_features = sorted(t_scores, key=t_scores.get, reverse=True)[:top_n]
    logger.info(f"Top {top_n} discriminative features: {top_features}")

    # Create subplot grid
    n_cols = 2
    n_rows = int(np.ceil(top_n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = axes.flatten()

    palette = {0: sns.color_palette(PLOT_CONFIG["color_palette"])[0],
               1: sns.color_palette(PLOT_CONFIG["color_palette"])[1]}

    for idx, feat in enumerate(top_features):
        ax = axes[idx]
        sns.violinplot(
            data=df,
            x=target_col,
            y=feat,
            hue=target_col,
            palette=palette,
            inner="quartile",
            ax=ax,
            legend=False,
        )
        ax.set_xticklabels(class_names)
        ax.set_title(feat, fontsize=PLOT_CONFIG["font_size"])
        ax.set_xlabel("")

    # Hide unused subplots
    for idx in range(top_n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Top {top_n} Discriminative Features — Violin Plots",
        fontsize=PLOT_CONFIG["title_size"],
        y=1.01,
    )
    fig.tight_layout()

    filepath = save_plot(fig, "feature_distributions_violin.png", FIGURES_DIR)
    plt.close("all")
    return filepath


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Pairplot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pairplot(
    df: pd.DataFrame,
    target_col: str,
    features: list = None,
) -> str:
    """Seaborn pairplot of the top 5 features, coloured by class.

    Args:
        df: Full dataset including target.
        target_col: Target column name.
        features: Optional explicit list of feature names. If *None*,
            the 5 most discriminative features (by t-stat) are used.

    Returns:
        Filepath of the saved plot.
    """
    logger.info("Plotting pairplot …")
    class_names = DATA_CONFIG["class_names"]

    if features is None:
        # Pick top-5 by absolute t-statistic
        feature_cols = [c for c in df.columns if c != target_col]
        t_scores = {}
        group0 = df[df[target_col] == 0]
        group1 = df[df[target_col] == 1]
        for col in feature_cols:
            t_stat, _ = stats.ttest_ind(group0[col], group1[col], equal_var=False)
            t_scores[col] = abs(t_stat)
        features = sorted(t_scores, key=t_scores.get, reverse=True)[:5]

    logger.info(f"Pairplot features: {features}")

    # Create a labelled copy for nicer legend
    plot_df = df[features + [target_col]].copy()
    plot_df["Diagnosis"] = plot_df[target_col].map(
        {i: name for i, name in enumerate(class_names)}
    )

    g = sns.pairplot(
        plot_df,
        hue="Diagnosis",
        vars=features,
        palette=sns.color_palette(PLOT_CONFIG["color_palette"], n_colors=2),
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 30, "edgecolor": "white", "linewidth": 0.3},
    )
    g.figure.suptitle("Pairplot — Top 5 Discriminative Features", y=1.02,
                       fontsize=PLOT_CONFIG["title_size"])

    filepath = save_plot(g.figure, "pairplot_top5.png", FIGURES_DIR)
    plt.close("all")
    return filepath


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Statistical Summary
# ═══════════════════════════════════════════════════════════════════════════════

def generate_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for every numeric feature.

    Metrics: mean, std, min, 25%, 50%, 75%, max, skewness, kurtosis.
    The summary is saved as a CSV to RESULTS_DIR.

    Args:
        df: DataFrame with numeric features.

    Returns:
        Summary DataFrame.
    """
    logger.info("Generating statistical summary …")
    numeric_df = df.select_dtypes(include=[np.number])

    summary = numeric_df.describe().T
    summary["skew"] = numeric_df.skew()
    summary["kurtosis"] = numeric_df.kurtosis()
    summary = summary.round(4)

    # Save to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, "statistical_summary.csv")
    summary.to_csv(filepath)
    logger.info(f"Statistical summary saved → {filepath}")
    logger.info(f"Summary shape: {summary.shape}")

    # Log features with high skew / kurtosis
    high_skew = summary[summary["skew"].abs() > 1].index.tolist()
    if high_skew:
        logger.info(f"Features with |skew| > 1: {high_skew}")

    high_kurt = summary[summary["kurtosis"].abs() > 3].index.tolist()
    if high_kurt:
        logger.info(f"Features with |kurtosis| > 3: {high_kurt}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Identify Highly Correlated Features
# ═══════════════════════════════════════════════════════════════════════════════

def identify_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> list:
    """Return pairs of features with |Pearson r| above *threshold*.

    Args:
        df: DataFrame with numeric features (no target).
        threshold: Correlation threshold (default 0.95).

    Returns:
        List of tuples ``(feature_a, feature_b, correlation)``.
    """
    logger.info(f"Scanning for feature pairs with |r| > {threshold} …")
    corr = df.select_dtypes(include=[np.number]).corr().abs()

    # Extract upper triangle (no diagonal)
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and val > threshold:
                pairs.append((idx, col, round(val, 4)))

    if pairs:
        logger.info(f"Correlated pairs (|r| > {threshold}):")
        for a, b, r in pairs:
            logger.info(f"  • {a}  ↔  {b}  (r={r})")
    else:
        logger.info(f"No feature pairs exceed |r| > {threshold}")

    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Master EDA Runner
# ═══════════════════════════════════════════════════════════════════════════════

@timer
def run_eda(df: pd.DataFrame, target_col: str = "target") -> dict:
    """Execute the full Exploratory Data Analysis pipeline.

    Runs every EDA step in sequence and collects output paths / artefacts
    into a summary dictionary.

    Args:
        df: Complete dataset (features + target).
        target_col: Name of the target column (default ``"target"``).

    Returns:
        Dictionary with keys pointing to saved artefacts.
    """
    ensure_directories()
    logger.info(f"Starting full EDA — {df.shape[0]} samples, {df.shape[1]} columns")
    artefacts = {}

    # 1 — Class distribution
    artefacts["class_distribution"] = plot_class_distribution(df, target_col)

    # 2 — Correlation heatmap (features only)
    feature_df = df.drop(columns=[target_col], errors="ignore")
    artefacts["correlation_heatmap"] = plot_correlation_heatmap(feature_df)

    # 3 — Feature violin plots
    artefacts["feature_distributions"] = plot_feature_distributions(
        df, target_col, top_n=10
    )

    # 4 — Pairplot
    artefacts["pairplot"] = plot_pairplot(df, target_col)

    # 5 — Statistical summary
    summary_df = generate_statistical_summary(feature_df)
    artefacts["statistical_summary"] = os.path.join(
        RESULTS_DIR, "statistical_summary.csv"
    )

    # 6 — Correlated features
    correlated_pairs = identify_correlated_features(
        feature_df, threshold=FEATURE_CONFIG["correlation_threshold"]
    )
    artefacts["correlated_pairs"] = correlated_pairs

    logger.info(f"EDA complete — {len(artefacts)} artefacts generated")
    return artefacts


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    ensure_directories()
    logger.info("Loading Wisconsin Breast Cancer dataset …")

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[DATA_CONFIG["target_column"]] = data.target

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Class counts:\n{df[DATA_CONFIG['target_column']].value_counts()}")

    artefacts = run_eda(df, target_col=DATA_CONFIG["target_column"])

    logger.info("═" * 60)
    logger.info("EDA artefacts:")
    for key, val in artefacts.items():
        logger.info(f"  {key}: {val}")
    logger.info("═" * 60)
