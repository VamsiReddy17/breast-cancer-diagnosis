"""
Model Evaluation — Breast Cancer Diagnosis
=============================================
Evaluate trained classifiers and produce publication-quality
visualisations and summary reports.

Capabilities:
    - Per-model metrics  (accuracy, precision, recall, F1, ROC-AUC)
    - Confusion matrices (seaborn heatmaps)
    - ROC curves         (all models on one plot)
    - Model comparison   (grouped bar chart)
    - Feature importance  (tree-based models)
    - Comparison table   (sorted DataFrame → CSV)
    - Text report        (best model, metrics, recommendations)
"""

import os
import sys

# ── Project root on sys.path ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force non-interactive backend BEFORE any other matplotlib import
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from src.utils import get_logger, timer, save_plot, save_json
from config.config import (
    MODEL_CONFIG,
    EVAL_CONFIG,
    PLOT_CONFIG,
    DATA_CONFIG,
    MODELS_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_directories,
)

logger = get_logger(__name__)

# ─── Metric helpers ─────────────────────────────────────────────────────────────

_METRIC_FNS = {
    "accuracy": accuracy_score,
    "precision": lambda y, yp: precision_score(y, yp, zero_division=0),
    "recall": lambda y, yp: recall_score(y, yp, zero_division=0),
    "f1": lambda y, yp: f1_score(y, yp, zero_division=0),
}


def _safe_roc_auc(model, X_test, y_test):
    """
    Compute ROC-AUC, gracefully handling models without predict_proba.

    Falls back to decision_function when predict_proba is not available.
    Returns ``None`` when neither method exists.
    """
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            logger.warning(
                f"Model {type(model).__name__} has no predict_proba or "
                f"decision_function — ROC-AUC will be None."
            )
            return None
        return roc_auc_score(y_test, y_prob)
    except Exception as e:
        logger.warning(f"ROC-AUC computation failed: {e}")
        return None


# ─── Style helper ───────────────────────────────────────────────────────────────


def _apply_plot_style():
    """Apply the global plot style from PLOT_CONFIG."""
    plt.style.use(PLOT_CONFIG["style"])
    plt.rcParams.update({
        "font.size": PLOT_CONFIG["font_size"],
        "axes.titlesize": PLOT_CONFIG["title_size"],
        "figure.dpi": PLOT_CONFIG["dpi"],
    })


# ─── 1. Per-model metrics ──────────────────────────────────────────────────────


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Compute all metrics listed in ``EVAL_CONFIG['metrics']`` for one model.

    Args:
        model:       Fitted sklearn-compatible estimator.
        X_test:      Test feature matrix.
        y_test:      Test target vector.
        model_name:  Human-readable name (used for logging).

    Returns:
        dict[str, float]: Metric name → score.
    """
    y_pred = model.predict(X_test)
    metrics: dict = {}

    for metric_name in EVAL_CONFIG["metrics"]:
        if metric_name == "roc_auc":
            score = _safe_roc_auc(model, X_test, y_test)
        elif metric_name in _METRIC_FNS:
            score = _METRIC_FNS[metric_name](y_test, y_pred)
        else:
            logger.warning(f"Unknown metric '{metric_name}' — skipping.")
            continue
        metrics[metric_name] = score

    logger.info(f"📊 {model_name} metrics: {_fmt_metrics(metrics)}")
    return metrics


def _fmt_metrics(m: dict) -> str:
    """Pretty-format a metrics dict for logging."""
    parts = []
    for k, v in m.items():
        if v is None:
            parts.append(f"{k}=N/A")
        else:
            parts.append(f"{k}={v:.4f}")
    return "  ".join(parts)


# ─── 2. Evaluate all ───────────────────────────────────────────────────────────


@timer
def evaluate_all_models(trained_models: dict, X_test, y_test) -> dict:
    """
    Evaluate every model in *trained_models*.

    Args:
        trained_models: Output of ``train_all_models``
                        (name → {"model": ..., ...}).
        X_test:  Test feature matrix.
        y_test:  Test target vector.

    Returns:
        dict[str, dict]: Model name → metrics dict.
    """
    ensure_directories()
    all_results: dict = {}

    for name, info in trained_models.items():
        model = info["model"]
        metrics = evaluate_model(model, X_test, y_test, name)
        all_results[name] = metrics

    return all_results


# ─── 3. Confusion matrix ───────────────────────────────────────────────────────


def plot_confusion_matrix(model, X_test, y_test, model_name: str) -> str:
    """
    Plot and save a confusion-matrix heatmap for a single model.

    Args:
        model:      Fitted estimator.
        X_test:     Test feature matrix.
        y_test:     Test target vector.
        model_name: Human-readable name (used in title and filename).

    Returns:
        str: Absolute path to saved image.
    """
    _apply_plot_style()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    class_names = DATA_CONFIG["class_names"]

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize_default"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=EVAL_CONFIG["confusion_matrix_cmap"],
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted Label", fontsize=PLOT_CONFIG["font_size"])
    ax.set_ylabel("True Label", fontsize=PLOT_CONFIG["font_size"])
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=PLOT_CONFIG["title_size"])

    safe_name = model_name.lower().replace(" ", "_")
    filepath = save_plot(fig, f"confusion_matrix_{safe_name}.png", FIGURES_DIR)
    plt.close("all")
    logger.info(f"🖼  Confusion matrix saved → {filepath}")
    return filepath


# ─── 4. ROC curves ─────────────────────────────────────────────────────────────


def plot_roc_curves(trained_models: dict, X_test, y_test) -> str:
    """
    Plot ROC curves for **all** models on a single figure.

    Models that lack ``predict_proba`` / ``decision_function`` are
    skipped with a warning.

    Args:
        trained_models: name → {"model": ..., ...} dict.
        X_test:  Test feature matrix.
        y_test:  Test target vector.

    Returns:
        str: Absolute path to saved image.
    """
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=EVAL_CONFIG["roc_curve_figsize"])
    colors = sns.color_palette(PLOT_CONFIG["color_palette"], n_colors=len(trained_models))

    for (name, info), color in zip(trained_models.items(), colors):
        model = info["model"]

        # Obtain probability / decision scores
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            logger.warning(f"⚠️  {name}: no predict_proba / decision_function — "
                           f"skipping ROC curve.")
            continue

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc_val:.4f})")

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=PLOT_CONFIG["font_size"])
    ax.set_ylabel("True Positive Rate", fontsize=PLOT_CONFIG["font_size"])
    ax.set_title("ROC Curves — All Models", fontsize=PLOT_CONFIG["title_size"])
    ax.legend(loc="lower right", fontsize=PLOT_CONFIG["font_size"] - 2)

    filepath = save_plot(fig, "roc_curves_all_models.png", FIGURES_DIR)
    plt.close("all")
    logger.info(f"🖼  ROC curves saved → {filepath}")
    return filepath


# ─── 5. Model comparison bar chart ─────────────────────────────────────────────


def plot_model_comparison(results_df: pd.DataFrame) -> str:
    """
    Grouped bar chart comparing every model across all metrics.

    Args:
        results_df: DataFrame with model names as index and metric
                    names as columns.

    Returns:
        str: Absolute path to saved image.
    """
    _apply_plot_style()

    # Drop columns with any None/NaN for cleaner plotting
    plot_df = results_df.dropna(axis=1, how="any")

    fig, ax = plt.subplots(figsize=EVAL_CONFIG["comparison_figsize"])
    n_models = len(plot_df)
    n_metrics = len(plot_df.columns)
    x = np.arange(n_models)
    bar_width = 0.8 / max(n_metrics, 1)

    colors = sns.color_palette(PLOT_CONFIG["color_palette"], n_colors=n_metrics)

    for i, (metric, color) in enumerate(zip(plot_df.columns, colors)):
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            plot_df[metric].values,
            width=bar_width,
            label=metric.capitalize(),
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=PLOT_CONFIG["font_size"] - 4,
                rotation=45,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=20, ha="right",
                       fontsize=PLOT_CONFIG["font_size"])
    ax.set_ylabel("Score", fontsize=PLOT_CONFIG["font_size"])
    ax.set_title("Model Comparison — All Metrics",
                 fontsize=PLOT_CONFIG["title_size"])
    ax.legend(loc="lower right", fontsize=PLOT_CONFIG["font_size"] - 2)
    ax.set_ylim(0, 1.12)
    fig.tight_layout()

    filepath = save_plot(fig, "model_comparison.png", FIGURES_DIR)
    plt.close("all")
    logger.info(f"🖼  Model comparison chart saved → {filepath}")
    return filepath


# ─── 6. Feature importance ─────────────────────────────────────────────────────


def plot_feature_importance(model, feature_names, model_name: str):
    """
    Horizontal bar chart of the top-N most important features.

    Only works for tree-based models that expose ``feature_importances_``
    (Random Forest, XGBoost, etc.).

    Args:
        model:         Fitted estimator.
        feature_names: List/array of feature names.
        model_name:    Human-readable model name.

    Returns:
        str | None: Saved file path, or ``None`` when the model has no
                    ``feature_importances_`` attribute.
    """
    if not hasattr(model, "feature_importances_"):
        logger.info(f"ℹ️  {model_name} does not expose feature_importances_ — skipping.")
        return None

    _apply_plot_style()
    importances = model.feature_importances_
    top_k = EVAL_CONFIG["top_features_count"]

    indices = np.argsort(importances)[::-1][:top_k]
    top_names = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize_default"])
    colors = sns.color_palette(PLOT_CONFIG["color_palette"], n_colors=top_k)
    ax.barh(range(top_k), top_values[::-1], color=colors)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=PLOT_CONFIG["font_size"] - 1)
    ax.set_xlabel("Importance", fontsize=PLOT_CONFIG["font_size"])
    ax.set_title(f"Top {top_k} Features — {model_name}",
                 fontsize=PLOT_CONFIG["title_size"])
    fig.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    filepath = save_plot(fig, f"feature_importance_{safe_name}.png", FIGURES_DIR)
    plt.close("all")
    logger.info(f"🖼  Feature importance saved → {filepath}")
    return filepath


# ─── 7. Comparison table ───────────────────────────────────────────────────────


def generate_comparison_table(all_results: dict) -> pd.DataFrame:
    """
    Create, display, and persist a DataFrame comparing all models.

    The table is sorted by F1 score in descending order and saved as
    ``model_comparison.csv`` inside ``RESULTS_DIR``.

    Args:
        all_results: Model name → metrics dict.

    Returns:
        pd.DataFrame: Comparison table (model names as index).
    """
    ensure_directories()

    df = pd.DataFrame(all_results).T
    df.index.name = "Model"

    # Sort by F1 (descending), fall back to accuracy if F1 is absent
    sort_col = "f1" if "f1" in df.columns else df.columns[0]
    df = df.sort_values(by=sort_col, ascending=False)

    # Round for readability
    df_display = df.round(4)

    # Print to console
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON TABLE")
    logger.info("=" * 70)
    logger.info("\n" + df_display.to_string())
    logger.info("=" * 70)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df.to_csv(csv_path)
    logger.info(f"📄 Comparison CSV saved → {csv_path}")

    # Save JSON as well
    json_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    save_json(df.to_dict(orient="index"), json_path)

    return df


# ─── 8. Text report ────────────────────────────────────────────────────────────


def generate_report(all_results: dict, best_model_name: str) -> str:
    """
    Generate a human-readable text report summarising evaluation results.

    Contents:
        * Best model name and its metrics
        * Per-model metric summary
        * Recommendations

    Args:
        all_results:     Model name → metrics dict.
        best_model_name: Name of the top-performing model.

    Returns:
        str: Absolute path to the saved report file.
    """
    ensure_directories()
    best_metrics = all_results.get(best_model_name, {})

    lines = [
        "=" * 70,
        " BREAST CANCER DIAGNOSIS — EVALUATION REPORT",
        "=" * 70,
        "",
        f"Best Model: {best_model_name}",
        "-" * 40,
    ]

    for metric, value in best_metrics.items():
        if value is not None:
            lines.append(f"  {metric:<15s}: {value:.4f}")
        else:
            lines.append(f"  {metric:<15s}: N/A")

    lines += [
        "",
        "Per-Model Summary",
        "-" * 40,
    ]
    for name, metrics in all_results.items():
        marker = "  ★" if name == best_model_name else "   "
        f1 = metrics.get("f1")
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        lines.append(f"{marker} {name:<25s}  F1={f1_str}")

    lines += [
        "",
        "Recommendations",
        "-" * 40,
        f"  1. Deploy '{best_model_name}' for production inference.",
        "  2. Monitor precision-recall trade-off for clinical use-cases.",
        "  3. Consider ensemble of top-2 models if marginal gains matter.",
        "  4. Re-evaluate periodically with fresh data.",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"📝 Evaluation report saved → {report_path}")
    logger.info("\n" + report_text)

    return report_path


# ─── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from src.model_training import train_all_models, save_models

    ensure_directories()
    logger.info("=" * 60)
    logger.info("  Evaluation — standalone run")
    logger.info("=" * 60)

    # Load & split
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=DATA_CONFIG["test_size"],
        random_state=DATA_CONFIG["random_state"],
        stratify=y,
    )

    # Train
    trained = train_all_models(X_train, y_train)
    save_models(trained)

    # Evaluate
    all_results = evaluate_all_models(trained, X_test, y_test)

    # Plots
    for name, info in trained.items():
        plot_confusion_matrix(info["model"], X_test, y_test, name)
        plot_feature_importance(info["model"], feature_names, name)

    plot_roc_curves(trained, X_test, y_test)

    # Comparison table & report
    results_df = generate_comparison_table(all_results)
    plot_model_comparison(results_df)

    # Best model
    best_name = results_df.index[0]
    generate_report(all_results, best_name)

    logger.info("Done.")
