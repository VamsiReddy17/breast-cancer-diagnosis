"""
Pipeline Orchestrator — Breast Cancer Diagnosis
=================================================
Runs the entire ML pipeline end-to-end in sequence:
Load → Validate → EDA → Feature Engineering → Train → Evaluate → Report

Each step has error handling with logging to TRACKER.
Step status tracking (PASS/FAIL/SKIP) ensures nothing fails silently.
"""

import os
import sys
import json
from datetime import datetime

# ─── Path Setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    ensure_directories, RESULTS_DIR, LOGS_DIR, FIGURES_DIR, MODELS_DIR
)
from src.utils import (
    get_logger, timer, safe_execute, generate_run_summary,
    print_banner, save_json
)
from src.data_loader import load_breast_cancer_data, validate_data, split_data, save_data
from src.eda import run_eda
from src.feature_engineering import run_feature_engineering
from src.model_training import train_all_models, save_models
from src.evaluation import (
    evaluate_all_models, plot_confusion_matrix, plot_roc_curves,
    plot_model_comparison, plot_feature_importance,
    generate_comparison_table, generate_report
)

logger = get_logger("pipeline")


# ─── Pipeline Steps ─────────────────────────────────────────────────────────────

@safe_execute("Step 1: Data Loading")
def step_load_data():
    """Load breast cancer dataset from sklearn."""
    df = load_breast_cancer_data()
    logger.info(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    return df


@safe_execute("Step 2: Data Validation")
def step_validate_data(df):
    """Validate loaded data for integrity."""
    validation = validate_data(df)
    if not validation["is_valid"]:
        logger.warning(f"Validation issues: {validation['issues']}")
    return validation


@safe_execute("Step 3: Data Splitting")
def step_split_data(df):
    """Split data into train/test sets with stratification."""
    X_train, X_test, y_train, y_test = split_data(df)
    logger.info(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


@safe_execute("Step 4: Save Raw Data")
def step_save_data(df, X_train, X_test, y_train, y_test):
    """Save raw and processed data to disk."""
    save_data(df, X_train, X_test, y_train, y_test)
    return True


@safe_execute("Step 5: Exploratory Data Analysis")
def step_eda(df):
    """Run full EDA with auto-generated plots."""
    run_eda(df, target_col="target")
    return True


@safe_execute("Step 6: Feature Engineering")
def step_feature_engineering(X_train, X_test, y_train):
    """Apply scaling, feature selection, and transformations."""
    result = run_feature_engineering(X_train, X_test, y_train)
    return result


@safe_execute("Step 7: Model Training")
def step_train_models(X_train_processed, y_train):
    """Train all 6 models with hyperparameter tuning."""
    trained_models = train_all_models(X_train_processed, y_train)
    return trained_models


@safe_execute("Step 8: Save Models")
def step_save_models(trained_models):
    """Save trained models to disk."""
    save_models(trained_models)
    return True


@safe_execute("Step 9: Model Evaluation")
def step_evaluate_models(trained_models, X_test_processed, y_test):
    """Evaluate all models and generate metrics."""
    all_results = evaluate_all_models(trained_models, X_test_processed, y_test)
    return all_results


@safe_execute("Step 10: Generate Visualizations")
def step_generate_visualizations(trained_models, X_test_processed, y_test, feature_names):
    """Generate confusion matrices, ROC curves, and feature importance plots."""
    # Confusion matrices for each model
    for name, model_data in trained_models.items():
        model = model_data["model"]
        try:
            plot_confusion_matrix(model, X_test_processed, y_test, name)
        except Exception as e:
            logger.warning(f"Could not plot confusion matrix for {name}: {e}")

    # ROC curves comparison
    try:
        plot_roc_curves(trained_models, X_test_processed, y_test)
    except Exception as e:
        logger.warning(f"Could not plot ROC curves: {e}")

    # Feature importance for tree-based models
    for name in ["Random Forest", "XGBoost"]:
        if name in trained_models:
            try:
                plot_feature_importance(
                    trained_models[name]["model"], feature_names, name
                )
            except Exception as e:
                logger.warning(f"Could not plot feature importance for {name}: {e}")

    return True


@safe_execute("Step 11: Generate Reports")
def step_generate_reports(all_results):
    """Generate comparison table and final report."""
    # Comparison table
    results_df = generate_comparison_table(all_results)

    # Find best model
    if results_df is not None and len(results_df) > 0:
        best_model_name = results_df.index[0]
        logger.info(f"🏆 Best Model: {best_model_name}")

        # Final report
        generate_report(all_results, best_model_name)
    else:
        logger.warning("No results to generate report from")

    return results_df


# ─── Main Pipeline ──────────────────────────────────────────────────────────────

@timer
def run_pipeline():
    """
    Execute the full breast cancer diagnosis pipeline.

    Steps:
        1. Load Data
        2. Validate Data
        3. Split Data
        4. Save Raw Data
        5. Exploratory Data Analysis
        6. Feature Engineering
        7. Model Training
        8. Save Models
        9. Model Evaluation
        10. Generate Visualizations
        11. Generate Reports

    Returns:
        dict: Pipeline run summary with step statuses
    """
    print_banner("BREAST CANCER DIAGNOSIS PIPELINE — Phase 1A")
    logger.info("=" * 70)
    logger.info("PIPELINE START")
    logger.info("=" * 70)

    ensure_directories()
    step_statuses = []

    # ── Step 1: Load Data ────────────────────────────────────────────────────
    print_banner("Step 1: Data Loading", char="─")
    df, status = step_load_data()
    step_statuses.append(status)
    if df is None:
        logger.error("Pipeline aborted: Data loading failed")
        return generate_run_summary(step_statuses)

    # ── Step 2: Validate Data ────────────────────────────────────────────────
    print_banner("Step 2: Data Validation", char="─")
    validation, status = step_validate_data(df)
    step_statuses.append(status)

    # ── Step 3: Split Data ───────────────────────────────────────────────────
    print_banner("Step 3: Data Splitting", char="─")
    split_result, status = step_split_data(df)
    step_statuses.append(status)
    if split_result is None:
        logger.error("Pipeline aborted: Data splitting failed")
        return generate_run_summary(step_statuses)
    X_train, X_test, y_train, y_test = split_result

    # ── Step 4: Save Raw Data ────────────────────────────────────────────────
    print_banner("Step 4: Saving Data", char="─")
    _, status = step_save_data(df, X_train, X_test, y_train, y_test)
    step_statuses.append(status)

    # ── Step 5: EDA ──────────────────────────────────────────────────────────
    print_banner("Step 5: Exploratory Data Analysis", char="─")
    _, status = step_eda(df)
    step_statuses.append(status)

    # ── Step 6: Feature Engineering ──────────────────────────────────────────
    print_banner("Step 6: Feature Engineering", char="─")
    fe_result, status = step_feature_engineering(X_train, X_test, y_train)
    step_statuses.append(status)
    if fe_result is None:
        logger.error("Pipeline aborted: Feature engineering failed")
        return generate_run_summary(step_statuses)

    X_train_processed = fe_result["X_train_scaled"]
    X_test_processed = fe_result["X_test_scaled"]
    feature_names = fe_result.get("feature_names", list(X_train.columns))

    # ── Step 7: Model Training ───────────────────────────────────────────────
    print_banner("Step 7: Model Training", char="─")
    trained_models, status = step_train_models(X_train_processed, y_train)
    step_statuses.append(status)
    if trained_models is None:
        logger.error("Pipeline aborted: Model training failed")
        return generate_run_summary(step_statuses)

    # ── Step 8: Save Models ──────────────────────────────────────────────────
    print_banner("Step 8: Saving Models", char="─")
    _, status = step_save_models(trained_models)
    step_statuses.append(status)

    # ── Step 9: Model Evaluation ─────────────────────────────────────────────
    print_banner("Step 9: Model Evaluation", char="─")
    all_results, status = step_evaluate_models(
        trained_models, X_test_processed, y_test
    )
    step_statuses.append(status)

    # ── Step 10: Generate Visualizations ─────────────────────────────────────
    if all_results:
        print_banner("Step 10: Generating Visualizations", char="─")
        _, status = step_generate_visualizations(
            trained_models, X_test_processed, y_test, feature_names
        )
        step_statuses.append(status)
    else:
        step_statuses.append({
            "step": "Step 10: Generate Visualizations",
            "status": "SKIP",
            "error": "No evaluation results available",
        })

    # ── Step 11: Generate Reports ────────────────────────────────────────────
    if all_results:
        print_banner("Step 11: Generating Reports", char="─")
        _, status = step_generate_reports(all_results)
        step_statuses.append(status)
    else:
        step_statuses.append({
            "step": "Step 11: Generate Reports",
            "status": "SKIP",
            "error": "No evaluation results available",
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = generate_run_summary(step_statuses)

    print_banner("PIPELINE COMPLETE", char="═")
    logger.info(f"Total Steps: {summary['total_steps']}")
    logger.info(f"Passed: {summary['passed']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Skipped: {summary['skipped']}")
    logger.info(f"Success: {'✅ YES' if summary['success'] else '❌ NO'}")

    # Save run summary
    summary_path = os.path.join(RESULTS_DIR, "pipeline_run_summary.json")
    save_json(summary, summary_path)
    logger.info(f"Run summary saved: {summary_path}")

    # Print step-by-step status
    print("\n📋 Step Status:")
    print("-" * 50)
    for s in step_statuses:
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(s["status"], "❓")
        print(f"  {icon} {s['step']}: {s['status']}")
    print("-" * 50)

    return summary


# ─── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "🔬" * 35)
    print("  BREAST CANCER DIAGNOSIS — Phase 1A")
    print("  Wisconsin Diagnostic Dataset")
    print("  Self-Developing Pipeline v0.1.0")
    print("🔬" * 35 + "\n")

    summary = run_pipeline()

    if summary["success"]:
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Check reports at: reports/")
        print(f"📈 Check figures at: reports/figures/")
        print(f"🤖 Check models at: models/")
        print(f"📝 Check logs at: logs/pipeline.log")
    else:
        print("\n❌ Pipeline completed with errors!")
        print("Check TRACKER.md and logs/pipeline.log for details.")
        failed = [s for s in summary["steps"] if s["status"] == "FAIL"]
        for f in failed:
            print(f"  ❌ {f['step']}: {f.get('error', 'Unknown error')}")
