"""
SEER Clinical Pipeline — Breast Cancer Diagnosis
=================================================
Downloads the SEER clinical breast cancer dataset, preprocesses and encodes
categorical columns, scales numerical attributes, performs hyperparameter
optimization for 5 classifiers, and renders clinical report figures.
"""

import os
import sys
import json
import urllib.request
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.feature_selection import mutual_info_classif

# ─── Path Setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import get_logger, timer

logger = get_logger("seer_pipeline")

DATA_URL = "https://raw.githubusercontent.com/Mahfuzmania/Breast-Cancer-Survival-Prediction/main/Breast_Cancer.csv"
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "seer_raw.csv")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SEER_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "seer")
SEER_FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "seer")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "reports", "results")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(SEER_MODELS_DIR, exist_ok=True)
os.makedirs(SEER_FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Categorical Mapping Configurations ─────────────────────────────────────────
RACE_MAP = {'White': 0, 'Black': 1, 'Other': 2}
MARITAL_MAP = {'Married': 0, 'Single': 1, 'Divorced': 2, 'Widowed': 3, 'Separated': 4}
T_MAP = {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3}
N_MAP = {'N1': 0, 'N2': 1, 'N3': 2}
STAGE_MAP = {'IIA': 0, 'IIB': 1, 'IIIA': 2, 'IIIB': 3, 'IIIC': 4}
DIFF_MAP = {
    'Well differentiated': 0,
    'Moderately differentiated': 1,
    'Poorly differentiated': 2,
    'Undifferentiated': 3
}
GRADE_MAP = {'1': 0, '2': 1, '3': 2, 'anaplastic; Grade IV': 3}
A_MAP = {'Regional': 0, 'Distant': 1}
ER_MAP = {'Negative': 0, 'Positive': 1}
PR_MAP = {'Negative': 0, 'Positive': 1}
STATUS_MAP = {'Alive': 1, 'Dead': 0}

@timer
def download_dataset():
    """Download raw SEER dataset if not present on disk."""
    if not os.path.exists(RAW_DATA_PATH):
        logger.info(f"Downloading raw SEER dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
        logger.info(f"Saved dataset successfully to {RAW_DATA_PATH}")
    else:
        logger.info("Raw SEER dataset already exists. Skipping download.")

@timer
def preprocess_dataset():
    """Read raw SEER data, encode clinical categoricals, and save sets."""
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Strip spaces from object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    # Apply Mappings
    df['target'] = df['Status'].map(STATUS_MAP)
    df.drop(columns=['Status'], inplace=True)

    df['Race'] = df['Race'].map(RACE_MAP)
    df['Marital Status'] = df['Marital Status'].map(MARITAL_MAP)
    df['T_stage'] = df['T_stage'].map(t_map := T_MAP)
    df['N Stage'] = df['N Stage'].map(n_map := N_MAP)
    df['6th Stage'] = df['6th Stage'].map(stage_map := STAGE_MAP)
    df['differentiate'] = df['differentiate'].map(diff_map := DIFF_MAP)
    df['Grade'] = df['Grade'].map(grade_map := GRADE_MAP)
    df['A Stage'] = df['A Stage'].map(a_map := A_MAP)
    df['Estrogen Status'] = df['Estrogen Status'].map(er_map := ER_MAP)
    df['Progesterone Status'] = df['Progesterone Status'].map(pr_map := PR_MAP)

    # Save cleaned full dataset for local visualization and tables
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, "seer_processed.csv"), index=False)
    
    # Split
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save split datasets
    X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "seer_X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "seer_X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "seer_y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "seer_y_test.csv"), index=False)

    logger.info(f"Preprocessing completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

@timer
def run_eda(X_train, y_train):
    """Generate and save EDA plots for the SEER dataset."""
    df_train = X_train.copy()
    df_train['target'] = y_train

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Class distribution
    plt.figure(figsize=(6, 4))
    counts = df_train['target'].value_counts()
    sns.barplot(x=counts.index.map({1: 'Alive', 0: 'Dead'}), y=counts.values, hue=counts.index, legend=False, palette='coolwarm')
    plt.title("SEER Survival Outcomes (Class Distribution)", fontsize=12, fontweight='bold')
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(SEER_FIGURES_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_train.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={"size": 7})
    plt.title("Clinical Feature Correlation Matrix", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SEER_FIGURES_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # 3. Age vs Stage distribution
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df_train, x='6th Stage', y='Age', hue='6th Stage', legend=False, palette='muted')
    plt.title("Patient Age Distribution across Cancer Stages", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SEER_FIGURES_DIR, "feature_distributions_violin.png"), dpi=150)
    plt.close()

    # 4. Feature importance score ranking using Mutual Info
    scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_df = pd.DataFrame({'feature': X_train.columns, 'mi_score': scores})
    mi_df = mi_df.sort_values(by='mi_score', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=mi_df, x='mi_score', y='feature', hue='feature', legend=False, palette='viridis')
    plt.title("Clinical Feature Relevance (Mutual Information Scores)", fontsize=11, fontweight='bold')
    plt.xlabel("Mutual Info Score")
    plt.tight_layout()
    plt.savefig(os.path.join(SEER_FIGURES_DIR, "mutual_info_scores.png"), dpi=150)
    plt.close()

    logger.info("SEER EDA visualization figures successfully exported.")

@timer
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Scale data, fit scaler, run GridSearchCV, and evaluate 5 models."""
    # Fit & save scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(SEER_MODELS_DIR, "scaler.joblib"))

    # Convert to DataFrame with original column names
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Models & Grid bounds
    models_config = {
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            }
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"]
            }
        },
        "SVM": {
            "model": SVC(class_weight='balanced', probability=True, random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight='balanced', random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 5, 10]
            }
        },
        "MLP": {
            "model": MLPClassifier(max_iter=500, early_stopping=True, random_state=42),
            "params": {
                "hidden_layer_sizes": [(64, 32)],
                "alpha": [0.001, 0.01]
            }
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_comparison_results = []
    roc_plot_data = {}

    for name, cfg in models_config.items():
        logger.info(f"Optimizing hyperparameters for {name}...")
        grid = GridSearchCV(
            estimator=cfg["model"],
            param_grid=cfg["params"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1
        )
        grid.fit(X_train_df, y_train)
        best_model = grid.best_estimator_
        
        # Save model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(best_model, os.path.join(SEER_MODELS_DIR, f"{safe_name}_best.joblib"))

        # Predict
        y_pred = best_model.predict(X_test_df)
        y_prob = [0.0, 0.0]
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test_df)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_prob = best_model.decision_function(X_test_df)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_prob)

        roc_plot_data[name] = (y_test, y_prob)

        logger.info(f"Model {name} Evaluation -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Macro: {f1:.4f}, AUC: {auc:.4f}")

        # Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Dead", "Alive"], yticklabels=["Dead", "Alive"])
        plt.title(f"{name} Confusion Matrix", fontsize=11, fontweight='bold')
        plt.ylabel("True Status")
        plt.xlabel("Predicted Status")
        plt.tight_layout()
        plt.savefig(os.path.join(SEER_FIGURES_DIR, f"confusion_matrix_{safe_name}.png"), dpi=150)
        plt.close()

        # Feature Importance (Random Forest only)
        if name == "Random Forest":
            importances = best_model.feature_importances_
            feat_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
            feat_df = feat_df.sort_values(by='importance', ascending=False)

            plt.figure(figsize=(8, 5))
            sns.barplot(data=feat_df, x='importance', y='feature', hue='feature', legend=False, palette='coolwarm')
            plt.title("Random Forest Clinical Feature Importance", fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(SEER_FIGURES_DIR, "feature_importance_random_forest.png"), dpi=150)
            plt.close()

        model_comparison_results.append({
            "name": name,
            "metrics": {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "roc_auc": round(float(auc), 4)
            },
            "parameters": grid.best_params_,
            "confusion_matrix_url": f"/static/figures/seer/confusion_matrix_{safe_name}.png",
            "feature_importance_url": f"/static/figures/seer/feature_importance_{safe_name}.png" if name == "Random Forest" else None
        })

    # Save multi-model ROC Curves Comparison Plot
    plt.figure(figsize=(7, 6))
    for name, (y_t, y_p) in roc_plot_data.items():
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc_val = roc_auc_score(y_t, y_p)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Clinical Prognosis ROC Curve Comparison', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(SEER_FIGURES_DIR, "roc_curves_all_models.png"), dpi=150)
    plt.close()

    # Save model comparison summary chart
    comp_names = [r["name"] for r in model_comparison_results]
    comp_f1 = [r["metrics"]["f1"] for r in model_comparison_results]
    comp_auc = [r["metrics"]["roc_auc"] for r in model_comparison_results]
    
    x_pos = np.arange(len(comp_names))
    width = 0.35
    
    plt.figure(figsize=(7, 4.5))
    plt.bar(x_pos - width/2, comp_f1, width, label='F1 Macro', color='#3b82f6')
    plt.bar(x_pos + width/2, comp_auc, width, label='ROC AUC', color='#10b981')
    
    plt.title('Clinical Model Outcomes Comparison', fontsize=12, fontweight='bold')
    plt.xticks(x_pos, comp_names)
    plt.ylabel('Score')
    plt.ylim([0, 1.1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SEER_FIGURES_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # Save output comparison json
    comparison_payload = {
        "models": model_comparison_results,
        "overall_plots": {
            "roc_curves": "/static/figures/seer/roc_curves_all_models.png",
            "model_comparison": "/static/figures/seer/model_comparison.png"
        }
    }
    with open(os.path.join(RESULTS_DIR, "seer_model_comparison.json"), "w") as f:
        json.dump(comparison_payload, f, indent=4)
        
    logger.info("SEER pipeline metrics and results JSON written successfully.")

def run_seer_pipeline():
    """Orchestrates end-to-end execution of the SEER clinical pipeline."""
    logger.info("=" * 60)
    logger.info("🚀 Starting SEER Clinical Survival ML Pipeline")
    logger.info("=" * 60)
    
    download_dataset()
    X_train, X_test, y_train, y_test = preprocess_dataset()
    run_eda(X_train, y_train)
    train_and_evaluate(X_train, X_test, y_train, y_test)
    
    logger.info("=" * 60)
    logger.info("🎉 SEER Clinical Survival ML Pipeline Completed Successfully")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_seer_pipeline()
