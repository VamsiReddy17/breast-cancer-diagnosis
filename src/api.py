"""
FastAPI Backend — Breast Cancer Diagnosis
==========================================
Provides REST API endpoints for:
  - Excel-like tabular raw data exploration
  - Statistical summary & EDA figures
  - Model metrics comparison
  - Live prediction (inference) using saved model pipelines
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import joblib
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ─── Path Bootstrap ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    RAW_DATA_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, DATA_CONFIG, DL_CONFIG
)
from src.utils import get_logger, load_json

logger = get_logger("api")

# ─── Self-Healing Startup Check ────────────────────────────────────────────────
# Render/Heroku container builds may discard ignored files. If models or comparison
# results are missing on startup, dynamically trigger the ML pipeline to train them.
STARTUP_ERROR = None
try:
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    if not os.path.exists(comparison_path) or not os.path.exists(scaler_path):
        logger.info("⚠️ Deployed model assets or comparisons are missing. Initializing pipeline...")
        from src.pipeline import run_pipeline
        run_pipeline()
        logger.info("✅ ML Pipeline training completed successfully on startup.")
except Exception as e:
    import traceback
    STARTUP_ERROR = {
        "error": str(e),
        "traceback": traceback.format_exc()
    }
    logger.error(f"❌ Failed to execute self-healing pipeline: {str(e)}")

app = FastAPI(
    title="Breast Cancer Diagnosis API",
    description="Backend API for data exploration, model evaluation, and inference.",
    version="1.0.0"
)

# Enable CORS for React dev server (usually http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev purposes, allow all. In prod, restrict.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount figures directory so the frontend can load images directly
if os.path.exists(FIGURES_DIR):
    app.mount("/static/figures", StaticFiles(directory=FIGURES_DIR), name="figures")
else:
    logger.warning(f"Figures directory not found: {FIGURES_DIR}")

# ─── Helper Functions ───────────────────────────────────────────────────────────

def get_raw_data_path() -> str:
    path = os.path.join(RAW_DATA_DIR, "breast_cancer_raw.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Raw data file not found. Please run pipeline first.")
    return path


# ─── Inference Schema ───────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use (e.g., 'KNN', 'Logistic Regression')")
    features: Dict[str, float] = Field(..., description="Dict mapping features to their float values")


# ─── API Routes ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}


@app.get("/api/data")
def get_data(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    search: str = Query(None),
    target: int = Query(None, ge=0, le=1)
):
    """
    Get a paginated, searchable list of raw dataset records.
    """
    csv_path = get_raw_data_path()
    df = pd.read_csv(csv_path)
    
    # Add an ID column if it doesn't exist
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    # Apply search filter (if search query matches ID or any column)
    if search:
        search = search.lower()
        # Create a boolean mask across all columns
        mask = df.astype(str).apply(lambda x: x.str.lower().str.contains(search)).any(axis=1)
        df = df[mask]

    # Apply target filter
    if target is not None:
        df = df[df[DATA_CONFIG["target_column"]] == target]

    total_records = len(df)
    total_pages = int(np.ceil(total_records / size))
    
    # Slice the dataframe for pagination
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    sliced_df = df.iloc[start_idx:end_idx]

    records = sliced_df.to_dict(orient="records")

    return {
        "page": page,
        "size": size,
        "total_records": total_records,
        "total_pages": total_pages,
        "columns": list(df.columns),
        "data": records
    }


@app.get("/api/data/random")
def get_random_sample(target: int = Query(None, ge=0, le=1)):
    """
    Returns a random sample from the dataset (used for pre-filling input forms).
    """
    csv_path = get_raw_data_path()
    df = pd.read_csv(csv_path)
    
    if target is not None:
        filtered_df = df[df[DATA_CONFIG["target_column"]] == target]
        if len(filtered_df) == 0:
            raise HTTPException(status_code=404, detail=f"No samples with target={target}")
        sample = filtered_df.sample(1, random_state=np.random.randint(1, 10000))
    else:
        sample = df.sample(1, random_state=np.random.randint(1, 10000))

    return sample.to_dict(orient="records")[0]


@app.get("/api/eda")
def get_eda_details():
    """
    Returns summary stats and filenames of generated figures.
    """
    summary_path = os.path.join(RESULTS_DIR, "statistical_summary.csv")
    if not os.path.exists(summary_path):
        # Generate dynamically if missing
        csv_path = get_raw_data_path()
        df = pd.read_csv(csv_path)
        feature_df = df.drop(columns=[DATA_CONFIG["target_column"]], errors="ignore")
        summary = feature_df.describe().T
        summary["skew"] = feature_df.skew()
        summary["kurtosis"] = feature_df.kurtosis()
        summary = summary.round(4)
        summary.index.name = "feature"
        stats_list = summary.reset_index().to_dict(orient="records")
    else:
        summary_df = pd.read_csv(summary_path)
        # Check column name mapping
        if "Unnamed: 0" in summary_df.columns:
            summary_df.rename(columns={"Unnamed: 0": "feature"}, inplace=True)
        stats_list = summary_df.to_dict(orient="records")

    # Available static plots (filenames)
    plots = [
        {"name": "Class Distribution", "filename": "class_distribution.png", "type": "bar"},
        {"name": "Correlation Heatmap", "filename": "correlation_heatmap.png", "type": "heatmap"},
        {"name": "Feature Violin Plots", "filename": "feature_distributions_violin.png", "type": "violin"},
        {"name": "Top 5 Feature Pairplot", "filename": "pairplot_top5.png", "type": "pairplot"},
        {"name": "PCA Variance Explained", "filename": "pca_variance_explained.png", "type": "line"},
        {"name": "Mutual Information Feature Importance", "filename": "mutual_info_scores.png", "type": "bar"}
    ]

    return {
        "statistical_summary": stats_list,
        "plots": plots,
        "base_url": "/static/figures/"
    }


@app.get("/api/models")
def get_model_details():
    """
    Returns metrics comparison and configuration of all models.
    """
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    manifest_path = os.path.join(MODELS_DIR, "model_manifest.json")
    
    if not os.path.exists(comparison_path):
        raise HTTPException(status_code=500, detail="Model comparison results missing. Run pipeline first.")
        
    comparison_data = load_json(comparison_path)
    manifest_data = load_json(manifest_path) if os.path.exists(manifest_path) else {}

    # Standardize comparison data structure
    formatted_models = []
    for model_name, metrics in comparison_data.items():
        formatted_models.append({
            "name": model_name,
            "metrics": metrics,
            "parameters": manifest_data.get(model_name, {}).get("params", {}),
            "confusion_matrix_url": f"/static/figures/confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
            "feature_importance_url": f"/static/figures/feature_importance_{model_name.lower().replace(' ', '_')}.png" if model_name in ["Random Forest", "XGBoost"] else None
        })

    # Available overall visual comparisons
    overall_plots = {
        "roc_curves": "/static/figures/roc_curves_all_models.png",
        "model_comparison": "/static/figures/model_comparison.png"
    }

    return {
        "models": formatted_models,
        "overall_plots": overall_plots
    }
# ─── SEER Clinical API Routes ──────────────────────────────────────────────────

SEER_MODELS_DIR = os.path.join(MODELS_DIR, "seer")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def get_seer_data_path() -> str:
    path = os.path.join(PROCESSED_DATA_DIR, "seer_processed.csv")
    if not os.path.exists(path):
        logger.info("⚠️ SEER processed dataset is missing. Running SEER pipeline...")
        try:
            from src.seer_pipeline import run_seer_pipeline
            run_seer_pipeline()
            logger.info("✅ SEER pipeline completed successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to run SEER pipeline: {str(e)}")
            raise HTTPException(status_code=500, detail=f"SEER data missing. Failed to run pipeline: {str(e)}")
    return path

@app.get("/api/seer/data")
def get_seer_data(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    search: str = Query(None),
    target: int = Query(None, ge=0, le=1)
):
    """
    Get a paginated, searchable list of SEER raw/processed clinical dataset records.
    """
    csv_path = get_seer_data_path()
    df = pd.read_csv(csv_path)
    
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    if search:
        search = search.lower()
        mask = df.astype(str).apply(lambda x: x.str.lower().str.contains(search)).any(axis=1)
        df = df[mask]

    if target is not None:
        df = df[df["target"] == target]

    total_records = len(df)
    total_pages = int(np.ceil(total_records / size))
    
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    sliced_df = df.iloc[start_idx:end_idx]

    records = sliced_df.to_dict(orient="records")

    return {
        "page": page,
        "size": size,
        "total_records": total_records,
        "total_pages": total_pages,
        "columns": list(df.columns),
        "data": records
    }


@app.get("/api/seer/data/random")
def get_seer_random_sample(target: int = Query(None, ge=0, le=1)):
    """
    Returns a random sample from the SEER dataset for pre-filling form inputs.
    """
    csv_path = get_seer_data_path()
    df = pd.read_csv(csv_path)
    
    if target is not None:
        filtered_df = df[df["target"] == target]
        if len(filtered_df) == 0:
            raise HTTPException(status_code=404, detail=f"No samples with target={target}")
        sample = filtered_df.sample(1, random_state=np.random.randint(1, 10000))
    else:
        sample = df.sample(1, random_state=np.random.randint(1, 10000))

    return sample.to_dict(orient="records")[0]


@app.get("/api/seer/eda")
def get_seer_eda_details():
    """
    Returns summary stats and filenames of generated figures for the SEER dataset.
    """
    csv_path = get_seer_data_path()
    df = pd.read_csv(csv_path)
    
    # Exclude non-feature columns
    features_df = df.drop(columns=["target", "id"], errors="ignore")
    summary = features_df.describe().T
    summary["skew"] = features_df.skew()
    summary["kurtosis"] = features_df.kurtosis()
    summary = summary.round(4)
    summary.index.name = "feature"
    stats_list = summary.reset_index().to_dict(orient="records")

    plots = [
        {"name": "Class Distribution", "filename": "seer/class_distribution.png", "type": "bar"},
        {"name": "Correlation Heatmap", "filename": "seer/correlation_heatmap.png", "type": "heatmap"},
        {"name": "Age across Stages", "filename": "seer/feature_distributions_violin.png", "type": "box"},
        {"name": "Mutual Information Clinical Relevance", "filename": "seer/mutual_info_scores.png", "type": "bar"}
    ]

    return {
        "statistical_summary": stats_list,
        "plots": plots,
        "base_url": "/static/figures/"
    }


@app.get("/api/seer/models")
def get_seer_model_details():
    """
    Returns metrics comparison and configuration of all SEER models.
    """
    comparison_path = os.path.join(RESULTS_DIR, "seer_model_comparison.json")
    if not os.path.exists(comparison_path):
        get_seer_data_path()
        
    if not os.path.exists(comparison_path):
        raise HTTPException(status_code=500, detail="SEER model comparison results missing.")
        
    return load_json(comparison_path)


@app.post("/api/seer/predict")
def run_seer_prediction(payload: InferenceRequest):
    """
    Accepts 15 SEER clinical features, scales them, and runs inference.
    """
    try:
        scaler_path = os.path.join(SEER_MODELS_DIR, "scaler.joblib")
        if not os.path.exists(scaler_path):
            get_seer_data_path()
            
        if not os.path.exists(scaler_path):
            raise HTTPException(status_code=500, detail="SEER fitted scaler missing.")
        
        scaler = joblib.load(scaler_path)
        
        safe_model_name = payload.model_name.lower().replace(" ", "_")
        model_path = os.path.join(SEER_MODELS_DIR, f"{safe_model_name}_best.joblib")
        
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Fitted model '{payload.model_name}' not found for SEER. Supported: KNN, MLP, SVM, Random Forest, Logistic Regression"
            )
        
        model = joblib.load(model_path)
        
        seer_columns = [
            'Age', 'Race', 'Marital Status', 'T_stage', 'N Stage', '6th Stage',
            'differentiate', 'Grade', 'A Stage', 'Tumor Size', 'Estrogen Status',
            'Progesterone Status', 'Regional Node Examined', 'Reginol Node Positive',
            'Survival Months'
        ]

        missing_features = [col for col in seer_columns if col not in payload.features]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Incomplete clinical features. Missing columns: {missing_features}"
            )

        input_df = pd.DataFrame([payload.features])[seer_columns]
        scaled_array = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled_array, columns=seer_columns)
        
        prediction_idx = int(model.predict(scaled_df)[0])
        
        probabilities = [0.0, 0.0]
        if hasattr(model, "predict_proba"):
            probabilities = [float(p) for p in model.predict_proba(scaled_df)[0]]
        elif hasattr(model, "decision_function"):
            decision = float(model.decision_function(scaled_df)[0])
            p1 = 1 / (1 + np.exp(-decision))
            probabilities = [1 - p1, p1]
        else:
            probabilities[prediction_idx] = 1.0

        class_names = ["Dead", "Alive"]
        class_name = class_names[prediction_idx]
        
        return {
            "model_used": payload.model_name,
            "prediction": prediction_idx,
            "class_label": class_name,
            "confidence": round(probabilities[prediction_idx] * 100, 2),
            "probabilities": {
                "Dead": round(probabilities[0] * 100, 2),
                "Alive": round(probabilities[1] * 100, 2)
            }
        }
    except Exception as e:
        import traceback
        err_msg = f"SEER inference failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(err_msg)
        raise HTTPException(status_code=500, detail=err_msg)





@app.post("/api/predict")
def run_prediction(payload: InferenceRequest):
    """
    Accepts 30 features, scales them using the saved scaler, and runs inference.
    """
    try:
        # 1. Load scaler
        scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
        if not os.path.exists(scaler_path):
            raise HTTPException(status_code=500, detail="Fitted scaler missing in models/ directory.")
        
        scaler = joblib.load(scaler_path)
        
        # 2. Resolve model path
        safe_model_name = payload.model_name.lower().replace(" ", "_")
        model_path = os.path.join(MODELS_DIR, f"{safe_model_name}_best.joblib")
        
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Fitted model '{payload.model_name}' not found. Supported: KNN, MLP, SVM, Random Forest, Logistic Regression"
            )
        
        model = joblib.load(model_path)
        
        # 3. Align features
        # Sklearn wisconsin bunch features are sorted in specific order
        csv_path = get_raw_data_path()
        df_columns = list(pd.read_csv(csv_path).columns)
        df_columns.remove(DATA_CONFIG["target_column"])
        if "id" in df_columns:
            df_columns.remove("id")

        # Validate feature count and presence
        input_values = []
        missing_features = []
        for col in df_columns:
            if col in payload.features:
                input_values.append(payload.features[col])
            else:
                missing_features.append(col)
                
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Incomplete features array. Missing columns: {missing_features}"
            )

        # 4. Convert to DataFrame and scale
        input_df = pd.DataFrame([payload.features])[df_columns]
        scaled_array = scaler.transform(input_df)
        
        # 5. Predict
        # Wrap scaled array in DataFrame with correct column names to prevent model warnings
        scaled_df = pd.DataFrame(scaled_array, columns=df_columns)
        prediction_idx = int(model.predict(scaled_df)[0])
        
        # Calculate probabilities/confidence if model supports it
        probabilities = [0.0, 0.0]
        if hasattr(model, "predict_proba"):
            probabilities = [float(p) for p in model.predict_proba(scaled_df)[0]]
        elif hasattr(model, "decision_function"):
            # SVM fallback for distance margins
            decision = float(model.decision_function(scaled_df)[0])
            # Simple sigmoid mapping to mock probability
            p1 = 1 / (1 + np.exp(-decision))
            probabilities = [1 - p1, p1]
        else:
            # Default fallback
            probabilities[prediction_idx] = 1.0

        class_name = DATA_CONFIG["class_names"][prediction_idx] # 0 = Malignant, 1 = Benign
        
        # Format return dictionary
        return {
            "model_used": payload.model_name,
            "prediction": prediction_idx,  # 0 or 1
            "class_label": class_name,     # "Malignant" or "Benign"
            "confidence": round(probabilities[prediction_idx] * 100, 2), # Percentage
            "probabilities": {
                DATA_CONFIG["class_names"][0]: round(probabilities[0] * 100, 2),
                DATA_CONFIG["class_names"][1]: round(probabilities[1] * 100, 2)
            }
        }
    except Exception as e:
        import traceback
        err_msg = f"Inference failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(err_msg)
        raise HTTPException(status_code=500, detail=err_msg)


@app.get("/api/debug")
def get_debug_info():
    log_path = os.path.join(PROJECT_ROOT, "logs", "pipeline.log")
    log_contents = ""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                log_contents = f.read()[-5000:] # Last 5000 chars
        except Exception:
            pass
            
    # Check what files exist
    files_check = {
        "model_comparison_exists": os.path.exists(os.path.join(RESULTS_DIR, "model_comparison.json")),
        "scaler_exists": os.path.exists(os.path.join(MODELS_DIR, "scaler.joblib")),
        "model_manifest_exists": os.path.exists(os.path.join(MODELS_DIR, "model_manifest.json")),
        "figures_dir_exists": os.path.exists(FIGURES_DIR),
        "figures_list": os.listdir(FIGURES_DIR) if os.path.exists(FIGURES_DIR) else [],
        "seer_model_comparison_exists": os.path.exists(os.path.join(RESULTS_DIR, "seer_model_comparison.json")),
        "seer_scaler_exists": os.path.exists(os.path.join(SEER_MODELS_DIR, "scaler.joblib")),
        "seer_figures_list": os.listdir(os.path.join(FIGURES_DIR, "seer")) if os.path.exists(os.path.join(FIGURES_DIR, "seer")) else []
    }
            
    return {
        "startup_error": STARTUP_ERROR,
        "files_check": files_check,
        "log_tail": log_contents
    }



@app.post("/api/predict/image")
async def run_image_prediction(file: UploadFile = File(...)):
    """
    Accepts an uploaded histopathology image file, runs inference on the PyTorch CNN model,
    and returns malignant/benign probabilities.
    """
    try:
        model_path = os.path.join(MODELS_DIR, "deep_learning", "best_model.pth")
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=500, 
                detail="Deep learning model weights missing in models/deep_learning/ directory. Train model first."
            )

        from src.cnn_model import HistopathologyCNN
        
        # Load model structure and weights on CPU
        backbone_name = DL_CONFIG["backbone"]
        model = HistopathologyCNN(model_name=backbone_name, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        # Read image
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Apply evaluation transforms
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = val_transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
            prediction_idx = torch.argmax(probs).item()

        class_name = "Malignant" if prediction_idx == 1 else "Benign"
        confidence = round(probs[prediction_idx].item() * 100, 2)
        
        return {
            "model_used": f"{backbone_name.replace('_', '-').title()} CNN",
            "prediction": prediction_idx,
            "class_label": class_name,
            "confidence": confidence,
            "probabilities": {
                "Benign": round(probs[0].item() * 100, 2),
                "Malignant": round(probs[1].item() * 100, 2)
            }
        }
    except Exception as e:
        import traceback
        err_msg = f"Image inference failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(err_msg)
        raise HTTPException(status_code=500, detail=err_msg)


# ─── Launch Helper ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting local API development server on port 8000 …")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
