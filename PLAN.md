# Project Plan — Breast Cancer Diagnosis

> **Last Updated**: 2026-07-05
> **Status**: Phase 2 — Completed

---

## Phase 1A: Wisconsin Dataset + Classical ML

### Milestone 1: Project Setup ✅
- [x] Create project tree structure
- [x] Create PLAN.md, TRACKER.md, CHANGELOG.md
- [x] Create requirements.txt
- [x] Create config/config.py

### Milestone 2: Data Pipeline ✅
- [x] Load Wisconsin dataset from sklearn
- [x] Validate data integrity (nulls, shape, types)
- [x] Stratified train/test split (80/20)
- [x] Save raw + processed data with checksums
- [x] Create data_registry.json

### Milestone 3: Exploratory Data Analysis ✅
- [x] Class distribution plot
- [x] Feature correlation heatmap
- [x] Feature distributions by class (violin plots)
- [x] Statistical summary (mean, std, skew per feature)
- [x] Auto-save all plots to reports/figures/

### Milestone 4: Feature Engineering ✅
- [x] StandardScaler normalization
- [x] Feature selection (variance threshold, mutual info)
- [x] PCA analysis (variance explained plot)
- [x] Log all transformations

### Milestone 5: Model Training ✅
- [x] Logistic Regression (with regularization tuning)
- [x] Random Forest (n_estimators, max_depth tuning)
- [x] SVM (kernel, C, gamma tuning)
- [x] KNN (k, distance metric tuning)
- [x] XGBoost (learning rate, depth tuning)
- [x] MLP Neural Network (layers, dropout tuning)
- [x] 10-Fold Stratified Cross Validation for all
- [x] GridSearchCV for hyperparameter optimization

### Milestone 6: Evaluation & Reporting ✅
- [x] Accuracy, Precision, Recall, F1, AUC-ROC per model
- [x] Confusion matrix per model
- [x] ROC curve comparison plot
- [x] Model comparison table (CSV)
- [x] Feature importance (top 10 features)
- [x] Best model selection with justification

### Milestone 7: Testing & Validation ✅
- [x] pytest for data loader
- [x] pytest for model training
- [x] pytest for end-to-end pipeline
- [x] Pipeline dry run
- [x] Full pipeline execution

---

## Phase 1B: React + FastAPI Web Application ✅

### Milestone 8: Backend API Development ✅
- [x] Save fitted scaler in training pipeline (`models/scaler.joblib`)
- [x] Create `src/api.py` using FastAPI
- [x] Endpoint `/api/data` for paginated, searchable tabular raw data
- [x] Endpoint `/api/eda` for statistical summaries and figure asset maps
- [x] Endpoint `/api/models` for comparison metrics and manifest parameters
- [x] Endpoint `/api/predict` for processing input features and returning inference outcomes

### Milestone 9: Frontend Initialization & Setup ✅
- [x] Initialize Vite + React project in `frontend/`
- [x] Setup visual layout matching Zinc color palette and light/dark theme toggle
- [x] Configure Axios and ECharts

### Milestone 10: Tab 1 — Data & EDA Explorer ✅
- [x] Implement Excel-like paginated table showing raw sample rows with horizontal scrollbar
- [x] Display visual charts (correlation heatmap, distributions)

### Milestone 11: Tab 2 — Model Comparison & Performance ✅
- [x] Add KPI dashboard cards of primary metric
- [x] Integrate interactive model comparison charts and selectable confusion matrix visuals

### Milestone 12: Tab 3 — Live Diagnosis Form (Inference) ✅
- [x] Build feature input form with pre-defined ranges and model selector dropdown
- [x] Create "Prefill Random Sample" helper buttons (Benign / Malignant presets)
- [x] Setup prediction result viewer showing diagnosis classification & confidence gauges

### Milestone 13: E2E Verification & Integration ✅
- [x] Write integration test cases (`tests/test_api.py`)
- [x] Confirm layout aesthetics and functionality with browser subagent
- [x] Successfully deploy application services live on Render and Vercel

---

## Phase 1C: SEER Dataset ✅
- [x] Download SEER cohort dataset
- [x] Data cleaning & categorical mapping (4,024 patient records)
- [x] Feature engineering and scaling for clinical attributes
- [x] Train 5 survival prediction models (KNN, Logistic Regression, MLP, SVM, Random Forest)
- [x] Extend FastAPI backend routes and update React multi-mode dashboard switching

---

## Phase 2: Image-Based Deep Learning (Completed)

### Milestone 14: Image Dataset Ingestion & Preparation ✅
- [x] Write Python script (`src/download_images.py`) to programmatically download BreaKHis images using Kaggle API.
- [x] Extract, structure folders (`data/raw/images/benign`, `data/raw/images/malignant`).
- [x] Implement data loading script (`src/dataset.py`) defining train/val/test splits (70/15/15) with random rotation, color jitter, and normalization transforms.

### Milestone 15: Deep Learning CNN Modeling & Local Training ✅
- [x] Define PyTorch model script (`src/cnn_model.py`) supporting Transfer Learning (ResNet50 / EfficientNet) with custom binary classification head.
- [x] Implement training pipeline (`src/train_cnn.py`) configuring loss optimization, validation checks, and early stopping callbacks.
- [x] Output training logs to `logs/deep_learning.log` tracking training/validation loss and accuracy per epoch.
- [x] Save best weights model file to `models/deep_learning/best_model.pth`.

### Milestone 16: Model Evaluation & Visual Outputs ✅
- [x] Save training/validation loss curves and accuracy curves to `reports/figures/dl/loss_accuracy_curves.png`.
- [x] Generate confusion matrix and ROC curves on the test set, saving outputs to `reports/figures/dl/test_evaluation.png`.
- [x] Export final classification performance metrics (Accuracy, Precision, Recall, F1) to `reports/results/dl_metrics.json`.

### Milestone 17: FastAPI Endpoint & Frontend UI Integration ✅
- [x] Add FastAPI endpoint `POST /api/predict/image` accepting multi-part image file uploads, resizing, scaling, and running inference.
- [x] Add Tab 4: **Image Classifier** in the React frontend (`frontend/src/App.jsx`) with image drag-and-drop file uploader.
- [x] Display predictions, classification logits (Benign vs. Malignant), and confidence levels dynamically.

---

## Hand‑off & LLM Compatibility

- **Backbone model**: EfficientNet‑B0 (transfer learning) – lightweight, fast, portable across environments.
- **Training epochs**: 10 (early‑stopping enabled) – balances training time and performance.
- **GPU policy**: Use Apple‑Silicon MPS if `torch.backends.mps.is_available()`; otherwise fall back to CPU. This conditional logic ensures the pipeline runs on any machine without manual changes.
- **Documentation**: All new scripts include docstrings and a `requirements.txt` entry specifying `torch>=2.0`, `torchvision`, and `PyYAML`.
- **Hand‑off**: The implementation plan, issue tracker, and code comments explicitly mention the chosen defaults so a new LLM (or developer) can pick up without needing to infer design decisions.
- **Recent UI Fixes**: Removed stray duplicate `<form>` tag outside the Inference tab, ensured the inference submit button is correctly wrapped inside the form, and integrated the Image Classifier tab with proper file upload handling.

---

## Decision Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-04 | Start with Wisconsin dataset | Immediately available, good for pipeline validation |
| 2026-07-04 | Use 6 ML models | Comprehensive comparison, covers linear + nonlinear |
| 2026-07-04 | 10-Fold CV | Small dataset — maximize use of all data |
| 2026-07-04 | Self-developing loop pattern | Prevents hallucination, tracks all errors |
| 2026-07-04 | Add React+FastAPI Web App (Phase 1B) | User requested interactive web application with tabular data, EDA plots, model insights, and inference form. |
| 2026-07-05 | Implement SEER Clinical Survival Prediction (Phase 1C) | Expand portal scope to clinical staging and mortality risk outcomes using SEER Breast Cancer cohort. |
| 2026-07-05 | Add Dynamic Multi-Mode Switcher UI | Connect both Wisconsin FNA Cytology (Diagnosis) and SEER Clinical Cohort (Prognosis) in a unified React portal. |
