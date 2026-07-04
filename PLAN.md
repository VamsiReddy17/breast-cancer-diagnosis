# Project Plan — Breast Cancer Diagnosis

> **Last Updated**: 2026-07-04
> **Status**: Phase 1A — Completed (Phase 1B — Planned)

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

## Phase 1B: React + FastAPI Web Application (In Progress)

### Milestone 8: Backend API Development
- [ ] Save fitted scaler in training pipeline (`models/scaler.joblib`)
- [ ] Create `src/api.py` using FastAPI
- [ ] Endpoint `/api/data` for paginated, searchable tabular raw data
- [ ] Endpoint `/api/eda` for statistical summaries and figure asset maps
- [ ] Endpoint `/api/models` for comparison metrics and manifest parameters
- [ ] Endpoint `/api/predict` for processing input features and returning inference outcomes

### Milestone 9: Frontend Initialization & Setup
- [ ] Initialize Vite + React project in `frontend/`
- [ ] Setup visual layout matching Zinc color palette and light/dark theme toggle
- [ ] Configure Axios and ECharts

### Milestone 10: Tab 1 — Data & EDA Explorer
- [ ] Implement Excel-like paginated table showing raw sample rows
- [ ] Display visual charts (correlation heatmap, distributions)

### Milestone 11: Tab 2 — Model Comparison & Performance
- [ ] Add KPI dashboard cards of primary metric
- [ ] Integrate interactive model comparison charts and selectable confusion matrix visuals

### Milestone 12: Tab 3 — Live Diagnosis Form (Inference)
- [ ] Build feature input form with pre-defined ranges and model selector dropdown
- [ ] Create "Prefill Random Sample" helper buttons (Benign / Malignant presets)
- [ ] Setup prediction result viewer showing diagnosis classification & confidence gauges

### Milestone 13: E2E Verification & Integration
- [ ] Write integration test cases (`tests/test_api.py`)
- [ ] Confirm layout aesthetics and functionality with browser subagent

---

## Phase 1C: SEER Dataset (Planned)
- [ ] Download SEER dataset from IEEE Dataport
- [ ] Data cleaning & preprocessing (handle 4M records)
- [ ] Feature engineering for clinical data
- [ ] Survival prediction models
- [ ] Scale pipeline to handle large datasets

---

## Phase 2: Image-Based Deep Learning (Planned)
- [ ] Download BreakHis / CBIS-DDSM dataset
- [ ] Image preprocessing pipeline
- [ ] CNN architectures (ResNet, EfficientNet)
- [ ] Transfer learning
- [ ] Data augmentation
- [ ] GPU training setup

---

## Decision Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-04 | Start with Wisconsin dataset | Immediately available, good for pipeline validation |
| 2026-07-04 | Use 6 ML models | Comprehensive comparison, covers linear + nonlinear |
| 2026-07-04 | 10-Fold CV | Small dataset — maximize use of all data |
| 2026-07-04 | Self-developing loop pattern | Prevents hallucination, tracks all errors |
| 2026-07-04 | Add React+FastAPI Web App (Phase 1B) | User requested interactive web application with tabular data, EDA plots, model insights, and inference form. |
