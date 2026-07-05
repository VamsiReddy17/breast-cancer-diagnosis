# OncoSense — Unified Breast Cancer Diagnosis & Prognosis Pipeline

OncoSense is a production-grade, multi-modal clinical intelligence portal for breast cancer classification. It combines tabular cytological diagnosis, clinical staging survival prognosis, and deep learning histopathology image classification in a unified web application.

---

## 🖥️ Portal Interface Features

### 1. Wisconsin Cytology Diagnosis (Phase 1A & 1B)
*   Spreadsheet raw data explorer of 30 numeric cytological features.
*   ECharts comparison visualization of **5 classical ML models** (KNN, SVM, Random Forest, Logistic Regression, MLP).
*   Live numerical diagnosis inference panel.

### 2. SEER Clinical Survival Prognosis (Phase 1C)
*   Ingestion and scaling of the **4,024-patient SEER staging cohort**.
*   Real-time mortality/survival prediction based on clinical features (Staging, Tumor Size, Grade, Hormonal Receptor statuses).
*   Dynamic comparison plots and clinical correlation heatmaps.

### 3. Histopathology Image Classifier (Phase 2)
*   **EfficientNet-B0 CNN** binary classification (Benign vs. Malignant) trained on histopathology images.
*   Drag-and-drop file uploader with class logit probabilities and metric progress gauges.

---

## 🚀 Getting Started

### 1. Setup & Installation
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt
```

### 2. Running the Training Pipelines
Centralized deep learning configurations live in `config/deep_learning.yaml`. You can trigger either ML flow using the unified orchestrator:

*   **Train Classical ML (Wisconsin & SEER)**:
    ```bash
    python src/pipeline.py
    ```
*   **Train Deep Learning CNN (BreaKHis)**:
    ```bash
    python src/pipeline.py --cnn
    ```

### 3. Running Web Application Services Locally

**A. Start FastAPI Backend:**
```bash
source venv/bin/activate
python src/api.py
```
*API runs at `http://localhost:8000` (docs available at `/docs`).*

**B. Start Vite React Frontend:**
```bash
cd frontend
npm install
npm run dev
```
*Dashboard portal runs at `http://localhost:5173`.*

### 4. Running the Test Suite
```bash
./venv/bin/pytest
```

---

## ☁️ Deployment Environment Configuration

### Frontend (Vercel)
Set the environment variable:
*   `VITE_API_BASE_URL`: *[Your hosted Render backend API URL]*

### Backend (Render)
To keep the Git repository lightweight, `.pth` binary weights are ignored in Git. To make the **Image Classifier** work on the live server, configure:
*   `DL_MODEL_WEIGHTS_URL`: The direct download URL of your trained `best_model.pth` file (e.g., hosted on GitHub Releases, Dropbox, or GCS). 
    
    *On startup, the FastAPI backend will automatically download and cache the model weights from this URL.*

---

## 🔄 Project Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1A** | Wisconsin Dataset + 6 Classical ML Classifiers | ✅ Completed |
| **Phase 1B** | Interactive React + FastAPI Dashboard App | ✅ Completed |
| **Phase 1C** | SEER Dataset Ingestion, Grid Search Tuning, & Risk Prognosis | ✅ Completed |
| **Phase 2**  | PyTorch EfficientNet-B0 CNN Image Classifier | ✅ Completed |

