# Breast Cancer Diagnosis — Phase 1A

A production-grade breast cancer classification system using the Wisconsin Diagnostic dataset.

## 🎯 Project Goal
Build a self-documenting, error-tracking ML pipeline for breast cancer diagnosis (benign vs malignant) using classical ML models.

## 📊 Dataset
- **Name**: Wisconsin Breast Cancer Diagnostic
- **Samples**: 569 (357 benign, 212 malignant)
- **Features**: 30 numeric features (radius, texture, perimeter, area, smoothness, etc.)
- **Source**: Built-in `sklearn.datasets`

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python src/pipeline.py

# 4. Run tests
python -m pytest tests/ -v
```

## 📁 Project Structure
```
breast-cancer-diagnosis/
├── README.md               # This file
├── PLAN.md                 # Living project plan
├── TRACKER.md              # Issue tracker (error memory)
├── CHANGELOG.md            # Version history
├── requirements.txt        # Dependencies
├── config/config.py        # Centralized configuration
├── data/                   # Raw + processed data
├── src/                    # Source code
│   ├── data_loader.py      # Data loading & validation
│   ├── eda.py              # Exploratory Data Analysis
│   ├── feature_engineering.py  # Feature processing
│   ├── model_training.py   # Model training (6 models)
│   ├── evaluation.py       # Metrics & visualization
│   ├── pipeline.py         # End-to-end orchestrator
│   └── utils.py            # Logging & helpers
├── models/                 # Saved model artifacts
├── reports/                # Plots & result CSVs
├── tests/                  # Automated tests
└── logs/                   # Runtime logs
```

## 🔄 Phases
| Phase | Description | Status |
|-------|-------------|--------|
| 1A | Wisconsin dataset + Classical ML | 🟡 In Progress |
| 1B | SEER dataset (~4M records) | ⬜ Planned |
| 2  | Image-based DL (BreakHis/CBIS-DDSM) | ⬜ Planned |

## 📝 Key Files
- **PLAN.md** — What to do next
- **TRACKER.md** — What went wrong and how we fixed it
- **CHANGELOG.md** — What changed and when
