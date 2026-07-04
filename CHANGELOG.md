# Changelog — Breast Cancer Diagnosis

All notable changes to this project will be documented in this file.

---

## [0.1.0] — 2026-07-04

### Added
- Project structure with 8 components
- README.md with setup instructions
- PLAN.md with phase tracking
- TRACKER.md with pre-populated pitfalls
- CHANGELOG.md (this file)
- requirements.txt with all dependencies
- config/config.py — centralized configuration
- src/utils.py — logging, timing, validation helpers
- src/data_loader.py — data loading with validation
- src/eda.py — exploratory data analysis with auto-plots
- src/feature_engineering.py — scaling, selection, PCA
- src/model_training.py — 6 ML models with cross-validation
- src/evaluation.py — metrics, confusion matrix, ROC curves
- src/pipeline.py — end-to-end orchestrator
- tests/ — automated test suite
- data/ — raw and processed data directories
- models/ — saved model artifacts directory
- reports/ — figures and results directories
- logs/ — pipeline runtime logs
