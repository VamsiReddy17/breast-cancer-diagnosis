# Issue Tracker — Breast Cancer Diagnosis

> **Purpose**: Every error gets tracked with root cause + fix + prevention rule.
> LLMs and developers MUST read this before making changes to avoid repeating mistakes.

---

## How to Use This File

1. When an error occurs, add it below with a unique ID
2. Fill in ALL fields — no shortcuts
3. The **Prevention Rule** is the most important field — it prevents the same mistake
4. Check this file BEFORE writing new code

---

## Issue Template

```
### ISSUE-XXX: [Short Title]
- **Date**: YYYY-MM-DD
- **Component**: [file/module name]
- **Error**: [exact error message]
- **Root Cause**: [why it happened]
- **Fix Applied**: [what was changed]
- **Prevention Rule**: [rule to prevent recurrence]
- **Status**: 🔴 Open | 🟡 In Progress | 🟢 Resolved
```

---

## Known Pitfalls (Pre-populated)

> These are common mistakes in breast cancer ML projects. Read before coding.

### PITFALL-001: Data Leakage via Scaling Before Split
- **Component**: feature_engineering.py
- **Risk**: Fitting scaler on full dataset before train/test split leaks test info
- **Prevention Rule**: ALWAYS fit scaler on training data only, then transform test data
- **Status**: 🟢 Prevented by design

### PITFALL-002: Class Imbalance Ignored
- **Component**: model_training.py
- **Risk**: Wisconsin dataset is imbalanced (357B vs 212M). Models may bias toward majority class
- **Prevention Rule**: Use `class_weight='balanced'` or stratified sampling. Always report per-class metrics, not just accuracy
- **Status**: 🟢 Prevented by design

### PITFALL-003: Overfitting on Small Dataset
- **Component**: model_training.py
- **Risk**: Complex models (XGBoost, MLP) may overfit 569 samples
- **Prevention Rule**: Use cross-validation, regularization, and compare train vs test accuracy. Flag >5% gap as overfitting
- **Status**: 🟢 Prevented by design

### PITFALL-004: Feature Multicollinearity
- **Component**: feature_engineering.py
- **Risk**: Wisconsin features have high correlation (e.g., radius_mean ↔ perimeter_mean ↔ area_mean)
- **Prevention Rule**: Run correlation analysis in EDA. Consider PCA or dropping correlated features (r > 0.95)
- **Status**: 🟢 Prevented by design

### PITFALL-005: Hardcoded Paths
- **Component**: All files
- **Risk**: Absolute paths break on different machines
- **Prevention Rule**: ALL paths must come from config/config.py using os.path relative to project root
- **Status**: 🟢 Prevented by design

### PITFALL-006: Silent Failures
- **Component**: pipeline.py
- **Risk**: Errors caught but not logged, pipeline appears to succeed when it didn't
- **Prevention Rule**: Every try/except MUST log the error with traceback. Pipeline must report step-by-step PASS/FAIL status
- **Status**: 🟢 Prevented by design

### PITFALL-007: Non-Reproducible Results
- **Component**: model_training.py
- **Risk**: Random seeds not set, results change between runs
- **Prevention Rule**: Set random_state in ALL model constructors AND train_test_split. Store seed in config.py
- **Status**: 🟢 Prevented by design

---

## Active Issues

> No active issues yet. Issues will be logged here as they arise during development.

---

## Resolved Issues

### ISSUE-001: XGBoost import crashes — missing libomp on macOS
- **Date**: 2026-07-04
- **Component**: src/model_training.py (line 47)
- **Error**: `XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded. libomp.dylib not found.`
- **Root Cause**: XGBoost on macOS requires `libomp` (OpenMP runtime). When missing, it throws `XGBoostError` at import time — NOT `ImportError`. The try/except only caught `ImportError`, so the crash propagated.
- **Fix Applied**: Changed `except ImportError` to `except Exception` in model_training.py to catch all import-time failures gracefully. Added comment explaining why.
- **Prevention Rule**: When guarding optional dependency imports, ALWAYS use `except Exception` (not `except ImportError`) because some libraries throw custom errors during import when system dependencies are missing.
- **Status**: 🟢 Resolved
