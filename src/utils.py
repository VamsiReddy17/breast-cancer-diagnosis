"""
Utilities — Breast Cancer Diagnosis
=====================================
Centralized logging, timing, validation, and I/O helpers.
Every other module imports from here.
"""

import os
import sys
import time
import json
import hashlib
import logging
import functools
import traceback
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import LOG_CONFIG, LOGS_DIR


# ─── Logger Setup ───────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    Logs to both console and file simultaneously.

    Args:
        name: Logger name (usually module name)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_CONFIG["log_level"]))

    formatter = logging.Formatter(
        LOG_CONFIG["log_format"],
        datefmt=LOG_CONFIG["date_format"]
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    os.makedirs(LOGS_DIR, exist_ok=True)
    file_handler = logging.FileHandler(LOG_CONFIG["log_file"])
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ─── Decorators ─────────────────────────────────────────────────────────────────

def timer(func):
    """
    Decorator to measure and log function execution time.
    Usage: @timer
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__ or "timer")
        start = time.time()
        logger.info(f"▶ START: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"✅ DONE: {func.__name__} ({elapsed:.2f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"❌ FAIL: {func.__name__} ({elapsed:.2f}s) — {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper


def safe_execute(step_name: str):
    """
    Decorator for pipeline steps. Catches errors, logs them, and returns
    a status dict instead of crashing.

    Usage: @safe_execute("Data Loading")
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger("pipeline")
            status = {
                "step": step_name,
                "status": "UNKNOWN",
                "start_time": datetime.now().isoformat(),
                "error": None,
            }
            try:
                result = func(*args, **kwargs)
                status["status"] = "PASS"
                status["end_time"] = datetime.now().isoformat()
                logger.info(f"✅ PASS: {step_name}")
                return result, status
            except Exception as e:
                status["status"] = "FAIL"
                status["error"] = str(e)
                status["traceback"] = traceback.format_exc()
                status["end_time"] = datetime.now().isoformat()
                logger.error(f"❌ FAIL: {step_name} — {str(e)}")
                logger.error(traceback.format_exc())
                return None, status
        return wrapper
    return decorator


# ─── Data Validation ────────────────────────────────────────────────────────────

def validate_dataframe(df, expected_rows=None, expected_cols=None, name="DataFrame"):
    """
    Validate a pandas DataFrame for common issues.

    Args:
        df: pandas DataFrame to validate
        expected_rows: Expected number of rows (optional)
        expected_cols: Expected number of columns (optional)
        name: Name for logging

    Returns:
        dict with validation results

    Raises:
        ValueError: If critical validation fails
    """
    logger = get_logger("validation")
    issues = []

    # Check nulls
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        issues.append(f"Found {null_count} null values")
        logger.warning(f"{name}: {null_count} null values detected")

    # Check duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"Found {dup_count} duplicate rows")
        logger.warning(f"{name}: {dup_count} duplicate rows detected")

    # Check expected shape
    if expected_rows and len(df) != expected_rows:
        issues.append(f"Expected {expected_rows} rows, got {len(df)}")
        logger.warning(f"{name}: Row count mismatch")

    if expected_cols and len(df.columns) != expected_cols:
        issues.append(f"Expected {expected_cols} columns, got {len(df.columns)}")
        logger.warning(f"{name}: Column count mismatch")

    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    inf_count = df[numeric_cols].isin([float("inf"), float("-inf")]).sum().sum()
    if inf_count > 0:
        issues.append(f"Found {inf_count} infinite values")
        logger.warning(f"{name}: {inf_count} infinite values detected")

    result = {
        "name": name,
        "shape": df.shape,
        "nulls": null_count,
        "duplicates": dup_count,
        "infinites": inf_count,
        "issues": issues,
        "is_valid": len(issues) == 0,
    }

    if result["is_valid"]:
        logger.info(f"✅ {name} validation passed: shape={df.shape}")
    else:
        logger.warning(f"⚠️ {name} validation: {len(issues)} issue(s) found")

    return result


# ─── File I/O Helpers ───────────────────────────────────────────────────────────

def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file for data versioning."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json(data: dict, filepath: str):
    """Save dictionary as JSON with pretty formatting."""
    logger = get_logger("io")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON: {filepath}")


def load_json(filepath: str) -> dict:
    """Load JSON file as dictionary."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_plot(fig, filename: str, directory: str):
    """
    Save a matplotlib figure to the specified directory.

    Args:
        fig: matplotlib Figure object
        filename: Name of the file (e.g., "correlation_heatmap.png")
        directory: Target directory path
    """
    logger = get_logger("io")
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved plot: {filepath}")
    return filepath


# ─── Pipeline Helpers ───────────────────────────────────────────────────────────

def generate_run_summary(step_statuses: list) -> dict:
    """
    Generate a summary of pipeline run from step statuses.

    Args:
        step_statuses: List of status dicts from safe_execute decorated functions

    Returns:
        Summary dict with counts and details
    """
    passed = sum(1 for s in step_statuses if s["status"] == "PASS")
    failed = sum(1 for s in step_statuses if s["status"] == "FAIL")
    skipped = sum(1 for s in step_statuses if s["status"] == "SKIP")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_steps": len(step_statuses),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "success": failed == 0,
        "steps": step_statuses,
    }

    return summary


def print_banner(title: str, char: str = "═", width: int = 70):
    """Print a formatted banner for pipeline steps."""
    border = char * width
    padding = (width - len(title) - 2) // 2
    print(f"\n{border}")
    print(f"{char}{' ' * padding}{title}{' ' * (width - padding - len(title) - 2)}{char}")
    print(f"{border}\n")
