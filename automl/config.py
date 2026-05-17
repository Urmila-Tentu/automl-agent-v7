"""
AutoML Agent – Central Configuration
All tuneable knobs live here; override via environment variables or .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── Paths ──────────────────────────────────────────────────────────────────
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    models_dir: Path = BASE_DIR / "models"
    logs_dir: Path = BASE_DIR / "logs"
    experiments_dir: Path = BASE_DIR / "experiments"
    mlflow_tracking_uri: str = f"sqlite:///{BASE_DIR}/mlruns/mlflow.db"

    # ── API ────────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    secret_key: str = "CHANGE-ME-IN-PROD-USE-256-BIT-RANDOM"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ── Training ───────────────────────────────────────────────────────────────
    max_training_time_seconds: int = 600        # hard limit per experiment
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    n_trials_optuna: int = 30                   # HPO trials per model
    enable_deep_learning: bool = True
    dl_row_threshold: int = 5_000              # min rows to enable DL
    n_jobs: int = -1                            # parallelism (-1 = all cores)

    # ── Drift Detection ────────────────────────────────────────────────────────
    drift_check_interval_minutes: int = 60
    drift_threshold: float = 0.1

    # ── Inference ──────────────────────────────────────────────────────────────
    batch_size_limit: int = 10_000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure all required directories exist
for _dir in [settings.data_dir, settings.models_dir,
             settings.logs_dir, settings.experiments_dir,
             BASE_DIR / "mlruns"]:
    _dir.mkdir(parents=True, exist_ok=True)
