"""
AutoML Agent – Test Suite
Tests cover: data loading, EDA, preprocessing, training, and API endpoints.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def iris_df() -> pd.DataFrame:
    """Tiny Iris-like classification dataset."""
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        "sepal_length": np.random.normal(5.8, 0.8, n),
        "sepal_width":  np.random.normal(3.1, 0.4, n),
        "petal_length": np.random.normal(3.7, 1.8, n),
        "petal_width":  np.random.normal(1.2, 0.8, n),
        "species": np.random.choice(["setosa", "versicolor", "virginica"], n),
    })


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Simple regression dataset."""
    np.random.seed(0)
    n = 200
    X = np.random.randn(n, 4)
    y = 3 * X[:, 0] - 2 * X[:, 1] + np.random.randn(n) * 0.5
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = y
    return df


@pytest.fixture
def iris_csv(iris_df, tmp_path) -> Path:
    path = tmp_path / "iris.csv"
    iris_df.to_csv(path, index=False)
    return path


@pytest.fixture
def api_client():
    from api.main import app
    return TestClient(app)


@pytest.fixture
def auth_token(api_client) -> str:
    res = api_client.post(
        "/auth/token",
        data={"username": "admin", "password": "admin123"},
    )
    assert res.status_code == 200
    return res.json()["access_token"]


@pytest.fixture
def auth_headers(auth_token) -> dict:
    return {"Authorization": f"Bearer {auth_token}"}


# ─────────────────────────────────────────────────────────────────────────────
# Data handler tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataHandler:
    def test_load_csv(self, iris_csv):
        from automl.data_handler import load_dataset
        df = load_dataset(iris_csv)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 150

    def test_load_unsupported(self, tmp_path):
        from automl.data_handler import load_dataset
        bad = tmp_path / "data.xyz"
        bad.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset(bad)

    def test_detect_classification(self, iris_df):
        from automl.data_handler import detect_problem_type
        pt = detect_problem_type(iris_df, "species")
        assert pt == "classification"

    def test_detect_regression(self, regression_df):
        from automl.data_handler import detect_problem_type
        pt = detect_problem_type(regression_df, "target")
        assert pt == "regression"

    def test_eda_runs(self, iris_df):
        from automl.data_handler import EDAEngine
        engine = EDAEngine(iris_df, "species")
        report = engine.run()
        assert "shape" in report
        assert "missing" in report
        assert "target_summary" in report
        assert report["shape"]["rows"] == 150


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_builds_pipeline(self, iris_df):
        from automl.preprocessor import PreprocessingEngine
        from sklearn.model_selection import train_test_split

        X = iris_df.drop(columns=["species"])
        y = iris_df["species"]
        X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=0.2, random_state=42)

        engine = PreprocessingEngine(
            df=iris_df, target_col="species", problem_type="classification"
        )
        pipeline = engine.build_best_pipeline(X_tr, y_tr, X_v, y_v)
        assert pipeline is not None
        assert engine.best_strategy is not None

        X_transformed = pipeline.transform(X_v[engine.all_feature_cols])
        assert X_transformed.shape[0] == len(X_v)

    def test_label_encoder(self):
        from automl.preprocessor import TargetLabelEncoder
        le = TargetLabelEncoder()
        y = pd.Series(["cat", "dog", "cat", "bird"])
        encoded = le.fit_transform(y)
        assert len(encoded) == 4
        decoded = le.inverse_transform(encoded)
        assert list(decoded) == ["cat", "dog", "cat", "bird"]


# ─────────────────────────────────────────────────────────────────────────────
# Trainer tests (lightweight – no HPO)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainer:
    def test_trains_classification(self):
        from automl.trainer import TrainingEngine
        from sklearn.datasets import load_iris

        iris = load_iris()
        X_tr, X_v = iris.data[:100], iris.data[100:]
        y_tr, y_v = iris.target[:100], iris.target[100:]

        engine = TrainingEngine(
            problem_type="classification",
            n_trials=0,        # skip HPO for speed
            cv_folds=2,
            enable_dl=False,
        )
        leaderboard = engine.train_all(X_tr, y_tr, X_v, y_v)
        assert len(leaderboard) > 0
        assert "model_name" in leaderboard[0]
        assert "metrics" in leaderboard[0]
        assert "accuracy" in leaderboard[0]["metrics"]

    def test_trains_regression(self):
        from automl.trainer import TrainingEngine
        from sklearn.datasets import load_diabetes

        data = load_diabetes()
        X_tr, X_v = data.data[:300], data.data[300:]
        y_tr, y_v = data.target[:300], data.target[300:]

        engine = TrainingEngine(
            problem_type="regression",
            n_trials=0,
            cv_folds=2,
            enable_dl=False,
        )
        leaderboard = engine.train_all(X_tr, y_tr, X_v, y_v)
        assert any("r2" in r["metrics"] for r in leaderboard)


# ─────────────────────────────────────────────────────────────────────────────
# API endpoint tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAPI:
    def test_health(self, api_client):
        res = api_client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

    def test_login_success(self, api_client):
        res = api_client.post(
            "/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        assert res.status_code == 200
        assert "access_token" in res.json()

    def test_login_failure(self, api_client):
        res = api_client.post(
            "/auth/token",
            data={"username": "admin", "password": "wrongpass"},
        )
        assert res.status_code == 401

    def test_me_authenticated(self, api_client, auth_headers):
        res = api_client.get("/me", headers=auth_headers)
        # /me isn't defined, but /auth/me is
        res = api_client.get("/auth/me", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["username"] == "admin"

    def test_me_unauthenticated(self, api_client):
        res = api_client.get("/auth/me")
        assert res.status_code == 401

    def test_list_experiments(self, api_client, auth_headers):
        res = api_client.get("/experiments", headers=auth_headers)
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_start_training(self, api_client, auth_headers, iris_csv):
        with open(iris_csv, "rb") as f:
            res = api_client.post(
                "/experiments/train",
                headers={"Authorization": auth_headers["Authorization"]},
                files={"file": ("iris.csv", f, "text/csv")},
                data={
                    "target_col": "species",
                    "problem_type": "classification",
                    "n_trials": "0",
                    "handle_outliers": "true",
                },
            )
        assert res.status_code == 200
        data = res.json()
        assert "experiment_id" in data
        return data["experiment_id"]

    def test_metrics_endpoint(self, api_client):
        res = api_client.get("/metrics")
        assert res.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Drift detection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDrift:
    def test_no_drift(self, iris_df, tmp_path):
        from automl.drift import DriftDetector
        features = iris_df.drop(columns=["species"])
        detector = DriftDetector(features, tmp_path)
        report = detector.check(features.sample(50, random_state=1))
        assert "overall_drift_detected" in report
        assert isinstance(report["drift_rate"], float)

    def test_drift_detected(self, tmp_path):
        from automl.drift import DriftDetector
        ref = pd.DataFrame({"x": np.random.normal(0, 1, 200)})
        cur = pd.DataFrame({"x": np.random.normal(10, 1, 100)})   # very different!
        detector = DriftDetector(ref, tmp_path)
        report = detector.check(cur)
        assert report["overall_drift_detected"] is True
