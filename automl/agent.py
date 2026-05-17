"""
AutoML Agent – Main Orchestrator
Single entry point that wires together the entire AutoML pipeline:
  data → EDA → preprocessing → training → evaluation → explainability → deployment
"""
from __future__ import annotations

import json
import pickle
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from automl.config import settings
from automl.data_handler import EDAEngine, detect_problem_type, load_dataset
from automl.drift import DriftDetector
from automl.explainability import ExplainabilityEngine
from automl.preprocessor import (
    PreprocessingEngine,
    TargetLabelEncoder,
    remove_outliers_isolation_forest,
)
from automl.trainer import TrainingEngine


# ─────────────────────────────────────────────────────────────────────────────
# Experiment artifact structure persisted to disk
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentArtifacts:
    """Everything saved for a completed AutoML run."""

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        self.run_dir = settings.experiments_dir / experiment_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save(self, agent: "AutoMLAgent") -> None:
        # Best model
        model_path = self.run_dir / "best_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(agent.best_model, f)

        # Preprocessing pipeline
        pp_path = self.run_dir / "preprocessor.pkl"
        with open(pp_path, "wb") as f:
            pickle.dump(agent.preprocessing_engine.best_pipeline, f)

        # Label encoder (classification only)
        if agent.label_encoder and agent.label_encoder.is_fitted:
            le_path = self.run_dir / "label_encoder.pkl"
            with open(le_path, "wb") as f:
                pickle.dump(agent.label_encoder, f)

        # Metadata JSON
        meta = {
            "experiment_id":    self.experiment_id,
            "problem_type":     agent.problem_type,
            "target_col":       agent.target_col,
            "feature_names":    agent.feature_names,
            "best_model_name":  agent.best_model_name,
            "leaderboard":      [
                {k: v for k, v in r.items() if k != "fitted_model"}
                for r in agent.leaderboard
            ],
            "eda_report":       agent.eda_report,
            "explainability":   agent.explainability_report,
            "preprocessing_scores": agent.preprocessing_engine.get_scores(),
            "preprocessing_strategy": agent.preprocessing_engine.best_strategy,
        }
        meta_path = self.run_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, default=_json_serialise)
        )
        logger.info(f"Experiment artifacts saved to {self.run_dir}")

    @classmethod
    def load(cls, experiment_id: str) -> Dict[str, Any]:
        run_dir = settings.experiments_dir / experiment_id
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found.")

        data: Dict[str, Any] = {}
        data["meta"] = json.loads(meta_path.read_text())

        model_path = run_dir / "best_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                data["model"] = pickle.load(f)

        pp_path = run_dir / "preprocessor.pkl"
        if pp_path.exists():
            with open(pp_path, "rb") as f:
                data["preprocessor"] = pickle.load(f)

        le_path = run_dir / "label_encoder.pkl"
        if le_path.exists():
            with open(le_path, "rb") as f:
                data["label_encoder"] = pickle.load(f)

        return data


def _json_serialise(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Main Agent
# ─────────────────────────────────────────────────────────────────────────────

class AutoMLAgent:
    """
    End-to-end AutoML orchestrator.

    Usage
    -----
    >>> agent = AutoMLAgent(target_col="species")
    >>> result = agent.run("iris.csv")
    """

    def __init__(
        self,
        target_col: str,
        problem_type: Optional[str] = None,  # auto-detect if None
        datetime_col: Optional[str] = None,
        feature_selection: str = "mutual_info",
        handle_outliers: bool = True,
        remove_outlier_rows: bool = False,
        n_trials: int = settings.n_trials_optuna,
        cv_folds: int = settings.cv_folds,
        test_size: float = settings.test_size,
        enable_dl: bool = settings.enable_deep_learning,
    ) -> None:
        self.target_col = target_col
        self.problem_type = problem_type
        self.datetime_col = datetime_col
        self.feature_selection = feature_selection
        self.handle_outliers = handle_outliers
        self.remove_outlier_rows = remove_outlier_rows
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.enable_dl = enable_dl

        # Set during run()
        self.df: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_val:   Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_val:   Optional[np.ndarray] = None

        self.preprocessing_engine: Optional[PreprocessingEngine] = None
        self.label_encoder: Optional[TargetLabelEncoder] = None
        self.training_engine: Optional[TrainingEngine] = None

        self.feature_names:         List[str] = []
        self.eda_report:            Dict = {}
        self.leaderboard:           List[Dict] = []
        self.best_model_name:       Optional[str] = None
        self.best_model             = None
        self.explainability_report: Dict = {}
        self.experiment_id:         Optional[str] = None

    # ── public entry point ─────────────────────────────────────────────────────

    def run(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Full pipeline: load → EDA → preprocess → train → explain → save.
        Returns a summary dict with all results.
        """
        wall_start = time.time()
        self.experiment_id = str(uuid.uuid4())[:8]
        logger.info(f"▶ AutoML run started  [experiment_id={self.experiment_id}]")

        # 1. Load data
        self.df = load_dataset(file_path)
        self._validate_target()

        # 2. Detect problem type
        if self.problem_type is None:
            self.problem_type = detect_problem_type(
                self.df, self.target_col, self.datetime_col
            )

        # 3. EDA
        eda = EDAEngine(self.df, self.target_col)
        self.eda_report = eda.run()

        # 4. Prepare features / target
        X_raw, y_raw = self._prepare_xy()

        # 5. Remove outlier rows (optional)
        if self.remove_outlier_rows:
            mask = remove_outliers_isolation_forest(X_raw)
            X_raw, y_raw = X_raw[mask], y_raw[mask]

        # 6. Train/val split
        stratify = y_raw if self.problem_type == "classification" else None
        X_tr_raw, X_v_raw, y_tr, y_v = train_test_split(
            X_raw, y_raw,
            test_size=self.test_size,
            random_state=settings.random_state,
            stratify=stratify,
        )

        # 7. Encode target if classification
        if self.problem_type == "classification":
            self.label_encoder = TargetLabelEncoder()
            y_tr = self.label_encoder.fit_transform(y_tr)
            y_v  = self.label_encoder.transform(y_v)

        # 8. Preprocessing — use LLM-chosen strategy if available
        llm_plan_early = getattr(self, "_llm_plan", None) or {}
        forced_pp_strategy = llm_plan_early.get("preprocessing_strategy")  # e.g. "robust"
        forced_feat_sel    = llm_plan_early.get("feature_selection") or self.feature_selection

        if forced_pp_strategy:
            logger.info(f"🤖 LLM chose preprocessing: {forced_pp_strategy}, feature_sel: {forced_feat_sel}")

        self.preprocessing_engine = PreprocessingEngine(
            df=pd.concat([X_tr_raw, X_v_raw]),
            target_col=self.target_col,
            problem_type=self.problem_type,
            feature_selection=forced_feat_sel,
            handle_outliers=self.handle_outliers,
        )
        best_pp = self.preprocessing_engine.build_best_pipeline(
            X_tr_raw, pd.Series(y_tr), X_v_raw, pd.Series(y_v),
        )

        # Transform datasets
        X_tr = best_pp.transform(X_tr_raw[self.preprocessing_engine.all_feature_cols])
        X_v  = best_pp.transform(X_v_raw[self.preprocessing_engine.all_feature_cols])

        # Infer feature names post-transformation (best effort)
        self.feature_names = self._infer_feature_names(best_pp, X_tr.shape[1])

        self.X_train, self.y_train = X_tr, y_tr
        self.X_val,   self.y_val   = X_v,  y_v

        # ── Read LLM plan if set by AgenticAutoML ────────────────────────────
        llm_plan = getattr(self, "_llm_plan", None) or {}
        model_filter   = llm_plan.get("recommended_models") or None
        agent_n_trials = llm_plan.get("n_trials")

        logger.info("=" * 60)
        logger.info(f"🤖 LLM PLAN APPLIED TO TRAINING")
        logger.info(f"   _llm_plan present : {bool(llm_plan)}")
        logger.info(f"   model_filter      : {model_filter}")
        logger.info(f"   n_trials from plan: {agent_n_trials}")
        logger.info(f"   current n_trials  : {self.n_trials}")
        logger.info("=" * 60)

        if agent_n_trials and agent_n_trials != self.n_trials:
            logger.info(f"🤖 LLM overrides n_trials: {self.n_trials} → {agent_n_trials}")
            self.n_trials = int(agent_n_trials)

        # 9. Train selected models
        self.training_engine = TrainingEngine(
            problem_type=self.problem_type,
            n_trials=self.n_trials,
            cv_folds=self.cv_folds,
            random_state=settings.random_state,
            enable_dl=self.enable_dl,
            dl_row_threshold=settings.dl_row_threshold,
        )
        self.leaderboard = self.training_engine.train_all(
            X_tr, y_tr, X_v, y_v,
            experiment_name=f"automl_{self.experiment_id}",
            model_filter=model_filter,         # ← LLM-chosen models only
        )
        self.best_model_name, self.best_model = self.training_engine.get_best_model()

        # 10. Explainability
        exp_engine = ExplainabilityEngine(
            model=self.best_model,
            X_train=X_tr,
            X_val=X_v,
            feature_names=self.feature_names,
            problem_type=self.problem_type,
        )
        self.explainability_report = exp_engine.run()

        # 11. Save artifacts
        artifacts = ExperimentArtifacts(self.experiment_id)
        artifacts.save(self)

        wall_time = round(time.time() - wall_start, 1)
        logger.info(f"✅ AutoML run complete  [wall_time={wall_time}s]")

        return self._build_summary(wall_time)

    # ── inference ──────────────────────────────────────────────────────────────

    def predict(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict on new data using the best fitted model.
        """
        if self.best_model is None:
            raise RuntimeError("Call run() first to train the model.")

        feature_cols = self.preprocessing_engine.all_feature_cols  # type: ignore[union-attr]
        missing = [c for c in feature_cols if c not in input_df.columns]
        if missing:
            raise ValueError(f"Input is missing columns: {missing}")

        X = self.preprocessing_engine.best_pipeline.transform(  # type: ignore[union-attr]
            input_df[feature_cols]
        )
        y_pred_raw = self.best_model.predict(X)

        if self.problem_type == "classification" and self.label_encoder and self.label_encoder.is_fitted:
            y_pred = self.label_encoder.inverse_transform(y_pred_raw).tolist()
        else:
            y_pred = y_pred_raw.tolist()

        result: Dict[str, Any] = {"predictions": y_pred}

        if self.problem_type == "classification" and hasattr(self.best_model, "predict_proba"):
            probs = self.best_model.predict_proba(X).tolist()
            result["probabilities"] = probs
            if self.label_encoder and self.label_encoder.classes_ is not None:
                result["classes"] = self.label_encoder.classes_.tolist()

        return result

    # ── helpers ────────────────────────────────────────────────────────────────

    def _validate_target(self) -> None:
        # 1. Strip whitespace from ALL column names first
        self.df.columns = self.df.columns.str.strip()

        # 2. Exact match (after strip)
        if self.target_col in self.df.columns:
            return

        # 3. Case-insensitive match
        lower_map = {c.lower(): c for c in self.df.columns}
        if self.target_col.lower() in lower_map:
            actual = lower_map[self.target_col.lower()]
            logger.warning(
                f"Target column '{self.target_col}' matched case-insensitively "
                f"to '{actual}' — using '{actual}'."
            )
            self.target_col = actual
            return

        # 4. Nothing matched — give a helpful error
        raise ValueError(
            f"Target column '{self.target_col}' not found.\n"
            f"Available columns: {list(self.df.columns)}\n"
            f"Tip: check for extra spaces or wrong capitalisation."
        )

    def _prepare_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.df.dropna(subset=[self.target_col]).copy()
        feature_cols = [c for c in df.columns if c != self.target_col]
        if self.datetime_col and self.datetime_col in feature_cols:
            feature_cols.remove(self.datetime_col)
        return df[feature_cols], df[self.target_col]

    @staticmethod
    def _infer_feature_names(pipeline, n_out: int) -> List[str]:
        """Best-effort feature name inference post-transformation."""
        try:
            if hasattr(pipeline, "get_feature_names_out"):
                return list(pipeline.get_feature_names_out())
        except Exception:
            pass
        return [f"feature_{i}" for i in range(n_out)]

    def _build_summary(self, wall_time: float) -> Dict[str, Any]:
        primary = "accuracy" if self.problem_type == "classification" else "r2"
        best_metrics = self.leaderboard[0]["metrics"] if self.leaderboard else {}

        return {
            "experiment_id":      self.experiment_id,
            "problem_type":       self.problem_type,
            "target_col":         self.target_col,
            "dataset_shape":      list(self.df.shape),  # type: ignore[union-attr]
            "best_model":         self.best_model_name,
            "best_metrics":       best_metrics,
            "leaderboard": [
                {k: v for k, v in row.items() if k != "fitted_model"}
                for row in self.leaderboard
            ],
            "preprocessing": {
                "best_strategy": self.preprocessing_engine.best_strategy,  # type: ignore[union-attr]
                "scores":        self.preprocessing_engine.get_scores(),    # type: ignore[union-attr]
                "report":        getattr(self.preprocessing_engine, "report", {}),
            },
            "top_features":   self.explainability_report.get("top_features", []),
            "eda_report":     self.eda_report,
            "wall_time_sec":  wall_time,
        }
