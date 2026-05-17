"""AutoML Agent – Training Engine  (fixed HPO param names + ensembles + speed)"""
from __future__ import annotations
import time, warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import mlflow; HAS_MLFLOW=True
except: HAS_MLFLOW=False
try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING); HAS_OPTUNA=True
except: HAS_OPTUNA=False
try:
    from loguru import logger
except:
    import logging; logger=logging.getLogger("automl")

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBClassifier, XGBRegressor; HAS_XGB=True
except: HAS_XGB=False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor; HAS_LGBM=True
except: HAS_LGBM=False
try:
    from catboost import CatBoostClassifier, CatBoostRegressor; HAS_CAT=True
except: HAS_CAT=False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# HPO spaces — keys MUST match exact sklearn __init__ param names
# ─────────────────────────────────────────────────────────────────────────────
def _xgb_space(trial, is_clf):
    return dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 500, step=100),
        max_depth        = trial.suggest_int("max_depth", 3, 9),
        learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
    )

def _lgbm_space(trial, is_clf):
    return dict(
        n_estimators  = trial.suggest_int("n_estimators", 100, 500, step=100),
        max_depth     = trial.suggest_int("max_depth", 3, 9),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        num_leaves    = trial.suggest_int("num_leaves", 20, 150),
        subsample     = trial.suggest_float("subsample", 0.5, 1.0),
        reg_alpha     = trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
    )

def _cat_space(trial, is_clf):
    return dict(
        iterations    = trial.suggest_int("iterations", 100, 500, step=100),
        depth         = trial.suggest_int("depth", 3, 8),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    )

def _rf_space(trial, is_clf):
    return dict(
        n_estimators     = trial.suggest_int("n_estimators", 50, 400, step=50),
        max_depth        = trial.suggest_int("max_depth", 3, 25),
        min_samples_split= trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    )

def _et_space(trial, is_clf):
    return dict(
        n_estimators     = trial.suggest_int("n_estimators", 50, 300, step=50),
        max_depth        = trial.suggest_int("max_depth", 3, 20),
        min_samples_split= trial.suggest_int("min_samples_split", 2, 15),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8),
    )

def _gb_space(trial, is_clf):
    return dict(
        n_estimators  = trial.suggest_int("n_estimators", 50, 300, step=50),
        max_depth     = trial.suggest_int("max_depth", 2, 7),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        subsample     = trial.suggest_float("subsample", 0.5, 1.0),
    )

def _svm_space(trial, is_clf):
    return dict(
        C      = trial.suggest_float("C", 1e-3, 100, log=True),
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"]),
    )

def _knn_space(trial, is_clf):
    return dict(
        n_neighbors = trial.suggest_int("n_neighbors", 3, 30),
        weights     = trial.suggest_categorical("weights", ["uniform", "distance"]),
    )

def _mlp_space(trial, is_clf):
    h = trial.suggest_categorical("hidden_layer_sizes", ["64", "128", "256", "128_64", "256_128"])
    return dict(
        hidden_layer_sizes = tuple(int(x) for x in h.split("_")),
        alpha              = trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    )

def _lr_space(trial, is_clf):
    return dict(C = trial.suggest_float("C", 1e-4, 100, log=True))

def _ridge_space(trial, is_clf):
    return dict(alpha = trial.suggest_float("alpha", 1e-3, 100, log=True))


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
def _build_registry(is_clf: bool, fast: bool = False) -> Dict[str, Tuple]:
    reg = {}
    reg["logistic_regression" if is_clf else "ridge"] = (
        LogisticRegression if is_clf else Ridge,
        _lr_space if is_clf else _ridge_space,
        {"max_iter": 500, "random_state": 42} if is_clf else {"random_state": 42},
    )
    reg["random_forest"] = (
        RandomForestClassifier if is_clf else RandomForestRegressor,
        _rf_space, {"random_state": 42, "n_jobs": -1},
    )
    reg["extra_trees"] = (
        ExtraTreesClassifier if is_clf else ExtraTreesRegressor,
        _et_space, {"random_state": 42, "n_jobs": -1},
    )
    reg["gradient_boosting"] = (
        GradientBoostingClassifier if is_clf else GradientBoostingRegressor,
        _gb_space, {"random_state": 42},
    )
    reg["adaboost"] = (
        AdaBoostClassifier if is_clf else AdaBoostRegressor,
        None,
        {"random_state": 42, "n_estimators": 100,
         **({"algorithm": "SAMME"} if is_clf else {})},  # suppress SAMME.R deprecation
    )
    reg["knn"] = (
        KNeighborsClassifier if is_clf else KNeighborsRegressor,
        _knn_space, {"n_jobs": -1},
    )
    if is_clf:
        reg["naive_bayes"] = (GaussianNB, None, {})

    # Slow models — only include when not in fast mode
    if not fast:
        reg["svm"] = (
            SVC if is_clf else SVR,
            _svm_space,
            {"probability": True} if is_clf else {},
        )
        reg["mlp"] = (
            MLPClassifier if is_clf else MLPRegressor,
            _mlp_space, {"max_iter": 300, "random_state": 42},
        )

    if HAS_XGB:
        reg["xgboost"] = (
            XGBClassifier if is_clf else XGBRegressor,
            _xgb_space,
            {"eval_metric": "logloss" if is_clf else "rmse",
             "verbosity": 0, "random_state": 42},
        )
    if HAS_LGBM:
        reg["lightgbm"] = (
            LGBMClassifier if is_clf else LGBMRegressor,
            _lgbm_space,
            {"verbose": -1, "random_state": 42, "n_jobs": -1},
        )
    if HAS_CAT and not fast:
        reg["catboost"] = (
            CatBoostClassifier if is_clf else CatBoostRegressor,
            _cat_space, {"verbose": 0, "random_state": 42},
        )
    return reg


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(problem_type, y_true, y_pred, y_prob=None) -> Dict[str, float]:
    m = {}
    if problem_type == "classification":
        m["accuracy"]    = round(float(accuracy_score(y_true, y_pred)), 4)
        m["f1_weighted"] = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
        m["f1_macro"]    = round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4)
        if y_prob is not None:
            try:
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    m["roc_auc"] = round(float(roc_auc_score(y_true, y_prob[:, 1])), 4)
                elif y_prob.ndim == 2:
                    m["roc_auc"] = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")), 4)
            except Exception:
                pass
    else:
        mse = mean_squared_error(y_true, y_pred)
        m["rmse"] = round(float(np.sqrt(mse)), 4)
        m["mae"]  = round(float(mean_absolute_error(y_true, y_pred)), 4)
        m["r2"]   = round(float(r2_score(y_true, y_pred)), 4)
        m["mape"] = round(float(np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / (np.abs(np.array(y_true)) + 1e-8))) * 100), 4)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Training Engine
# ─────────────────────────────────────────────────────────────────────────────
class TrainingEngine:
    def __init__(self, problem_type, n_trials=10, cv_folds=5, random_state=42,
                 max_time_per_model=60, enable_dl=False, dl_row_threshold=5000,
                 enable_ensemble=True):
        self.problem_type = problem_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.max_time_per_model = max_time_per_model
        self.enable_ensemble = enable_ensemble

        self.leaderboard: List[Dict] = []
        self.trained_models: Dict[str, Any] = {}
        self.best_model_name: Optional[str] = None

    def train_all(self, X_train, y_train, X_val, y_val,
                  experiment_name="automl",
                  model_filter: list = None) -> List[Dict]:
        if HAS_MLFLOW:
            try: mlflow.set_experiment(experiment_name)
            except: pass

        is_clf = self.problem_type == "classification"
        fast = self.n_trials <= 10
        registry = _build_registry(is_clf, fast=fast)

        # ── LLM-selected model filter ────────────────────────────────────────
        # Map common LLM names → registry keys (LLM may say "neural_net", registry has "mlp")
        NAME_MAP = {
            "neural_net":          "mlp",
            "neural_network":      "mlp",
            "mlp":                 "mlp",
            "gradient_boosting":   "gradient_boosting",
            "logistic_regression": "logistic_regression" if is_clf else "ridge",
            "linear":              "logistic_regression" if is_clf else "ridge",
            "linear_regression":   "ridge",
            "lasso":               "ridge",
            "catboost":            "catboost",
            "xgboost":             "xgboost",
            "lightgbm":            "lightgbm",
            "lgbm":                "lightgbm",
            "random_forest":       "random_forest",
            "extra_trees":         "extra_trees",
            "adaboost":            "adaboost",
            "svm":                 "svm",
            "knn":                 "knn",
            "ridge":               "logistic_regression" if is_clf else "ridge",
        }
        if model_filter:
            normalised = [NAME_MAP.get(m, m) for m in model_filter]
            ordered    = [k for k in normalised if k in registry]

            logger.info("=" * 60)
            logger.info(f"🤖 LLM MODEL FILTER ACTIVE")
            logger.info(f"   Requested : {model_filter}")
            logger.info(f"   Normalised: {normalised}")
            logger.info(f"   Available : {list(registry.keys())}")
            logger.info(f"   Will train: {ordered if ordered else '⚠️ none matched — training all'}")
            logger.info("=" * 60)

            if ordered:
                registry = {k: registry[k] for k in ordered}
            else:
                logger.warning(
                    "⚠️ LLM model filter produced no matches — "
                    "training full registry. "
                    f"(requested={model_filter}, registry keys={list(registry.keys())})"
                )
        else:
            logger.info(f"No LLM filter — training full registry: {list(registry.keys())}")

        for name, (model_cls, space_fn, fixed_kw) in registry.items():
            logger.info(f"Training [{name}] …")
            try:
                result = self._train_one(name, model_cls, space_fn, fixed_kw,
                                         X_train, y_train, X_val, y_val)
                self.leaderboard.append(result)
                self.trained_models[name] = result["fitted_model"]
            except Exception as e:
                logger.error(f"[{name}] failed: {e}")

        # Ensembles
        if self.enable_ensemble and len(self.trained_models) >= 3:
            for fn in (self._add_voting, self._add_stacking):
                try: fn(X_train, y_train, X_val, y_val, is_clf)
                except Exception as e: logger.warning(f"Ensemble failed: {e}")

        primary = "accuracy" if is_clf else "r2"
        self.leaderboard.sort(
            key=lambda x: x.get("metrics", {}).get(primary, -999), reverse=True
        )
        if self.leaderboard:
            self.best_model_name = self.leaderboard[0]["model_name"]
            logger.info(f"Best: {self.best_model_name} "
                        f"({primary}={self.leaderboard[0]['metrics'].get(primary,'?')})")
        return self.leaderboard

    def get_best_model(self):
        if not self.best_model_name:
            raise RuntimeError("Call train_all() first.")
        return self.best_model_name, self.trained_models[self.best_model_name]

    def _train_one(self, name, model_cls, space_fn, fixed_kw,
                   X_train, y_train, X_val, y_val) -> Dict:
        t0 = time.time()
        is_clf = self.problem_type == "classification"
        cv_scoring = "accuracy" if is_clf else "r2"
        cv = (StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
              if is_clf
              else KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state))

        best_params = dict(fixed_kw)
        cv_score = None

        if space_fn is not None and self.n_trials > 0 and HAS_OPTUNA:
            def objective(trial):
                if time.time() - t0 > self.max_time_per_model:
                    raise optuna.TrialPruned()
                params = {**fixed_kw, **space_fn(trial, is_clf)}
                m = model_cls(**params)
                scores = cross_val_score(m, X_train, y_train, cv=cv,
                                          scoring=cv_scoring, n_jobs=-1, error_score=-999)
                return float(scores.mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials,
                           timeout=self.max_time_per_model, show_progress_bar=False)
            best_params = {**fixed_kw, **study.best_params}
            cv_score = round(study.best_value, 4)

        model = model_cls(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = None
        if is_clf:
            try: y_prob = model.predict_proba(X_val)
            except: pass

        metrics = compute_metrics(self.problem_type, y_val, y_pred, y_prob)
        if cv_score is not None:
            metrics["cv_score"] = cv_score
        duration = round(time.time() - t0, 2)

        if HAS_MLFLOW:
            try:
                with mlflow.start_run(run_name=name):
                    mlflow.log_params({k: str(v)[:250] for k, v in best_params.items()})
                    mlflow.log_metrics(metrics)
            except: pass

        logger.info(f"  [{name}] {metrics}  {duration}s")
        return {"model_name": name, "metrics": metrics, "best_params": best_params,
                "training_time": duration, "fitted_model": model, "model_type": "base"}

    def _add_voting(self, X_train, y_train, X_val, y_val, is_clf):
        t0 = time.time()
        primary = "accuracy" if is_clf else "r2"
        top = sorted(self.leaderboard,
                     key=lambda x: x.get("metrics", {}).get(primary, -999), reverse=True)[:5]
        estimators = [(r["model_name"], r["fitted_model"]) for r in top]

        if is_clf:
            soft = all(hasattr(m, "predict_proba") for _, m in estimators)
            ens = VotingClassifier(estimators=estimators, voting="soft" if soft else "hard", n_jobs=-1)
        else:
            ens = VotingRegressor(estimators=estimators, n_jobs=-1)

        ens.fit(X_train, y_train)
        y_pred = ens.predict(X_val)
        y_prob = None
        if is_clf:
            try: y_prob = ens.predict_proba(X_val)
            except: pass

        metrics = compute_metrics(self.problem_type, y_val, y_pred, y_prob)
        duration = round(time.time() - t0, 2)
        result = {"model_name": "voting_ensemble", "metrics": metrics,
                  "best_params": {"n_models": len(estimators)},
                  "training_time": duration, "fitted_model": ens, "model_type": "ensemble"}
        self.leaderboard.append(result)
        self.trained_models["voting_ensemble"] = ens
        logger.info(f"  [voting_ensemble] {metrics}")

    def _add_stacking(self, X_train, y_train, X_val, y_val, is_clf):
        t0 = time.time()
        primary = "accuracy" if is_clf else "r2"
        top = sorted(self.leaderboard,
                     key=lambda x: x.get("metrics", {}).get(primary, -999), reverse=True)[:4]
        base = [(r["model_name"], r["fitted_model"]) for r in top
                if r["model_name"] not in ("voting_ensemble", "stacking_ensemble")][:4]
        if len(base) < 2:
            return

        final = LogisticRegression(max_iter=500, random_state=42) if is_clf else Ridge(random_state=42)
        if is_clf:
            ens = StackingClassifier(estimators=base, final_estimator=final, cv=3, n_jobs=-1, passthrough=True)
        else:
            ens = StackingRegressor(estimators=base, final_estimator=final, cv=3, n_jobs=-1, passthrough=True)

        ens.fit(X_train, y_train)
        y_pred = ens.predict(X_val)
        y_prob = None
        if is_clf:
            try: y_prob = ens.predict_proba(X_val)
            except: pass

        metrics = compute_metrics(self.problem_type, y_val, y_pred, y_prob)
        duration = round(time.time() - t0, 2)
        result = {"model_name": "stacking_ensemble", "metrics": metrics,
                  "best_params": {"base_models": [n for n, _ in base]},
                  "training_time": duration, "fitted_model": ens, "model_type": "ensemble"}
        self.leaderboard.append(result)
        self.trained_models["stacking_ensemble"] = ens
        logger.info(f"  [stacking_ensemble] {metrics}")
