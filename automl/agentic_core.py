"""
AutoML Agentic Core — Production-Level Agentic AI
=================================================
Architecture:
  - ReAct Agent Loop: Reason → Act → Observe → Repeat until satisfied
  - Multi-Agent: Orchestrator + DataAgent + ModelAgent + FeatureAgent
  - Real tool execution that changes experiment state each iteration
  - Autonomous feature engineering with validation
  - Self-correcting: detects problems, changes strategy mid-run
  - Persistent memory: learns across iterations what works
  - Full thought streaming to UI in real-time
"""
from __future__ import annotations
import json, time, re, hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from openai import OpenAI as _OpenAI
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


# ─────────────────────────────────────────────────────────────────────────────
# Agent Memory — persists knowledge within a run AND across iterations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentMemory:
    """What the agent knows and has tried so far."""
    iterations:        List[Dict] = field(default_factory=list)
    best_score:        float      = -999.0
    best_model:        str        = ""
    best_config:       Dict       = field(default_factory=dict)
    tried_models:      List[str]  = field(default_factory=list)
    tried_features:    List[str]  = field(default_factory=list)
    feature_impacts:   Dict[str, float] = field(default_factory=dict)  # feat → score delta
    error_patterns:    List[str]  = field(default_factory=list)
    hypotheses_tested: List[str]  = field(default_factory=list)
    agent_thoughts:    List[str]  = field(default_factory=list)
    class_imbalance:   bool       = False
    data_issues:       List[str]  = field(default_factory=list)

    def add_iteration(self, iteration: int, action: str,
                      result: Dict, thought: str = ""):
        self.iterations.append({
            "iter": iteration, "action": action,
            "result": result, "thought": thought,
            "ts": time.strftime("%H:%M:%S"),
        })
        score = result.get("best_score", -999)
        if score > self.best_score:
            self.best_score  = score
            self.best_model  = result.get("best_model", "")
            self.best_config = result.get("config", {})

    def summary_for_llm(self) -> str:
        """Compact summary to feed into next LLM call."""
        lines = [
            f"Iterations so far: {len(self.iterations)}",
            f"Best score: {self.best_score:.4f} ({self.best_model})",
            f"Models tried: {self.tried_models}",
            f"Features engineered: {self.tried_features}",
            f"Issues found: {self.data_issues}",
        ]
        if self.iterations:
            last = self.iterations[-1]
            lines.append(
                f"Last action: {last['action']} → "
                f"score={last['result'].get('best_score','?'):.4f}"
                if isinstance(last['result'].get('best_score'), float)
                else f"Last action: {last['action']}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Real AutoML Tools — each tool EXECUTES actual ML code, not just plans
# ─────────────────────────────────────────────────────────────────────────────

class AutoMLToolkit:
    """
    Executable tools the agent can call.
    Each tool performs real computation and returns observable results.
    """

    def __init__(self, df: pd.DataFrame, target_col: str,
                 problem_type: str, push_log: Optional[Callable] = None):
        self.df_orig      = df.copy()
        self.df_working   = df.copy()
        self.target_col   = target_col
        self.problem_type = problem_type
        self.push_log     = push_log
        self._log_fn      = push_log or (lambda m: None)
        self._results_cache: Dict[str, Dict] = {}

    def _log(self, msg: str):
        self._log_fn(f"🔧 {msg}")

    # ── Tool 1: Quick Benchmark ───────────────────────────────────────────────

    def quick_benchmark(self, models: List[str], n_trials: int = 10,
                        class_weight: Optional[str] = None) -> Dict:
        """
        Train a specific set of models with given params.
        Returns scored leaderboard. This is the primary training tool.
        """
        self._log(f"Benchmarking {models} × {n_trials} trials…")
        from automl.trainer import TrainingEngine, _build_registry
        from automl.preprocessor import PreprocessingEngine
        from sklearn.model_selection import train_test_split

        df    = self.df_working.copy().dropna(subset=[self.target_col])
        y     = df[self.target_col]
        X_raw = df.drop(columns=[self.target_col])

        # Drop non-feature columns
        drop_cols = [c for c in X_raw.columns
                     if X_raw[c].dtype == object and X_raw[c].nunique() == len(X_raw)]
        if drop_cols:
            X_raw = X_raw.drop(columns=drop_cols)

        is_clf = self.problem_type == "classification"
        stratify = y if is_clf else None
        try:
            X_tr, X_v, y_tr, y_v = train_test_split(
                X_raw, y, test_size=0.2, random_state=42, stratify=stratify
            )
        except Exception:
            X_tr, X_v, y_tr, y_v = train_test_split(
                X_raw, y, test_size=0.2, random_state=42
            )

        # Encode target
        from automl.preprocessor import TargetLabelEncoder
        le = None
        if is_clf:
            le = TargetLabelEncoder()
            y_tr_enc = le.fit_transform(y_tr)
            y_v_enc  = le.transform(y_v)
        else:
            y_tr_enc = y_tr.values
            y_v_enc  = y_v.values

        # Preprocess
        try:
            pp_engine = PreprocessingEngine(
                df=pd.concat([X_tr, X_v]),
                target_col=self.target_col,
                problem_type=self.problem_type,
                feature_selection="mutual_info",
                handle_outliers=True,
            )
            best_pp = pp_engine.build_best_pipeline(
                X_tr, pd.Series(y_tr_enc), X_v, pd.Series(y_v_enc)
            )
            X_tr_t = best_pp.transform(X_tr[pp_engine.all_feature_cols])
            X_v_t  = best_pp.transform(X_v[pp_engine.all_feature_cols])
        except Exception as e:
            self._log(f"Preprocessing failed: {e}. Using raw.")
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline as SkPipe
            num_cols = X_tr.select_dtypes(include=np.number).columns.tolist()
            X_tr_t = SimpleImputer().fit_transform(X_tr[num_cols])
            X_v_t  = SimpleImputer().transform(X_v[num_cols])

        # Apply class weight if requested
        fixed_kw_override = {}
        if class_weight and is_clf:
            fixed_kw_override["class_weight"] = class_weight

        # Build model registry filtered to requested models
        NAME_MAP = {
            "neural_net": "mlp", "lgbm": "lightgbm",
            "linear": "logistic_regression" if is_clf else "ridge",
            "linear_regression": "ridge", "ridge": "ridge",
        }
        registry = _build_registry(is_clf, fast=(n_trials <= 12))
        req_norm = [NAME_MAP.get(m, m) for m in models]
        filtered = {k: v for k, v in registry.items() if k in req_norm}
        if not filtered:
            filtered = registry  # fallback

        engine = TrainingEngine(
            problem_type=self.problem_type,
            n_trials=n_trials, cv_folds=3,
            random_state=42, enable_dl=False, enable_ensemble=False,
        )
        lb = engine.train_all(X_tr_t, y_tr_enc, X_v_t, y_v_enc,
                              model_filter=req_norm)
        best_entry = lb[0] if lb else {}
        metric_key = "accuracy" if is_clf else "r2"
        score = best_entry.get("metrics", {}).get(metric_key, -999)

        result = {
            "best_model": best_entry.get("model_name", ""),
            "best_score": round(float(score), 4),
            "metric":     metric_key,
            "leaderboard": [
                {"model": r["model_name"],
                 "score": round(float(r.get("metrics", {}).get(metric_key, -999)), 4)}
                for r in lb[:6]
            ],
            "config": {"models": models, "n_trials": n_trials,
                       "class_weight": class_weight},
            "n_models_trained": len(lb),
        }
        self._log(f"Best: {result['best_model']} {metric_key}={result['best_score']:.4f}")
        return result

    # ── Tool 2: Data Quality Investigation ───────────────────────────────────

    def investigate_data(self) -> Dict:
        """
        Deep data quality check — finds real problems and quantifies them.
        Returns actionable findings.
        """
        self._log("Investigating data quality…")
        df     = self.df_working
        target = self.target_col
        issues = []
        stats  = {}

        # Missing values
        miss = df.isnull().mean()
        heavy_miss = miss[miss > 0.3].to_dict()
        if heavy_miss:
            issues.append(f"Heavy missing: {list(heavy_miss.keys())} (>30%)")
        stats["missing_pct_avg"] = round(float(miss.mean() * 100), 2)

        # Class imbalance
        if self.problem_type == "classification":
            vc = df[target].value_counts()
            ratio = float(vc.min() / vc.max()) if vc.max() > 0 else 1.0
            stats["class_imbalance_ratio"] = round(ratio, 3)
            stats["class_counts"] = {str(k): int(v) for k, v in vc.items()}
            if ratio < 0.3:
                issues.append(f"Severe class imbalance: ratio={ratio:.2f} — consider balanced weights")

        # High cardinality categoricals
        cat_cols = df.select_dtypes(include="object").columns
        high_card = {c: int(df[c].nunique()) for c in cat_cols
                     if df[c].nunique() > 50 and c != target}
        if high_card:
            issues.append(f"High-cardinality cols: {high_card} — CatBoost handles these best")

        # Skewness
        num_cols = df.select_dtypes(include=np.number).columns
        skewed = {c: round(float(df[c].skew()), 2)
                  for c in num_cols if c != target and abs(float(df[c].skew())) > 2}
        if skewed:
            issues.append(f"Highly skewed features: {skewed} — power transform recommended")
        stats["skewed_features"] = skewed

        # Duplicates
        dup_pct = round(float(df.duplicated().mean() * 100), 2)
        if dup_pct > 5:
            issues.append(f"Duplicates: {dup_pct}% of rows")
        stats["duplicate_pct"] = dup_pct

        # Target leakage candidates (near-perfect correlation)
        if self.problem_type == "regression":
            num_df = df[num_cols].dropna()
            if target in num_df.columns:
                corrs = num_df.drop(columns=[target]).corrwith(num_df[target]).abs()
                leakage = corrs[corrs > 0.98].to_dict()
                if leakage:
                    issues.append(f"Possible data leakage: {list(leakage.keys())} (corr>0.98)")
                stats["top_correlations"] = {
                    k: round(v, 3) for k, v in
                    corrs.sort_values(ascending=False).head(5).items()
                }

        result = {
            "issues": issues,
            "n_issues": len(issues),
            "stats": stats,
            "recommendation": (
                "Address class imbalance first" if any("imbalance" in i for i in issues)
                else "Use CatBoost for high cardinality" if any("cardinality" in i for i in issues)
                else "Apply power transform for skewed features" if skewed
                else "Data looks clean"
            )
        }
        self._log(f"Found {len(issues)} issues: {issues[:2]}")
        return result

    # ── Tool 3: Feature Engineering ───────────────────────────────────────────

    def engineer_features(self, strategy: str,
                          top_features: Optional[List[str]] = None) -> Dict:
        """
        Actually creates new features in df_working.
        Strategies: 'interactions', 'ratios', 'polynomial', 'target_stats', 'datetime'
        """
        self._log(f"Engineering features: strategy={strategy}…")
        df     = self.df_working.copy()
        target = self.target_col
        num_cols = [c for c in df.select_dtypes(include=np.number).columns
                    if c != target]

        if top_features:
            num_cols = [c for c in top_features if c in num_cols] or num_cols

        new_features: List[str] = []
        n_before = len(df.columns)

        if strategy == "interactions" and len(num_cols) >= 2:
            # Pairwise interactions of top 4 numeric features
            for i, a in enumerate(num_cols[:4]):
                for b in num_cols[i+1:4]:
                    col_name = f"{a}_x_{b}"
                    if col_name not in df.columns:
                        df[col_name] = df[a] * df[b]
                        new_features.append(col_name)

        elif strategy == "ratios" and len(num_cols) >= 2:
            for i, a in enumerate(num_cols[:4]):
                for b in num_cols[i+1:4]:
                    col_name = f"{a}_div_{b}"
                    if col_name not in df.columns:
                        df[col_name] = df[a] / (df[b].replace(0, np.nan) + 1e-8)
                        new_features.append(col_name)

        elif strategy == "polynomial" and num_cols:
            for c in num_cols[:5]:
                sq_name = f"{c}_sq"
                if sq_name not in df.columns:
                    df[sq_name] = df[c] ** 2
                    new_features.append(sq_name)

        elif strategy == "aggregates" and num_cols:
            # Row-level statistics across numeric features
            sub = df[num_cols[:10]].copy()
            df["_feat_mean"]  = sub.mean(axis=1)
            df["_feat_std"]   = sub.std(axis=1)
            df["_feat_max"]   = sub.max(axis=1)
            df["_feat_min"]   = sub.min(axis=1)
            new_features.extend(["_feat_mean","_feat_std","_feat_max","_feat_min"])

        elif strategy == "log_transform" and num_cols:
            for c in num_cols[:6]:
                if (df[c] > 0).all():
                    col_name = f"{c}_log"
                    if col_name not in df.columns:
                        df[col_name] = np.log1p(df[c])
                        new_features.append(col_name)

        # Commit engineered features to working df
        if new_features:
            self.df_working = df
            self._log(f"Created {len(new_features)} features: {new_features[:4]}")

        return {
            "strategy":     strategy,
            "new_features": new_features,
            "n_added":      len(new_features),
            "total_cols":   len(self.df_working.columns),
            "success":      len(new_features) > 0,
        }

    # ── Tool 4: Error Analysis ────────────────────────────────────────────────

    def analyze_errors(self, model_name: str, leaderboard: List[Dict]) -> Dict:
        """
        Analyze where the current best model fails.
        Returns actionable insights about hard cases.
        """
        self._log(f"Analysing model errors…")
        df     = self.df_working.dropna(subset=[self.target_col])
        target = self.target_col
        is_clf = self.problem_type == "classification"

        # Basic error analysis from leaderboard spread
        if not leaderboard:
            return {"error": "No leaderboard available"}

        scores  = [e["score"] for e in leaderboard if isinstance(e.get("score"), float)]
        top     = leaderboard[0]
        spread  = round(max(scores) - min(scores), 4) if len(scores) > 1 else 0

        findings = []
        suggestions = []

        if is_clf:
            vc = df[target].value_counts()
            if float(vc.min() / vc.max()) < 0.3:
                findings.append("Class imbalance is likely causing poor recall on minority class")
                suggestions.append("Use class_weight='balanced' or SMOTE")
            if top["score"] < 0.75:
                findings.append(f"Low accuracy {top['score']:.3f} — model may be underfitting")
                suggestions.append("Increase n_trials or try more powerful models (XGBoost/LightGBM)")
            elif top["score"] > 0.98:
                findings.append("Suspiciously high accuracy — check for data leakage")
                suggestions.append("Remove any columns derived from target")
        else:
            if top["score"] < 0.6:
                findings.append(f"Low R² {top['score']:.3f} — poor fit")
                suggestions.append("Try feature engineering or non-linear models")
            elif top["score"] > 0.99:
                findings.append("Near-perfect R² — likely data leakage")

        if spread > 0.1:
            findings.append(f"Large model spread ({spread:.3f}) — some models much better than others")
            suggestions.append("Focus HPO budget on top model families only")

        # Check if tried models could be improved
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) > 10:
            suggestions.append("High-dimensional: try feature selection to reduce noise")

        return {
            "best_score":   top["score"],
            "model_spread": spread,
            "findings":     findings,
            "suggestions":  suggestions,
            "n_models":     len(leaderboard),
        }

    # ── Tool 5: Targeted HPO ──────────────────────────────────────────────────

    def deep_tune(self, model: str, n_trials: int = 40,
                  class_weight: Optional[str] = None) -> Dict:
        """Deep hyperparameter optimisation on a single winning model."""
        self._log(f"Deep tuning {model} × {n_trials} trials…")
        return self.quick_benchmark(
            models=[model], n_trials=n_trials, class_weight=class_weight
        )

    # ── Tool 6: Cross-validate ────────────────────────────────────────────────

    def cross_validate(self, model: str, cv_folds: int = 5) -> Dict:
        """Robust CV estimate to check if benchmark score is real."""
        self._log(f"Cross-validating {model} × {cv_folds} folds…")
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from automl.trainer import _build_registry
        from automl.preprocessor import PreprocessingEngine
        from sklearn.impute import SimpleImputer

        df    = self.df_working.dropna(subset=[self.target_col])
        is_clf = self.problem_type == "classification"
        y     = df[self.target_col]
        X     = df.drop(columns=[self.target_col])
        X_num = X.select_dtypes(include=np.number)
        X_imp = SimpleImputer().fit_transform(X_num)

        if is_clf:
            from automl.preprocessor import TargetLabelEncoder
            le = TargetLabelEncoder()
            y_enc = le.fit_transform(y)
        else:
            y_enc = y.values

        registry = _build_registry(is_clf, fast=False)
        NAME_MAP = {"neural_net": "mlp", "lgbm": "lightgbm"}
        key = NAME_MAP.get(model, model)

        if key not in registry:
            return {"error": f"{model} not available", "cv_score": -999}

        model_cls, space_fn, fixed_kw = registry[key]
        clf = model_cls(**{k: v for k, v in fixed_kw.items()
                          if k not in ("eval_metric", "verbosity")})

        cv = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
              if is_clf
              else KFold(n_splits=cv_folds, shuffle=True, random_state=42))
        scoring = "accuracy" if is_clf else "r2"

        try:
            scores = cross_val_score(clf, X_imp, y_enc, cv=cv,
                                     scoring=scoring, n_jobs=-1)
            result = {
                "model": model,
                "cv_mean":  round(float(scores.mean()), 4),
                "cv_std":   round(float(scores.std()), 4),
                "cv_scores": [round(float(s), 4) for s in scores],
                "reliable": float(scores.std()) < 0.05,
            }
            self._log(f"CV {model}: {result['cv_mean']:.4f} ±{result['cv_std']:.4f}")
            return result
        except Exception as e:
            return {"error": str(e), "cv_score": -999}


# ─────────────────────────────────────────────────────────────────────────────
# ReAct Agent — the real agentic loop
# ─────────────────────────────────────────────────────────────────────────────

REACT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "investigate_data",
            "description": "Deeply investigate data quality, find issues, imbalance, leakage. Call this first.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "quick_benchmark",
            "description": "Train and evaluate specific models. Returns scored leaderboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {"type": "string",
                                  "enum": ["xgboost","lightgbm","catboost","random_forest",
                                           "extra_trees","logistic_regression","ridge","svm","knn",
                                           "neural_net","adaboost","gradient_boosting"]},
                        "description": "Models to benchmark"
                    },
                    "n_trials": {"type": "integer", "description": "HPO trials (10-50)"},
                    "class_weight": {"type": "string", "enum": ["balanced", "none"],
                                     "description": "Use 'balanced' for imbalanced classes"}
                },
                "required": ["models", "n_trials"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_errors",
            "description": "Analyze where current models fail and get targeted improvement suggestions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                },
                "required": ["model_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "engineer_features",
            "description": "Create new features to improve model performance. Actually modifies the dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["interactions","ratios","polynomial","aggregates","log_transform"],
                        "description": "interactions: multiply feature pairs. ratios: divide pairs. polynomial: squared features. aggregates: row-wise stats. log_transform: log of positive features."
                    },
                    "top_features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: limit engineering to these features"
                    }
                },
                "required": ["strategy"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deep_tune",
            "description": "Run intensive HPO on a single winning model with many more trials.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model to deep-tune"},
                    "n_trials": {"type": "integer", "description": "Trials (30-80)"},
                    "class_weight": {"type": "string", "enum": ["balanced", "none"]}
                },
                "required": ["model", "n_trials"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cross_validate",
            "description": "Run k-fold CV to get a reliable score estimate for the best model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "cv_folds": {"type": "integer", "description": "Number of folds (3-10)"}
                },
                "required": ["model"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Declare the best model and finish the agent loop. Call when satisfied.",
            "parameters": {
                "type": "object",
                "properties": {
                    "best_model":   {"type": "string"},
                    "final_score":  {"type": "number"},
                    "reasoning":    {"type": "string", "description": "Why this is the best result"},
                    "improvements": {"type": "array", "items": {"type": "string"},
                                     "description": "What the agent did to improve performance"},
                    "exec_summary": {"type": "string", "description": "Plain English summary for the user"}
                },
                "required": ["best_model", "final_score", "reasoning", "improvements", "exec_summary"]
            }
        }
    },
]

_SYS_REACT = """You are an expert AutoML agent running a ReAct (Reason-Act-Observe) loop.
Your goal: find the best ML model for this dataset through iterative experimentation.

Rules:
- Always start by calling investigate_data to understand the data
- Then run quick_benchmark with 2-4 promising models
- Observe results and reason about what to try next
- Try at least 2-3 different strategies before finishing
- Use engineer_features if performance is plateaued
- Use deep_tune on the winner once you've identified it
- Use cross_validate before finishing to confirm score is real
- Call finish() when satisfied (score not improving or max iterations reached)
- Be specific in your reasoning: cite actual numbers from observations
- Never call the same tool with the same params twice
"""


class ReactAgent:
    """
    Production ReAct agent that iterates until satisfied.
    Reason → Act → Observe → Reason → Act → ... → Finish
    """

    MAX_ITERATIONS = 8

    def __init__(self, api_key: Optional[str],
                 user_model: str = "llama-3.1-8b-instant"):
        self.api_key    = api_key
        self.user_model = user_model
        self.client: Optional[Any] = None

        if HAS_LLM and api_key:
            try:
                self.client = _OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
            except Exception as e:
                pass

    def is_available(self) -> bool:
        return self.client is not None

    def _call(self, messages: List[Dict],
              force_tool: Optional[str] = None) -> Optional[Any]:
        if not self.client:
            return None
        # Use 70B for agent reasoning when available — it's much better at tool chaining
        model = (self.user_model
                 if "70b" in self.user_model or "mixtral" in self.user_model
                 else "llama-3.3-70b-versatile")
        try:
            tool_choice: Any = "auto"
            if force_tool:
                tool_choice = {"type": "function", "function": {"name": force_tool}}
            resp = self.client.chat.completions.create(
                model=model,
                max_tokens=800,
                messages=messages,
                tools=REACT_TOOLS,
                tool_choice=tool_choice,
                temperature=0.2,
            )
            return resp
        except Exception as e:
            # Fallback to 8b if 70b fails (rate limit etc)
            try:
                resp = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    max_tokens=600,
                    messages=messages,
                    tools=REACT_TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                )
                return resp
            except Exception:
                return None

    def _parse_tool_call(self, resp: Any) -> Tuple[Optional[str], Optional[Dict]]:
        if not resp: return None, None
        try:
            msg = resp.choices[0].message
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                try:
                    return tc.function.name, json.loads(tc.function.arguments)
                except Exception:
                    return tc.function.name, {}
            # Try extracting from text
            text = msg.content or ""
            m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if m:
                try: return None, json.loads(m.group())
                except: pass
        except Exception:
            pass
        return None, None

    def _get_thought(self, resp: Any) -> str:
        if not resp: return ""
        try:
            msg = resp.choices[0].message
            return (msg.content or "").strip()
        except: return ""

    def run(self, toolkit: AutoMLToolkit, profile: Dict,
            push_log: Optional[Callable] = None,
            push_thought: Optional[Callable] = None) -> Tuple[Dict, ExperimentMemory]:
        """
        Run the full ReAct loop.
        Returns (final_result, memory).
        """
        def _log(m):
            if push_log: push_log(m)

        def _thought(t):
            if push_thought: push_thought(t)

        memory = ExperimentMemory()
        messages: List[Dict] = [
            {"role": "system", "content": _SYS_REACT},
            {"role": "user",   "content":
             f"Dataset profile:\n{json.dumps(profile, separators=(',',':'))}\n\n"
             f"Task: {profile['task']} | Target: {profile['target']} | "
             f"Rows: {profile['rows']} | Features: {profile['num_feats']} numeric, "
             f"{profile['cat_feats']} categorical\n\n"
             f"Begin the ReAct loop. Start by investigating the data."}
        ]

        final_result: Dict = {}
        current_leaderboard: List[Dict] = []

        for iteration in range(self.MAX_ITERATIONS):
            _log(f"🔄 Agent iteration {iteration+1}/{self.MAX_ITERATIONS}")

            if not self.client:
                # No LLM — run smart rule-based loop
                final_result = self._rule_based_loop(toolkit, profile, memory, _log)
                break

            # Get agent's next action
            resp = self._call(messages)
            tool_name, tool_args = self._parse_tool_call(resp)
            thought = self._get_thought(resp)

            if thought:
                memory.agent_thoughts.append(f"[iter {iteration+1}] {thought}")
                _thought(f"💭 {thought}")
                _log(f"💭 Agent: {thought[:120]}")

            if not tool_name:
                _log("⚠️ Agent gave no tool call — ending loop")
                break

            _log(f"⚡ Action: {tool_name}({json.dumps(tool_args, separators=(',',':'))})")

            # Execute the tool
            observation: Dict = {}
            try:
                if tool_name == "investigate_data":
                    observation = toolkit.investigate_data()
                    memory.data_issues = observation.get("issues", [])

                elif tool_name == "quick_benchmark":
                    models      = tool_args.get("models", ["xgboost","lightgbm"])
                    n_trials    = min(int(tool_args.get("n_trials", 15)), 50)
                    cw          = tool_args.get("class_weight")
                    cw          = cw if cw and cw != "none" else None
                    observation = toolkit.quick_benchmark(models, n_trials, cw)
                    memory.tried_models.extend(models)
                    lb = observation.get("leaderboard", [])
                    if lb:
                        current_leaderboard = lb
                    memory.add_iteration(iteration, tool_name, observation, thought)

                elif tool_name == "analyze_errors":
                    model_name  = tool_args.get("model_name", "")
                    observation = toolkit.analyze_errors(model_name, current_leaderboard)

                elif tool_name == "engineer_features":
                    strategy     = tool_args.get("strategy", "interactions")
                    top_feats    = tool_args.get("top_features")
                    observation  = toolkit.engineer_features(strategy, top_feats)
                    if observation.get("success"):
                        memory.tried_features.append(strategy)

                elif tool_name == "deep_tune":
                    model    = tool_args.get("model", "")
                    n_trials = min(int(tool_args.get("n_trials", 40)), 80)
                    cw       = tool_args.get("class_weight")
                    cw       = cw if cw and cw != "none" else None
                    observation = toolkit.deep_tune(model, n_trials, cw)
                    lb = observation.get("leaderboard", [])
                    if lb: current_leaderboard = lb
                    memory.add_iteration(iteration, tool_name, observation, thought)

                elif tool_name == "cross_validate":
                    model    = tool_args.get("model", memory.best_model)
                    cv_folds = int(tool_args.get("cv_folds", 5))
                    observation = toolkit.cross_validate(model, cv_folds)

                elif tool_name == "finish":
                    final_result = {
                        "best_model":   tool_args.get("best_model", memory.best_model),
                        "final_score":  tool_args.get("final_score", memory.best_score),
                        "reasoning":    tool_args.get("reasoning", ""),
                        "improvements": tool_args.get("improvements", []),
                        "exec_summary": tool_args.get("exec_summary", ""),
                        "leaderboard":  current_leaderboard,
                        "iterations":   len(memory.iterations),
                        "agent_thoughts": memory.agent_thoughts,
                    }
                    _log(f"✅ Agent finished: {final_result['best_model']} "
                         f"score={final_result['final_score']:.4f}")
                    break

            except Exception as e:
                observation = {"error": str(e)[:200]}
                _log(f"⚠️ Tool error: {e}")

            obs_str = json.dumps(observation, separators=(",",":"), default=str)
            _log(f"👁 Observation: {obs_str[:200]}")

            # Add to message history for next iteration
            messages.append({"role": "assistant", "content": resp.choices[0].message.content or "",
                              "tool_calls": [tc.model_dump() for tc in (resp.choices[0].message.tool_calls or [])]})
            messages.append({"role": "tool",
                             "tool_call_id": (resp.choices[0].message.tool_calls or [{}])[0].id
                             if resp.choices[0].message.tool_calls else "call_0",
                             "content": obs_str})
            messages.append({"role": "user",
                             "content": f"Observation: {obs_str[:600]}\n\n"
                             f"Memory so far:\n{memory.summary_for_llm()}\n\n"
                             f"Continue the ReAct loop. "
                             f"{'Call finish() if satisfied.' if iteration >= 3 else 'Try to improve further.'}"})

        # If agent didn't call finish, synthesise final result from memory
        if not final_result:
            final_result = {
                "best_model":   memory.best_model or "unknown",
                "final_score":  memory.best_score,
                "reasoning":    "Agent loop completed without explicit finish",
                "improvements": [f"Tried: {', '.join(set(memory.tried_models))}"],
                "exec_summary": f"Best model: {memory.best_model} (score={memory.best_score:.4f})",
                "leaderboard":  current_leaderboard,
                "iterations":   len(memory.iterations),
                "agent_thoughts": memory.agent_thoughts,
            }

        return final_result, memory

    def _rule_based_loop(self, toolkit: AutoMLToolkit, profile: Dict,
                         memory: ExperimentMemory, _log: Callable) -> Dict:
        """Smart rule-based multi-iteration loop when no LLM available."""
        n    = profile["rows"]
        clf  = profile["task"] == "classification"
        miss = profile["missing_pct"]

        # Iteration 1: Data investigation
        _log("🔄 [1/3] Investigating data…")
        di = toolkit.investigate_data()
        memory.data_issues = di.get("issues", [])
        imbalanced = any("imbalance" in i for i in di.get("issues", []))

        # Iteration 2: First model pass
        _log("🔄 [2/3] First model benchmark…")
        if n < 1000:
            models1 = ["random_forest","logistic_regression","svm"] if clf else ["ridge","random_forest","svm"]
        else:
            models1 = ["lightgbm","xgboost","random_forest"] if clf else ["lightgbm","xgboost","random_forest"]
        cw = "balanced" if imbalanced and clf else None
        r1 = toolkit.quick_benchmark(models1, n_trials=15, class_weight=cw)
        memory.add_iteration(0, "quick_benchmark", r1)
        memory.tried_models.extend(models1)

        # Iteration 3: Feature engineering if score is mediocre
        score1 = r1.get("best_score", 0)
        if score1 < 0.85:
            _log("🔄 [3/3] Score below 0.85 — engineering features…")
            if miss > 5: toolkit.engineer_features("aggregates")
            else: toolkit.engineer_features("interactions")
            # Re-benchmark winner with new features
            winner = r1.get("best_model", models1[0])
            r2 = toolkit.deep_tune(winner, n_trials=25, class_weight=cw)
            memory.add_iteration(1, "deep_tune_with_features", r2)
            best_result = r2 if r2.get("best_score",0) > score1 else r1
        else:
            best_result = r1

        return {
            "best_model":   best_result.get("best_model",""),
            "final_score":  best_result.get("best_score",0),
            "reasoning":    "Rule-based 3-iteration loop",
            "improvements": [f"Tried {models1}", "Feature engineering if needed"],
            "exec_summary": f"Best: {best_result.get('best_model','')} "
                            f"score={best_result.get('best_score',0):.4f}",
            "leaderboard":  best_result.get("leaderboard",[]),
            "iterations":   len(memory.iterations),
            "agent_thoughts": [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentAutoML:
    """
    Three specialised agents working together:
    - DataAgent:    investigates & fixes data quality
    - ModelAgent:   runs the ReAct experiment loop
    - InsightAgent: generates final narrative and recommendations
    """

    def __init__(self, api_key: Optional[str],
                 user_model: str = "llama-3.1-8b-instant"):
        self.api_key    = api_key
        self.user_model = user_model
        self.react      = ReactAgent(api_key, user_model)

    def run(self, file_path: str, target_col: str,
            push_log: Optional[Callable] = None,
            push_thought: Optional[Callable] = None) -> tuple:
        """
        Full multi-agent run. Returns (summary, agent_obj, memory).
        """
        from automl.agent import AutoMLAgent
        from automl.data_handler import load_dataset, detect_problem_type
        from automl.llm_agent import profile_dataset, LLMAgent, _format_ai_insights

        def _log(m):
            if push_log: push_log(f"🤖 {m}")

        def _think(t):
            if push_thought: push_thought(t)

        # ── Load & profile ────────────────────────────────────────────────────
        _log("Loading & profiling dataset…")
        df = load_dataset(file_path)
        problem_type = detect_problem_type(df, target_col, None)
        _log(f"Task: {problem_type} · {df.shape[0]:,} rows × {df.shape[1]} cols")
        profile = profile_dataset(df, target_col, problem_type)

        # ── DataAgent: pre-flight data checks ────────────────────────────────
        _log("━━━ DataAgent: pre-flight investigation ━━━")
        toolkit = AutoMLToolkit(df, target_col, problem_type, push_log=push_log)
        di      = toolkit.investigate_data()
        if di["issues"]:
            _log(f"🔍 DataAgent found {len(di['issues'])} issues:")
            for issue in di["issues"]:
                _log(f"   → {issue}")
        else:
            _log("✅ DataAgent: data looks clean")

        # ── ModelAgent: ReAct experiment loop ────────────────────────────────
        _log("━━━ ModelAgent: ReAct experiment loop ━━━")
        react_result, memory = self.react.run(
            toolkit, profile, push_log=push_log, push_thought=push_thought
        )

        # ── Build final AutoMLAgent from best config ──────────────────────────
        _log("━━━ Finalising best model pipeline ━━━")
        best_models_tried = list(set(memory.tried_models)) or ["lightgbm","xgboost","random_forest"]
        n_trials_final = 30

        # Run final AutoML with knowledge from agent
        final_agent = AutoMLAgent(
            target_col=target_col,
            problem_type=problem_type,
            feature_selection="mutual_info",
            handle_outliers=True,
            n_trials=n_trials_final,
        )
        # Pass the agent's learned plan
        final_agent._llm_plan = {
            "recommended_models": best_models_tried[:4],
            "n_trials": n_trials_final,
            "preprocessing_strategy": "robust" if di["stats"].get("missing_pct_avg",0) > 5 else "standard",
            "feature_selection": "mutual_info",
            "_source": "multi_agent",
        }

        # Use engineered df if features were added
        save_path = file_path
        if memory.tried_features and len(toolkit.df_working.columns) > len(df.columns):
            import tempfile, os
            tmp = tempfile.mktemp(suffix=".csv")
            toolkit.df_working.to_csv(tmp, index=False)
            save_path = tmp
            _log(f"Using enriched dataset (+{len(toolkit.df_working.columns)-len(df.columns)} features)")

        summary = final_agent.run(save_path)

        # ── InsightAgent: generate narrative ─────────────────────────────────
        _log("━━━ InsightAgent: generating narrative ━━━")
        llm_agent = LLMAgent(api_key=self.api_key, user_model=self.user_model)
        lb         = summary.get("leaderboard", [])
        top_feats  = summary.get("top_features", [])
        insights   = llm_agent.generate_insights(profile, lb, top_feats, push_log=push_log)

        # ── Merge everything into summary ─────────────────────────────────────
        elapsed = round(time.time() - getattr(self, "_start_ts", time.time()), 1)
        summary["agent_plan"]       = final_agent._llm_plan
        summary["agent_insights"]   = insights
        summary["agent_log"]        = memory.agent_thoughts
        summary["agent_tokens"]     = llm_agent.token_usage()
        summary["agent_iterations"] = memory.iterations
        summary["react_result"]     = react_result
        summary["data_issues"]      = di["issues"]
        summary["features_engineered"] = memory.tried_features
        summary["ai_insights"]      = _build_agent_insights(react_result, insights,
                                                             memory, summary)
        summary["dataset_profile"]  = profile

        _log(f"✅ Multi-agent run complete! "
             f"Iterations={len(memory.iterations)} "
             f"FeatEngineered={len(memory.tried_features)} "
             f"BestScore={memory.best_score:.4f}")

        return summary, final_agent, memory


def _build_agent_insights(react: Dict, insights: Dict,
                          memory: ExperimentMemory, summary: Dict) -> List[str]:
    out = []
    if insights.get("executive_summary"):
        out.append(f"📊 {insights['executive_summary']}")
    # Show agent's journey
    if memory.iterations:
        n = len(memory.iterations)
        out.append(f"🔄 Agent ran {n} experiment iteration{'s' if n>1 else ''} autonomously")
    if memory.tried_features:
        out.append(f"🔧 Feature engineering applied: {', '.join(memory.tried_features)}")
    if memory.data_issues:
        out.append(f"🔍 Data issues found & handled: {memory.data_issues[0]}")
    for f in (insights.get("key_findings") or [])[:2]:
        out.append(f"🔑 {f}")
    bm = summary.get("best_metrics", {})
    if bm:
        k, v = next(iter(bm.items()), (None, None))
        if k: out.append(f"🏆 {k}: {v:.4f} | {summary.get('best_model','—')}")
    return out
