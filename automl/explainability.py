"""AutoML Agent – Explainability Engine (SHAP + LIME, robust to all model types)"""
from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional
import numpy as np

try:
    from loguru import logger
except:
    import logging; logger = logging.getLogger("automl")

warnings.filterwarnings("ignore")


class ExplainabilityEngine:
    def __init__(self, model, X_train, X_val, feature_names, problem_type):
        self.model = model
        self.X_train = np.array(X_train)
        self.X_val   = np.array(X_val)
        self.feature_names = feature_names
        self.problem_type  = problem_type

    def run(self) -> Dict[str, Any]:
        logger.info("Computing explainability …")
        fi     = self._feature_importance()
        shap_s = self._shap_summary()

        # Use SHAP values if available, else fall back to feature_importances_
        combined = {}
        if shap_s.get("mean_abs_shap"):
            combined = shap_s["mean_abs_shap"]
        elif fi:
            combined = fi

        top = sorted(combined.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        return {
            "feature_importance": fi,
            "shap_summary": shap_s,
            "top_features": [{"feature": k, "importance": round(float(v), 6)} for k, v in top],
        }

    def _feature_importance(self) -> Dict[str, float]:
        m = self.model
        # For ensemble / stacking, try to get the final estimator's importances
        actual = getattr(m, "final_estimator_", None) or getattr(m, "estimators_", [m])[0] if hasattr(m, "estimators_") else m

        if hasattr(m, "feature_importances_"):
            fi = m.feature_importances_
        elif hasattr(actual, "feature_importances_"):
            fi = actual.feature_importances_
        elif hasattr(m, "coef_"):
            fi = np.abs(np.array(m.coef_)).flatten()
        elif hasattr(actual, "coef_"):
            fi = np.abs(np.array(actual.coef_)).flatten()
        else:
            return {}

        fi = np.array(fi).flatten()
        names = self.feature_names[:len(fi)]
        return {n: round(float(v), 6) for n, v in zip(names, fi)}

    def _shap_summary(self) -> Dict[str, Any]:
        try:
            import shap
            sample = min(150, self.X_val.shape[0])
            X_s = self.X_val[:sample]
            mname = type(self.model).__name__.lower()

            # Tree-based explainer (fast)
            if any(k in mname for k in ["forest", "xgb", "lgbm", "catboost", "boosting", "extratree"]):
                explainer = shap.TreeExplainer(self.model, feature_perturbation="interventional")
                raw = explainer.shap_values(X_s, check_additivity=False)
            else:
                # Linear explainer for linear models
                if any(k in mname for k in ["logistic", "ridge", "lasso", "linear"]):
                    try:
                        explainer = shap.LinearExplainer(self.model, self.X_train[:200])
                        raw = explainer.shap_values(X_s)
                    except Exception:
                        return {"method": "shap", "mean_abs_shap": {}, "error": "LinearExplainer failed"}
                else:
                    # Kernel explainer — slow, use tiny sample
                    bg = shap.kmeans(self.X_train, min(20, self.X_train.shape[0]))
                    predict_fn = (self.model.predict_proba
                                  if hasattr(self.model, "predict_proba")
                                  else self.model.predict)
                    explainer = shap.KernelExplainer(predict_fn, bg)
                    raw = explainer.shap_values(X_s[:30], nsamples=30, silent=True)

            # Normalize: handle list (multi-class) or 3d array
            if isinstance(raw, list):
                # list of [n_samples x n_features] — one per class
                arr = np.array(raw)           # shape: (n_classes, n_samples, n_features)
                mean_abs = np.abs(arr).mean(axis=(0, 1))  # (n_features,)
            elif isinstance(raw, np.ndarray):
                if raw.ndim == 3:             # (n_samples, n_features, n_classes)
                    mean_abs = np.abs(raw).mean(axis=(0, 2))
                elif raw.ndim == 2:           # (n_samples, n_features)
                    mean_abs = np.abs(raw).mean(axis=0)
                else:
                    mean_abs = np.abs(raw).flatten()
            else:
                mean_abs = np.zeros(len(self.feature_names))

            mean_abs = np.array(mean_abs).flatten()
            names = self.feature_names[:len(mean_abs)]
            result = {n: round(float(v), 6) for n, v in zip(names, mean_abs)}
            return {"method": "shap", "mean_abs_shap": result}

        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return {"method": "shap", "mean_abs_shap": {}, "error": str(e)}

    def explain_instance(self, instance: np.ndarray) -> Dict[str, Any]:
        """LIME explanation for a single instance."""
        try:
            import lime.lime_tabular
            mode = "classification" if self.problem_type == "classification" else "regression"
            exp_obj = lime.lime_tabular.LimeTabularExplainer(
                self.X_train, feature_names=self.feature_names,
                mode=mode, discretize_continuous=True, random_state=42,
            )
            predict_fn = (self.model.predict_proba
                          if hasattr(self.model, "predict_proba") and mode == "classification"
                          else self.model.predict)
            exp = exp_obj.explain_instance(instance, predict_fn, num_features=10, num_samples=300)
            return {
                "method": "lime",
                "features": [{"feature": f, "weight": round(float(w), 6)} for f, w in exp.as_list()],
            }
        except Exception as e:
            logger.warning(f"LIME failed: {e}")
            return {"method": "lime", "features": [], "error": str(e)}
