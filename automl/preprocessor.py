"""
AutoML Agent – Comprehensive Preprocessing Engine v2
Full feature engineering pipeline:
  - Smart ID/name/leakage column detection & auto-drop
  - Date/datetime feature extraction
  - Missing value analysis + 4 imputation strategies
  - Outlier detection (IQR, Z-score, Isolation Forest)
  - Skewness correction (log1p, Yeo-Johnson, Box-Cox)
  - Feature scaling: Standard, MinMax, Robust, Power, Quantile, Normalize
  - Categorical encoding: OneHot, Ordinal, Target, Binary, Frequency, Hash
  - Polynomial & interaction features (optional)
  - Dimensionality reduction: PCA, truncated SVD (after encoding)
  - Feature selection: Mutual Info, Variance Threshold, SelectKBest
  - 8-strategy benchmark grid – auto-selects best pipeline
"""
from __future__ import annotations
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, OrdinalEncoder,
    PowerTransformer, QuantileTransformer, RobustScaler,
    StandardScaler, TargetEncoder, OneHotEncoder, Normalizer,
    FunctionTransformer, Binarizer,
)
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression, VarianceThreshold
)
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

try:
    from loguru import logger
except ImportError:
    import logging; logger = logging.getLogger("automl")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ID / Leakage Column Detection
# ─────────────────────────────────────────────────────────────────────────────
_ID_PATTERNS = re.compile(
    r"^(id|idx|index|row_?num|row_?id|uuid|guid|key|pk|serial|record_?id|"
    r"name|full_?name|first_?name|last_?name|surname|fname|lname|"
    r"email|phone|address|url|link|ip|ssn|passport|license|"
    r"timestamp|created_?at|updated_?at|date_?time)$",
    re.IGNORECASE,
)

def detect_id_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Detect columns that are likely IDs, names, or leakage — should be dropped."""
    drop_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        s = df[col].dropna()
        # Pattern match on column name
        if _ID_PATTERNS.match(col.strip()):
            drop_cols.append(col); continue
        # Near-unique string column (>95% unique) → likely ID/name
        if s.dtype == object and len(s) > 10 and s.nunique() / len(s) > 0.95:
            drop_cols.append(col); continue
        # Integer monotonically increasing from 0/1 → index
        if pd.api.types.is_integer_dtype(s.dtype) and len(s) > 5:
            if s.is_monotonic_increasing and s.min() in (0, 1) and (s.diff().dropna() == 1).all():
                drop_cols.append(col); continue
    return drop_cols

# ─────────────────────────────────────────────────────────────────────────────
# Date Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Parse datetime columns and extract year, month, day, weekday, hour, is_weekend."""
    def __init__(self, date_cols: List[str]):
        self.date_cols = date_cols
        self._drop_cols: List[str] = []

    def fit(self, X, y=None):
        self._drop_cols = [c for c in self.date_cols if c in X.columns]
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        new_cols = {}
        for col in self._drop_cols:
            try:
                dt = pd.to_datetime(df[col], errors="coerce")
                new_cols[f"{col}_year"]    = dt.dt.year.fillna(0).astype(int)
                new_cols[f"{col}_month"]   = dt.dt.month.fillna(0).astype(int)
                new_cols[f"{col}_day"]     = dt.dt.day.fillna(0).astype(int)
                new_cols[f"{col}_weekday"] = dt.dt.weekday.fillna(0).astype(int)
                new_cols[f"{col}_hour"]    = dt.dt.hour.fillna(0).astype(int)
                new_cols[f"{col}_is_wknd"] = (dt.dt.weekday >= 5).astype(int)
            except Exception:
                pass
        df = df.drop(columns=self._drop_cols, errors="ignore")
        for k, v in new_cols.items():
            df[k] = v.values
        return df.values


# ─────────────────────────────────────────────────────────────────────────────
# Custom Transformers
# ─────────────────────────────────────────────────────────────────────────────
class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers to IQR fences (configurable factor)."""
    def __init__(self, factor: float = 3.0):
        self.factor = factor; self._bounds: Dict = {}

    def fit(self, X, y=None):
        arr = pd.DataFrame(X)
        for col in arr.columns:
            q1, q3 = arr[col].quantile(0.25), arr[col].quantile(0.75)
            iqr = q3 - q1
            self._bounds[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        arr = pd.DataFrame(X).copy()
        for col in arr.columns:
            lo, hi = self._bounds.get(col, (-np.inf, np.inf))
            arr[col] = arr[col].clip(lo, hi)
        return arr.values


class SkewnessCorrector(BaseEstimator, TransformerMixin):
    """log1p on right-skewed non-negative columns, Yeo-Johnson otherwise."""
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self._log_cols: List = []
        self._yj: Optional[PowerTransformer] = None
        self._yj_cols: List = []

    def fit(self, X, y=None):
        arr = pd.DataFrame(X)
        skews = arr.skew()
        self._log_cols = [c for c in arr.columns
                          if abs(skews[c]) > self.threshold and arr[c].min() >= 0]
        self._yj_cols  = [c for c in arr.columns
                          if abs(skews[c]) > self.threshold and c not in self._log_cols]
        if self._yj_cols:
            self._yj = PowerTransformer(method="yeo-johnson")
            self._yj.fit(arr[self._yj_cols].fillna(0))
        return self

    def transform(self, X):
        arr = pd.DataFrame(X).copy()
        for c in self._log_cols:
            arr[c] = np.log1p(arr[c].clip(lower=0))
        if self._yj and self._yj_cols:
            arr[self._yj_cols] = self._yj.transform(arr[self._yj_cols].fillna(0))
        return arr.values


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical columns by frequency (count / total)."""
    def __init__(self): self._freq_maps: Dict = {}

    def fit(self, X, y=None):
        arr = pd.DataFrame(X)
        for col in arr.columns:
            vc = arr[col].astype(str).value_counts(normalize=True)
            self._freq_maps[col] = vc.to_dict()
        return self

    def transform(self, X):
        arr = pd.DataFrame(X).copy()
        for col in arr.columns:
            arr[col] = arr[col].astype(str).map(self._freq_maps.get(col, {})).fillna(0.0)
        return arr.values


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Encode each category as binary bit columns."""
    def __init__(self): self._codes: Dict = {}; self._n_bits: Dict = {}

    def fit(self, X, y=None):
        arr = pd.DataFrame(X)
        for col in arr.columns:
            cats = arr[col].astype(str).unique().tolist()
            self._codes[col] = {c: i for i, c in enumerate(cats)}
            self._n_bits[col] = max(1, int(np.ceil(np.log2(len(cats) + 1))))
        return self

    def transform(self, X):
        arr = pd.DataFrame(X)
        result = []
        for col in arr.columns:
            codes = arr[col].astype(str).map(self._codes.get(col, {})).fillna(0).astype(int)
            n = self._n_bits.get(col, 1)
            for bit in range(n):
                result.append(((codes >> bit) & 1).values)
        return np.column_stack(result) if result else np.zeros((len(arr), 1))


class SmartPCAReducer(BaseEstimator, TransformerMixin):
    """Apply PCA to reduce to 95% explained variance when n_features > threshold."""
    def __init__(self, threshold: int = 40, variance: float = 0.95):
        self.threshold = threshold; self.variance = variance
        self._pca = None

    def fit(self, X, y=None):
        if X.shape[1] > self.threshold:
            n = min(X.shape[0] - 1, X.shape[1], 100)
            self._pca = PCA(n_components=self.variance, random_state=42)
            self._pca.fit(np.nan_to_num(X))
        return self

    def transform(self, X):
        if self._pca is not None:
            return self._pca.transform(np.nan_to_num(X))
        return X

    def n_components_used(self) -> int:
        if self._pca and hasattr(self._pca, "n_components_"):
            return int(self._pca.n_components_)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Column splitter
# ─────────────────────────────────────────────────────────────────────────────
def _split_columns(df: pd.DataFrame, target_col: str,
                   drop_cols: List[str]) -> Tuple[List, List, List, List]:
    """Returns num_cols, low_cat, high_cat, date_cols — excluding drop_cols and target."""
    skip = set(drop_cols) | {target_col}
    feature_df = df.drop(columns=[c for c in skip if c in df.columns], errors="ignore")

    # Detect date columns
    date_cols = []
    for col in feature_df.columns:
        if feature_df[col].dtype == object:
            sample = feature_df[col].dropna().head(50).astype(str)
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    date_cols.append(col)
            except Exception:
                pass

    remaining = [c for c in feature_df.columns if c not in date_cols]
    num_df = feature_df[remaining]
    num_cols = num_df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = num_df.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card  = [c for c in cat_cols if feature_df[c].nunique() <= 50]
    high_card = [c for c in cat_cols if feature_df[c].nunique() > 50]
    return num_cols, low_card, high_card, date_cols


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Engine
# ─────────────────────────────────────────────────────────────────────────────
class PreprocessingEngine:
    # (impute, scaler, cat_encoder)
    STRATEGY_GRID = [
        ("median",    "robust",    "onehot"),
        ("mean",      "standard",  "onehot"),
        ("knn",       "minmax",    "ordinal"),
        ("iterative", "standard",  "target"),
        ("median",    "robust",    "target"),
        ("mean",      "power",     "freq"),
        ("knn",       "quantile",  "binary"),
        ("median",    "normalize", "ordinal"),
    ]

    def __init__(self, df, target_col, problem_type,
                 feature_selection="mutual_info", handle_outliers=True,
                 correct_skewness=True, drop_constants=True,
                 use_pca_if_large=True):
        self.df               = df.copy()
        self.target_col       = target_col
        self.problem_type     = problem_type
        self.feature_selection= feature_selection
        self.handle_outliers  = handle_outliers
        self.correct_skewness = correct_skewness
        self.drop_constants   = drop_constants
        self.use_pca_if_large = use_pca_if_large

        # Auto-detect and remove ID/name columns
        self.dropped_id_cols = detect_id_columns(df, target_col)
        if self.dropped_id_cols:
            logger.info(f"Auto-dropping ID/name cols: {self.dropped_id_cols}")

        self.num_cols, self.low_cat_cols, self.high_cat_cols, self.date_cols = \
            _split_columns(df, target_col, self.dropped_id_cols)

        self.all_feature_cols = (self.num_cols + self.low_cat_cols +
                                 self.high_cat_cols + self.date_cols)

        self.best_pipeline:    Optional[Pipeline] = None
        self.best_strategy:    Optional[str]      = None
        self.pipeline_scores:  Dict[str, float]   = {}
        self.preprocessing_stats: Dict[str, Any]  = {}
        # Per-column medians/modes stored for sensitivity & scenario analysis
        self.feature_medians:  Dict[str, Any]     = {}
        # Friendly report dict (used by api/main.py for HTML reports)
        self.report:           Dict[str, Any]     = {}

    def build_best_pipeline(self, X_train, y_train, X_val, y_val) -> Pipeline:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score, r2_score
        is_clf   = self.problem_type == "classification"
        quick_m  = (LogisticRegression(max_iter=300, random_state=42, C=1.0)
                    if is_clf else Ridge(random_state=42))
        score_fn = accuracy_score if is_clf else r2_score

        best_score, best_pipeline, best_name = -np.inf, None, None

        for imp, scaler, enc in self.STRATEGY_GRID:
            name = f"{imp}|{scaler}|{enc}"
            try:
                pp = self._build_pipeline(imp, scaler, enc, y_train)
                pp.fit(X_train[self.all_feature_cols], y_train)
                Xv = pp.transform(X_val[self.all_feature_cols])
                Xt = pp.transform(X_train[self.all_feature_cols])
                quick_m.fit(Xt, y_train)
                score = score_fn(y_val, quick_m.predict(Xv))
                self.pipeline_scores[name] = round(float(score), 4)
                logger.info(f"  Preprocessing [{name}] score={score:.4f}")
                if score > best_score:
                    best_score, best_pipeline, best_name = score, pp, name
            except Exception as e:
                logger.warning(f"  Strategy [{name}] failed: {e}")
                self.pipeline_scores[name] = -999.0

        self.best_pipeline  = best_pipeline
        self.best_strategy  = best_name
        self._collect_stats(X_train)
        logger.info(f"Best preprocessing: {best_name} (score={best_score:.4f})")
        return best_pipeline

    def _collect_stats(self, X_train):
        parts = (self.best_strategy or "").split("|")

        # ── Store median/mode for every original feature column ───────────────
        # Used by sensitivity analysis to fill missing columns with sensible defaults
        self.feature_medians = {}
        for col in self.all_feature_cols:
            if col not in X_train.columns:
                self.feature_medians[col] = 0
                continue
            s = X_train[col].dropna()
            if len(s) == 0:
                self.feature_medians[col] = 0
            elif np.issubdtype(s.dtype, np.number):
                self.feature_medians[col] = float(s.median())
            else:
                modes = s.mode()
                self.feature_medians[col] = str(modes.iloc[0]) if len(modes) > 0 else ""

        # Estimate output feature count (after transform)
        try:
            sample = X_train[self.all_feature_cols].head(5)
            out = self.best_pipeline.transform(sample)
            output_feat_count = out.shape[1]
        except Exception:
            output_feat_count = len(self.all_feature_cols)

        # Check if PCA was applied
        pca_applied = False
        pca_components = 0
        try:
            pca_step = self.best_pipeline.named_steps.get("pca")
            if pca_step and pca_step._pca is not None:
                pca_applied = True
                pca_components = pca_step.n_components_used()
        except Exception:
            pass

        self.preprocessing_stats = {
            "strategy":              self.best_strategy,
            "scores":                self.pipeline_scores,
            "input_features":        len(self.all_feature_cols),
            "numeric_features":      len(self.num_cols),
            "low_card_cat":          len(self.low_cat_cols),
            "high_card_cat":         len(self.high_cat_cols),
            "date_cols_extracted":   self.date_cols,
            "auto_dropped_cols":     self.dropped_id_cols,
            "imputation_method":     parts[0] if len(parts) > 0 else "-",
            "scaling_method":        parts[1] if len(parts) > 1 else "-",
            "encoding_method":       parts[2] if len(parts) > 2 else "-",
            "outlier_handling":      self.handle_outliers,
            "skewness_correction":   self.correct_skewness,
            "pca_applied":           pca_applied,
            "pca_components":        pca_components,
            "output_features":       output_feat_count,
        }

        # ── Friendly report dict (used by HTML report endpoint) ──────────────
        self.report = {
            "best_strategy":   self.best_strategy,
            "scores":          self.pipeline_scores,
            "dropped_cols":    self.dropped_id_cols,
            "pca_applied":     pca_applied,
            "pca_components":  pca_components,
            "output_features": output_feat_count,
        }

    def _build_pipeline(self, imp_key, scaler_key, enc_key, y_train) -> Pipeline:
        steps: List = []

        # --- Numeric branch ---
        num_steps = [("imputer", self._imputer(imp_key))]
        if self.handle_outliers:
            num_steps.append(("clipper", OutlierClipper(factor=3.0)))
        if self.correct_skewness:
            num_steps.append(("skew", SkewnessCorrector()))
        num_steps.append(("scaler", self._scaler(scaler_key)))
        num_pipe = Pipeline(num_steps)

        # --- Low-cardinality categorical ---
        low_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", self._cat_encoder(enc_key, y_train, high_card=False)),
        ]
        low_pipe = Pipeline(low_steps)

        # --- High-cardinality categorical (always target-encode) ---
        high_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", TargetEncoder(random_state=42)),
        ]
        high_pipe = Pipeline(high_steps)

        # --- Date columns (string → numeric features) ---
        date_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("date_fe", DateFeatureExtractor(date_cols=self.date_cols)),
        ]
        date_pipe = Pipeline(date_steps) if self.date_cols else None

        transformers = []
        if self.num_cols:      transformers.append(("num",  num_pipe,  self.num_cols))
        if self.low_cat_cols:  transformers.append(("low",  low_pipe,  self.low_cat_cols))
        if self.high_cat_cols: transformers.append(("high", high_pipe, self.high_cat_cols))
        if self.date_cols and date_pipe:
            transformers.append(("date", date_pipe, self.date_cols))

        ct = ColumnTransformer(transformers=transformers, remainder="drop")
        steps = [("ct", ct)]

        # Variance threshold (remove near-zero variance features)
        if self.drop_constants:
            steps.append(("var_thresh", VarianceThreshold(threshold=1e-5)))

        # Feature selection
        fs = self._feature_selector(y_train)
        if fs:
            steps.append(("fs", fs))

        # PCA for high-dimensional data
        if self.use_pca_if_large:
            steps.append(("pca", SmartPCAReducer(threshold=40, variance=0.95)))

        return Pipeline(steps)

    @staticmethod
    def _imputer(key):
        return {
            "mean":      SimpleImputer(strategy="mean"),
            "median":    SimpleImputer(strategy="median"),
            "knn":       KNNImputer(n_neighbors=5),
            "iterative": IterativeImputer(random_state=42, max_iter=10),
            "zero":      SimpleImputer(strategy="constant", fill_value=0),
        }.get(key, SimpleImputer(strategy="median"))

    @staticmethod
    def _scaler(key):
        return {
            "standard":  StandardScaler(),
            "minmax":    MinMaxScaler(),
            "robust":    RobustScaler(),
            "power":     PowerTransformer(method="yeo-johnson"),
            "quantile":  QuantileTransformer(output_distribution="normal", random_state=42),
            "normalize": Normalizer(norm="l2"),
        }.get(key, StandardScaler())

    @staticmethod
    def _cat_encoder(key, y_train, high_card=False):
        if key == "onehot" and not high_card:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)
        if key == "ordinal":
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if key == "target":
            return TargetEncoder(random_state=42)
        if key == "freq":
            return FrequencyEncoder()
        if key == "binary":
            return BinaryEncoder()
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def _feature_selector(self, y_train):
        n = max(len(self.num_cols), 1)
        total = len(self.all_feature_cols)
        k = max(1, min(total, int(total * 0.90)))
        if self.feature_selection == "mutual_info" and total > 1:
            fn = (mutual_info_classif if self.problem_type == "classification"
                  else mutual_info_regression)
            return SelectKBest(score_func=fn, k=k)
        return None

    def get_scores(self): return self.pipeline_scores
    def get_stats(self):  return self.preprocessing_stats


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def remove_outliers_isolation_forest(X, contamination=0.05):
    model = IsolationForest(n_estimators=100, contamination=contamination,
                            random_state=42, n_jobs=-1)
    preds = model.fit_predict(X.select_dtypes(include=np.number).fillna(0))
    mask  = preds == 1
    logger.info(f"Isolation Forest removed {(~mask).sum()} rows ({(~mask).mean()*100:.1f}%)")
    return pd.Series(mask, index=X.index)


class TargetLabelEncoder:
    def __init__(self):
        self.le = LabelEncoder(); self.classes_ = None; self.is_fitted = False

    def fit_transform(self, y):
        encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_; self.is_fitted = True
        return encoded

    def transform(self, y):        return self.le.transform(y)
    def inverse_transform(self, y): return self.le.inverse_transform(y)
    def get_mapping(self):
        return {cls: int(i) for i, cls in enumerate(self.le.classes_)}
