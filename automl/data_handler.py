"""
AutoML Agent – Enhanced Data Handler & EDA Engine
Deep EDA: statistics, distributions, correlations, data quality scoring,
feature analysis, class balance, normality tests, mutual information.
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

try:
    from loguru import logger
except ImportError:
    import logging; logger = logging.getLogger("automl")

warnings.filterwarnings("ignore")

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")
    loaders = {
        ".csv":     lambda p: pd.read_csv(p),
        ".xlsx":    lambda p: pd.read_excel(p),
        ".xls":     lambda p: pd.read_excel(p),
        ".json":    lambda p: pd.read_json(p),
        ".parquet": lambda p: pd.read_parquet(p),
    }
    df = loaders[ext](path)
    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols from {path.name}")
    return df


def detect_problem_type(df: pd.DataFrame, target_col: str, datetime_col: str | None = None) -> str:
    if datetime_col and datetime_col in df.columns:
        return "time_series"
    target = df[target_col].dropna()
    n_unique = target.nunique()
    dtype = target.dtype
    if set(target.unique()).issubset({0, 1, True, False}):
        return "classification"
    if dtype == object or dtype.name == "category" or n_unique <= 20:
        return "classification"
    if np.issubdtype(dtype, np.floating) or (np.issubdtype(dtype, np.integer) and n_unique > 20):
        return "regression"
    return "classification"


class EDAEngine:
    """
    Comprehensive EDA — returns a rich JSON-serialisable dict with:
    shape, dtypes, missing analysis, target analysis, numeric stats,
    categorical stats, correlations, skewness/kurtosis, outlier analysis,
    normality tests, mutual information, data quality score, recommendations.
    """

    def __init__(self, df: pd.DataFrame, target_col: str) -> None:
        self.df = df.copy()
        self.target_col = target_col
        self._num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self._cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    def run(self) -> Dict[str, Any]:
        logger.info("Running comprehensive EDA …")
        report = {
            "shape":              self._shape(),
            "dtypes":             self._dtypes(),
            "memory_mb":          round(self.df.memory_usage(deep=True).sum() / 1e6, 3),
            "duplicate_rows":     int(self.df.duplicated().sum()),
            "missing":            self._missing(),
            "target_summary":     self._target_summary(),
            "numeric_stats":      self._numeric_stats(),
            "distribution":       self._distribution(),
            "categorical_stats":  self._categorical_stats(),
            "correlations":       self._correlations(),
            "skewness_kurtosis":  self._skewness_kurtosis(),
            "outlier_analysis":   self._outlier_analysis(),
            "normality_tests":    self._normality_tests(),
            "mutual_information": self._mutual_info(),
            "feature_types":      self._feature_types(),
            "data_quality_score": self._quality_score(),
            "recommendations":    self._recommendations(),
        }
        logger.info("EDA complete")
        return report

    # ── shape / types ─────────────────────────────────────────────────────────
    def _shape(self):
        return {"rows": int(self.df.shape[0]), "cols": int(self.df.shape[1])}

    def _dtypes(self):
        return {col: str(dt) for col, dt in self.df.dtypes.items()}

    def _feature_types(self):
        return {
            "numeric":     self._num_cols,
            "categorical": self._cat_cols,
            "datetime":    self.df.select_dtypes(include=["datetime"]).columns.tolist(),
            "boolean":     self.df.select_dtypes(include=["bool"]).columns.tolist(),
        }

    # ── missing ───────────────────────────────────────────────────────────────
    def _missing(self):
        miss = self.df.isnull().sum()
        pct  = (miss / len(self.df) * 100).round(2)
        total_missing = int(miss.sum())
        return {
            "total_missing_values": total_missing,
            "total_missing_pct": round(total_missing / self.df.size * 100, 2),
            "columns": {
                col: {
                    "count": int(miss[col]),
                    "pct":   float(pct[col]),
                    "severity": "high" if pct[col] > 30 else "medium" if pct[col] > 5 else "low"
                }
                for col in self.df.columns if miss[col] > 0
            }
        }

    # ── target ────────────────────────────────────────────────────────────────
    def _target_summary(self):
        target = self.df[self.target_col].dropna()
        summary: Dict[str, Any] = {"dtype": str(target.dtype), "n_missing": int(self.df[self.target_col].isnull().sum())}
        if target.dtype == object or target.nunique() <= 20:
            vc = target.value_counts()
            counts = vc.to_dict()
            pcts   = (vc / len(target) * 100).round(2).to_dict()
            # imbalance ratio = max_class / min_class
            imbalance = round(max(vc) / max(min(vc), 1), 2)
            summary.update({
                "type": "categorical",
                "n_classes": int(target.nunique()),
                "class_counts": {str(k): int(v) for k, v in counts.items()},
                "class_pcts":   {str(k): float(v) for k, v in pcts.items()},
                "imbalance_ratio": imbalance,
                "is_imbalanced": imbalance > 3,
            })
        else:
            q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            qvals = target.quantile(q).round(4).tolist()
            summary.update({
                "type":    "numeric",
                "mean":    round(float(target.mean()), 4),
                "std":     round(float(target.std()), 4),
                "min":     round(float(target.min()), 4),
                "max":     round(float(target.max()), 4),
                "median":  round(float(target.median()), 4),
                "skewness": round(float(target.skew()), 4),
                "kurtosis": round(float(target.kurt()), 4),
                "percentiles": dict(zip([f"p{int(x*100)}" for x in q], qvals)),
            })
        return summary

    # ── numeric stats ─────────────────────────────────────────────────────────
    def _numeric_stats(self):
        if not self._num_cols:
            return {}
        result = {}
        for col in self._num_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            result[col] = {
                "count":  int(s.count()),
                "mean":   round(float(s.mean()), 4),
                "std":    round(float(s.std()), 4),
                "min":    round(float(s.min()), 4),
                "max":    round(float(s.max()), 4),
                "median": round(float(s.median()), 4),
                "q1":     round(q1, 4),
                "q3":     round(q3, 4),
                "iqr":    round(q3 - q1, 4),
                "cv":     round(float(s.std() / s.mean()) if s.mean() != 0 else 0, 4),  # coeff of variation
                "range":  round(float(s.max() - s.min()), 4),
            }
        return result

    # ── distribution histogram buckets (for charts) ───────────────────────────
    def _distribution(self):
        result = {}
        for col in self._num_cols[:12]:  # limit to 12 for perf
            s = self.df[col].dropna()
            if len(s) < 5:
                continue
            try:
                counts, edges = np.histogram(s, bins=20)
                result[col] = {
                    "bins":   [round(float(e), 4) for e in edges[:-1]],
                    "counts": [int(c) for c in counts],
                }
            except Exception:
                pass
        return result

    # ── categorical stats ─────────────────────────────────────────────────────
    def _categorical_stats(self):
        result = {}
        for col in self._cat_cols:
            vc = self.df[col].value_counts()
            result[col] = {
                "n_unique":   int(self.df[col].nunique()),
                "top10":      {str(k): int(v) for k, v in vc.head(10).items()},
                "missing":    int(self.df[col].isnull().sum()),
                "entropy":    round(float(stats.entropy(vc.values + 1e-9)), 4),
                "cardinality": "high" if self.df[col].nunique() > 50 else "medium" if self.df[col].nunique() > 10 else "low",
            }
        return result

    # ── correlations ──────────────────────────────────────────────────────────
    def _correlations(self):
        num_df = self.df[self._num_cols]
        if num_df.shape[1] < 2:
            return {}
        pearson  = num_df.corr(method="pearson").round(4)
        spearman = num_df.corr(method="spearman").round(4)

        # Top correlations with target
        target_corr = {}
        if self.target_col in pearson.columns:
            tc = pearson[self.target_col].drop(self.target_col, errors="ignore")
            target_corr = tc.abs().sort_values(ascending=False).head(15).round(4).to_dict()

        # Highly correlated feature pairs (|r| > 0.85) → multicollinearity warning
        high_corr_pairs = []
        cols = pearson.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                v = abs(pearson.iloc[i, j])
                if v > 0.85:
                    high_corr_pairs.append({"f1": cols[i], "f2": cols[j], "r": round(float(v), 4)})

        return {
            "pearson_matrix":    pearson.to_dict(),
            "spearman_matrix":   spearman.to_dict(),
            "target_correlations": target_corr,
            "high_correlation_pairs": high_corr_pairs[:10],
        }

    # ── skewness / kurtosis ───────────────────────────────────────────────────
    def _skewness_kurtosis(self):
        result = {}
        for col in self._num_cols:
            s = self.df[col].dropna()
            sk = float(s.skew())
            ku = float(s.kurt())
            result[col] = {
                "skewness": round(sk, 4),
                "kurtosis": round(ku, 4),
                "skew_label": "highly_positive" if sk > 2 else "positive" if sk > 0.5 else "highly_negative" if sk < -2 else "negative" if sk < -0.5 else "symmetric",
                "leptokurtic": ku > 0,
            }
        return result

    # ── outlier analysis ──────────────────────────────────────────────────────
    def _outlier_analysis(self):
        result = {}
        for col in self._num_cols:
            s = self.df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            # Z-score outliers
            z = np.abs(stats.zscore(s))
            n_iqr = int(((s < lo) | (s > hi)).sum())
            n_z   = int((z > 3).sum())
            if n_iqr > 0 or n_z > 0:
                result[col] = {
                    "iqr_outliers": n_iqr,
                    "zscore_outliers": n_z,
                    "iqr_pct":  round(n_iqr / len(s) * 100, 2),
                    "lower_fence": round(float(lo), 4),
                    "upper_fence": round(float(hi), 4),
                }
        return result

    # ── normality tests ───────────────────────────────────────────────────────
    def _normality_tests(self):
        result = {}
        for col in self._num_cols[:10]:
            s = self.df[col].dropna()
            if len(s) < 8 or len(s) > 5000:
                continue
            try:
                stat, p = stats.shapiro(s.sample(min(500, len(s)), random_state=42))
                result[col] = {
                    "shapiro_stat": round(float(stat), 4),
                    "shapiro_p":    round(float(p), 4),
                    "is_normal":    bool(p > 0.05),
                }
            except Exception:
                pass
        return result

    # ── mutual information ────────────────────────────────────────────────────
    def _mutual_info(self):
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            target = self.df[self.target_col].dropna()
            feat_df = self.df[self._num_cols].copy()
            # Remove target from features if numeric
            feat_df = feat_df.drop(columns=[self.target_col], errors="ignore")
            feat_df = feat_df.fillna(feat_df.median())
            if len(feat_df.columns) == 0:
                return {}
            idx = feat_df.index.intersection(target.index)
            X = feat_df.loc[idx]
            y = target.loc[idx]
            is_clf = y.dtype == object or y.nunique() <= 20
            fn = mutual_info_classif if is_clf else mutual_info_regression
            mi = fn(X, y, random_state=42)
            return dict(sorted(
                {col: round(float(v), 4) for col, v in zip(feat_df.columns, mi)}.items(),
                key=lambda x: x[1], reverse=True
            ))
        except Exception:
            return {}

    # ── data quality score (0-100) ────────────────────────────────────────────
    def _quality_score(self):
        scores = {}
        miss_pct = self.df.isnull().mean().mean() * 100
        scores["completeness"] = round(max(0, 100 - miss_pct * 2), 1)
        dup_pct = self.df.duplicated().mean() * 100
        scores["uniqueness"] = round(max(0, 100 - dup_pct * 5), 1)
        # Consistency: how many numeric cols have extreme outliers
        out_fracs = []
        for col in self._num_cols:
            s = self.df[col].dropna()
            if len(s) == 0: continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            f = ((s < q1 - 3*iqr) | (s > q3 + 3*iqr)).mean()
            out_fracs.append(f)
        avg_out = np.mean(out_fracs) * 100 if out_fracs else 0
        scores["consistency"] = round(max(0, 100 - avg_out * 10), 1)
        scores["overall"] = round(np.mean(list(scores.values())), 1)
        return scores

    # ── recommendations ───────────────────────────────────────────────────────
    def _recommendations(self) -> List[Dict]:
        tips = []
        miss = self.df.isnull().sum()
        pct  = (miss / len(self.df) * 100)

        high_miss = [c for c in self.df.columns if pct[c] > 30]
        med_miss  = [c for c in self.df.columns if 5 < pct[c] <= 30]
        if high_miss:
            tips.append({"severity": "high",   "msg": f"Columns {high_miss} have >30% missing — consider dropping.", "icon": "🔴"})
        if med_miss:
            tips.append({"severity": "medium", "msg": f"Columns {med_miss} have 5-30% missing — use KNN or iterative imputation.", "icon": "🟡"})

        dups = int(self.df.duplicated().sum())
        if dups:
            tips.append({"severity": "medium", "msg": f"{dups} duplicate rows detected — deduplication recommended.", "icon": "🟡"})

        cat_cols = self.df.select_dtypes(include="object").columns
        high_card = [c for c in cat_cols if self.df[c].nunique() > 50 and c != self.target_col]
        if high_card:
            tips.append({"severity": "medium", "msg": f"High-cardinality columns {high_card} — use target/hash encoding.", "icon": "🟡"})

        sk = self.df[self._num_cols].skew()
        highly_skewed = sk[abs(sk) > 2].index.tolist()
        if highly_skewed:
            tips.append({"severity": "low", "msg": f"Skewed features {highly_skewed} — log/Box-Cox transform recommended.", "icon": "🔵"})

        # Class imbalance
        target = self.df[self.target_col].dropna()
        if target.dtype == object or target.nunique() <= 20:
            vc = target.value_counts()
            if len(vc) > 1 and vc.max() / vc.min() > 3:
                tips.append({"severity": "high", "msg": f"Class imbalance detected (ratio {round(vc.max()/vc.min(),1)}:1) — SMOTE/class_weight recommended.", "icon": "🔴"})

        if not tips:
            tips.append({"severity": "good", "msg": "Dataset looks clean! No critical issues found.", "icon": "✅"})
        return tips
