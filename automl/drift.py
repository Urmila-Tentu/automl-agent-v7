"""
AutoML Agent – Drift Detection & Monitoring
Detects data drift between reference (training) and production data.
Uses Evidently for statistical drift tests + PSI / KS / chi-squared checks.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight drift checks (no heavy dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def _ks_drift(ref: pd.Series, cur: pd.Series, threshold: float = 0.1) -> Dict:
    """Kolmogorov-Smirnov drift test for numeric features."""
    r = ref.dropna().values
    c = cur.dropna().values
    if len(r) < 5 or len(c) < 5:
        return {"drifted": False, "score": 0.0, "p_value": 1.0}
    ks_stat, p_val = stats.ks_2samp(r, c)
    return {
        "drifted": bool(ks_stat > threshold),
        "score":   round(float(ks_stat), 4),
        "p_value": round(float(p_val), 4),
    }


def _psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> Dict:
    """Population Stability Index for numeric features."""
    try:
        breakpoints = np.linspace(
            min(ref.min(), cur.min()),
            max(ref.max(), cur.max()),
            bins + 1,
        )
        ref_pct = np.histogram(ref, bins=breakpoints)[0] / len(ref)
        cur_pct = np.histogram(cur, bins=breakpoints)[0] / len(cur)
        ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        drifted = psi > 0.2
        return {"drifted": drifted, "psi": round(psi, 4)}
    except Exception:
        return {"drifted": False, "psi": 0.0}


def _chi2_drift(ref: pd.Series, cur: pd.Series, threshold: float = 0.05) -> Dict:
    """Chi-squared drift test for categorical features."""
    cats = list(set(ref.unique()) | set(cur.unique()))
    ref_counts = ref.value_counts().reindex(cats, fill_value=0)
    cur_counts = cur.value_counts().reindex(cats, fill_value=0)
    try:
        chi2, p_val = stats.chisquare(cur_counts, f_exp=ref_counts * len(cur) / len(ref))
        return {
            "drifted": bool(p_val < threshold),
            "chi2":    round(float(chi2), 4),
            "p_value": round(float(p_val), 4),
        }
    except Exception:
        return {"drifted": False, "chi2": 0.0, "p_value": 1.0}


# ─────────────────────────────────────────────────────────────────────────────
# Main drift detector
# ─────────────────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Compares production data against reference (training) data.
    Saves drift reports to disk for auditing.
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        reports_dir: Path,
        drift_threshold: float = 0.1,
    ) -> None:
        self.reference_df = reference_df.copy()
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold

    def check(self, production_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run drift checks across all shared columns.
        Returns a structured report with per-column results.
        """
        shared_cols = [
            c for c in self.reference_df.columns
            if c in production_df.columns
        ]

        report: Dict[str, Any] = {
            "timestamp":        datetime.utcnow().isoformat(),
            "n_reference_rows": len(self.reference_df),
            "n_production_rows": len(production_df),
            "columns_checked":  shared_cols,
            "column_drift":     {},
            "overall_drift_detected": False,
        }

        drifted_cols = []

        for col in shared_cols:
            ref_col = self.reference_df[col].dropna()
            cur_col = production_df[col].dropna()

            if len(cur_col) == 0:
                continue

            if pd.api.types.is_numeric_dtype(ref_col):
                ks = _ks_drift(ref_col, cur_col, self.drift_threshold)
                psi = _psi(ref_col, cur_col)
                col_result = {
                    "type":    "numeric",
                    "ks_test": ks,
                    "psi":     psi,
                    "drifted": ks["drifted"] or psi["drifted"],
                }
            else:
                chi2 = _chi2_drift(ref_col, cur_col)
                col_result = {
                    "type":      "categorical",
                    "chi2_test": chi2,
                    "drifted":   chi2["drifted"],
                }

            report["column_drift"][col] = col_result
            if col_result["drifted"]:
                drifted_cols.append(col)

        report["drifted_columns"] = drifted_cols
        report["drift_rate"] = round(len(drifted_cols) / max(len(shared_cols), 1), 4)
        report["overall_drift_detected"] = len(drifted_cols) > 0

        if report["overall_drift_detected"]:
            logger.warning(
                f"DATA DRIFT detected in columns: {drifted_cols} "
                f"(drift_rate={report['drift_rate']:.0%})"
            )
        else:
            logger.info("No significant data drift detected.")

        # Persist report
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"drift_report_{ts}.json"
        report_path.write_text(json.dumps(report, indent=2))
        logger.debug(f"Drift report saved to {report_path}")

        return report
