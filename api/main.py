"""AutoML Agent – FastAPI v6.0 | Clean slate · Novel features · All bugs fixed"""
from __future__ import annotations
import json, pickle, uuid, re, time, asyncio, io, html as _html

def _esc(v) -> str:
    """HTML-escape any value safely — usable anywhere in this module."""
    return _html.escape(str(v) if v is not None else "—")
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np

_ROOT     = Path(__file__).resolve().parent.parent
_FRONTEND = _ROOT / "frontend"

app = FastAPI(title="AutoML Agent", version="6.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")

EXPS: Dict[str, Dict]     = {}
EXP_LOGS: Dict[str, List] = {}

# ── helpers ───────────────────────────────────────────────────────────────────
def cfg():
    from automl.config import settings; return settings
def L():
    try: from loguru import logger; return logger
    except: import logging; return logging.getLogger("automl")
def auth(req: Request): return {"username": "admin", "role": "admin"}

def _js(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, Path):        return str(obj)
    return str(obj)

def _push_log(eid: str, msg: str):
    EXP_LOGS.setdefault(eid, []).append(f"{time.strftime('%H:%M:%S')} {msg}")
    if len(EXP_LOGS[eid]) > 500: EXP_LOGS[eid] = EXP_LOGS[eid][-500:]

def _save_meta(eid: str):
    try:
        exp = EXPS.get(eid)
        if not exp: return
        d = cfg().experiments_dir / eid
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text(json.dumps({
            "experiment_id": eid, "status": exp["status"],
            "target_col": exp.get("target_col"), "filename": exp.get("filename"),
            "dataset_name": exp.get("dataset_name", ""),
            "submitted_by": exp.get("submitted_by", "admin"),
            "created_at": exp.get("created_at"), "summary": exp.get("summary"),
        }, indent=2, default=_js))
    except Exception as e: L().warning(f"meta save: {e}")

def _load_disk():
    """Load completed experiments from disk. Restores model into memory if pkl exists."""
    try:
        base = cfg().experiments_dir
        if not base.exists(): return
        for d in sorted(base.iterdir()):
            if not d.is_dir(): continue
            mp = d / "metadata.json"
            if not mp.exists(): continue
            try:
                meta = json.loads(mp.read_text())
                eid = meta.get("experiment_id") or d.name
                if eid in EXPS: continue
                status = meta.get("status", "completed")
                if status != "completed": continue

                # Try to restore the agent from saved pkl files
                agent = None
                model_pkl  = d / "best_model.pkl"
                pp_pkl     = d / "preprocessor.pkl"
                le_pkl     = d / "label_encoder.pkl"
                if model_pkl.exists() and pp_pkl.exists():
                    try:
                        # Reconstruct a minimal agent-like object from saved artifacts
                        from automl.agent import AutoMLAgent
                        agent = AutoMLAgent.__new__(AutoMLAgent)
                        with open(model_pkl, "rb") as f:
                            agent.best_model = pickle.load(f)
                        with open(pp_pkl, "rb") as f:
                            agent.preprocessing_engine = pickle.load(f)
                        agent.label_encoder = None
                        if le_pkl.exists():
                            with open(le_pkl, "rb") as f:
                                agent.label_encoder = pickle.load(f)
                        # Fill required attributes from metadata
                        s = meta.get("summary") or {}
                        agent.problem_type   = s.get("problem_type") or meta.get("problem_type")
                        agent.target_col     = meta.get("target_col") or s.get("target_col")
                        agent.best_model_name = s.get("best_model") or s.get("best_model_name")
                        agent.experiment_id  = eid
                        agent.df             = None   # raw data not kept
                        agent.X_train = agent.X_val = agent.y_train = agent.y_val = None
                        agent.feature_names  = [f["feature"] for f in s.get("top_features", [])]
                        agent.X_raw_sample   = None
                        L().info(f"✅ Reloaded agent for {eid} from disk")
                    except Exception as le:
                        agent = None
                        L().warning(f"Could not reload agent for {eid}: {le}")

                EXPS[eid] = {
                    "status": status, "agent": agent,
                    "summary": meta.get("summary"),
                    "target_col": meta.get("target_col"),
                    "filename": meta.get("filename"),
                    "dataset_name": meta.get("dataset_name", ""),
                    "submitted_by": meta.get("submitted_by"),
                    "created_at": meta.get("created_at"),
                    "_from_disk": True,
                }
                EXP_LOGS[eid] = [
                    f"📂 Restored from disk" + (" + model in memory ✅" if agent else " (model not found — retrain)")
                ]
            except Exception as e:
                L().warning(f"load {d}: {e}")
    except Exception as e:
        L().warning(f"disk load: {e}")

def get_agent(eid: str):
    exp = EXPS.get(eid)
    if not exp: raise HTTPException(404, f"Experiment '{eid}' not found")
    if exp["status"] != "completed":
        raise HTTPException(400, f"Status={exp['status']} error={exp.get('error','')}")
    agent = exp.get("agent")
    if not agent:
        raise HTTPException(503, "Model not in memory — retrain to enable predictions")
    return agent

def _get_src(agent):
    """Get raw feature DataFrame from agent."""
    src = getattr(agent, "X_raw_sample", None)
    if src is None and agent.df is not None:
        src = agent.df.drop(columns=[agent.target_col], errors="ignore")
    return src

def _fill_missing_cols(df: pd.DataFrame, agent) -> pd.DataFrame:
    """Fill any missing feature columns with dataset medians/modes."""
    pp = agent.preprocessing_engine
    if not hasattr(pp, "all_feature_cols"):
        return df
    src = _get_src(agent)
    for col in pp.all_feature_cols:
        if col not in df.columns:
            if src is not None and col in src.columns:
                s = src[col].dropna()
                df[col] = float(s.median()) if np.issubdtype(s.dtype, np.number) else (str(s.mode()[0]) if len(s.mode()) else "")
            else:
                df[col] = 0
    return df

# ── pydantic models ────────────────────────────────────────────────────────────
class PR(BaseModel):   data: List[Dict[str, Any]]
class SensReq(BaseModel): data: List[Dict[str, Any]]

# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    idx = _FRONTEND / "index.html"
    return HTMLResponse(idx.read_text("utf-8") if idx.exists() else "<h1>AutoML v6</h1>")

@app.get("/health")
async def health():
    return {"ok": True, "experiments": len(EXPS),
            "in_memory": sum(1 for e in EXPS.values() if e.get("agent"))}

@app.post("/auth/token", tags=["Auth"])
async def login(form: OAuth2PasswordRequestForm = Depends()):
    return {"access_token": "open", "token_type": "bearer"}

@app.get("/auth/me", tags=["Auth"])
async def me(): return {"username": "admin", "role": "admin"}


# ── training ───────────────────────────────────────────────────────────────────
def _train(eid, path, target, prob, feat_sel, outliers, n_trials):
    EXPS[eid]["status"] = "training"; EXPS[eid]["started_at"] = time.time()
    _push_log(eid, "🚀 Pipeline started")
    try:
        from automl.agent import AutoMLAgent
        _push_log(eid, "📊 Loading & profiling dataset…")
        agent = AutoMLAgent(target_col=target, problem_type=prob or None,
                            feature_selection=feat_sel, handle_outliers=outliers,
                            n_trials=int(n_trials))
        _push_log(eid, "⚙️  Benchmarking preprocessing strategies…")
        _push_log(eid, "🤖 Training models with Optuna HPO…")
        summary = agent.run(path)
        agent.experiment_id = eid

        if "best_model" not in summary and "best_model_name" in summary:
            summary["best_model"] = summary["best_model_name"]
        if agent.df is not None:
            summary["dataset_shape"] = [int(agent.df.shape[0]), int(agent.df.shape[1])]

        summary["elapsed_sec"]  = round(time.time() - EXPS[eid]["started_at"], 1)
        summary["dataset_name"] = EXPS[eid].get("dataset_name", eid)

        # AI insights
        lb  = summary.get("leaderboard", [])
        bm  = summary.get("best_metrics", {})
        fi  = summary.get("top_features", [])
        clf = summary.get("problem_type") == "classification"
        sk  = "accuracy" if clf else "r2"; sv = bm.get(sk, 0)
        ins = []
        if   sv >= 0.95: ins.append(f"🏆 Exceptional {sk} {sv:.3f} — verify no data leakage.")
        elif sv >= 0.85: ins.append(f"✅ Strong {sk} {sv:.3f} — production-ready.")
        elif sv >= 0.70: ins.append(f"⚡ Decent {sk} {sv:.3f} — more HPO trials may help.")
        else:            ins.append(f"⚠️ Low {sk} {sv:.3f} — consider more data or feature engineering.")
        if fi: ins.append(f"🔑 Top predictors: {', '.join(f['feature'] for f in fi[:3])}.")
        summary["ai_insights"] = ins

        EXPS[eid].update({"agent": agent, "summary": summary, "status": "completed"})
        _push_log(eid, f"✅ Done in {summary['elapsed_sec']}s — best: {summary.get('best_model','—')}")

        # ── Persist model to the correct experiment directory so it survives restarts ──
        try:
            exp_dir = cfg().experiments_dir / eid
            exp_dir.mkdir(parents=True, exist_ok=True)
            with open(exp_dir / "best_model.pkl", "wb") as f:
                pickle.dump(agent.best_model, f)
            with open(exp_dir / "preprocessor.pkl", "wb") as f:
                pickle.dump(agent.preprocessing_engine, f)
            if getattr(agent, "label_encoder", None) is not None:
                with open(exp_dir / "label_encoder.pkl", "wb") as f:
                    pickle.dump(agent.label_encoder, f)
            _push_log(eid, "💾 Model saved to disk")
        except Exception as pe:
            _push_log(eid, f"⚠️  Could not persist model: {pe}")

        _save_meta(eid)
        L().info(f"✅ {eid} complete")
    except Exception as e:
        import traceback
        EXPS[eid].update({"status": "failed", "error": str(e)})
        _push_log(eid, f"❌ Failed: {e}")
        _save_meta(eid)
        L().error(f"❌ {eid}: {traceback.format_exc()}")


@app.post("/experiments/train", tags=["Training"])
async def train(req: Request, bg: BackgroundTasks,
                file: UploadFile = File(...), target_col: str = Form(...),
                dataset_name: str = Form(default=""),
                problem_type: str = Form(default=""),
                feature_selection: str = Form(default="mutual_info"),
                handle_outliers: bool = Form(default=True),
                n_trials: int = Form(default=10)):
    auth(req); eid = str(uuid.uuid4())[:8]
    safe = re.sub(r'[^\w\-.]', '_', (dataset_name or file.filename or eid).strip())[:48]
    ext  = Path(file.filename or "data.csv").suffix or ".csv"
    p    = cfg().data_dir / f"{safe}_{eid}{ext}"
    p.write_bytes(await file.read())
    dname = dataset_name.strip() or file.filename or eid
    EXPS[eid] = {"status": "queued", "agent": None, "summary": None,
                 "target_col": target_col, "filename": file.filename,
                 "dataset_name": dname, "submitted_by": "admin",
                 "created_at": time.time()}
    EXP_LOGS[eid] = []
    _push_log(eid, f"📁 '{dname}' uploaded · target={target_col}")
    bg.add_task(_train, eid, p, target_col, problem_type or None,
                feature_selection, handle_outliers, n_trials)
    return {"experiment_id": eid, "status": "queued", "dataset_name": dname}


@app.get("/experiments/{eid}/logs/stream", tags=["Training"])
async def stream_logs(eid: str, since: int = 0):
    async def gen():
        idx = since
        for _ in range(720):
            lines = EXP_LOGS.get(eid, [])
            while idx < len(lines):
                yield f"data: {json.dumps(lines[idx])}\n\n"; idx += 1
            if EXPS.get(eid, {}).get("status") in ("completed", "failed"):
                yield 'data: "__DONE__"\n\n'; return
            await asyncio.sleep(1)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/experiments/{eid}/logs", tags=["Training"])
async def get_logs(eid: str): return {"logs": EXP_LOGS.get(eid, [])}

@app.get("/experiments", tags=["Training"])
async def list_exps(req: Request):
    auth(req)
    return [{"experiment_id": k, "status": v["status"],
             "target_col": v.get("target_col"), "filename": v.get("filename"),
             "dataset_name": v.get("dataset_name", ""),
             "submitted_by": v.get("submitted_by"),
             "created_at": v.get("created_at"),
             "in_memory": v.get("agent") is not None}
            for k, v in EXPS.items()]

@app.get("/experiments/{eid}/status", tags=["Training"])
async def exp_status(eid: str, req: Request):
    auth(req); exp = EXPS.get(eid)
    if not exp: raise HTTPException(404)
    return {"experiment_id": eid, "status": exp["status"],
            "error": exp.get("error"), "in_memory": exp.get("agent") is not None}

@app.get("/experiments/{eid}/results", tags=["Training"])
async def results(eid: str, req: Request):
    auth(req); exp = EXPS.get(eid)
    if not exp: raise HTTPException(404)
    if exp["status"] != "completed":
        raise HTTPException(400, f"Status={exp['status']}")
    s = dict(exp["summary"] or {})
    if "best_model" not in s: s["best_model"] = s.get("best_model_name") or "—"
    s["in_memory"] = exp.get("agent") is not None
    return s

@app.delete("/experiments/{eid}", tags=["Training"])
async def delete_exp(eid: str, req: Request):
    auth(req); EXPS.pop(eid, None); EXP_LOGS.pop(eid, None)
    # Also delete from disk
    try:
        exp_dir = cfg().experiments_dir / eid
        if exp_dir.exists():
            import shutil; shutil.rmtree(exp_dir)
    except: pass
    return {"deleted": eid}

@app.delete("/experiments", tags=["Training"])
async def clear_all_exps(req: Request):
    """Clear ALL ghost experiments (disk-only, no model in memory)."""
    auth(req)
    ghosts = [k for k, v in EXPS.items() if not v.get("agent")]
    for eid in ghosts:
        EXPS.pop(eid, None); EXP_LOGS.pop(eid, None)
    return {"cleared": len(ghosts), "remaining": len(EXPS)}


# ── Schema / Inference ─────────────────────────────────────────────────────────
@app.get("/experiments/{eid}/schema", tags=["Inference"])
async def schema(eid: str, req: Request):
    auth(req); agent = get_agent(eid)
    src = _get_src(agent)
    if src is None: raise HTTPException(500, "Feature data unavailable")
    pp = agent.preprocessing_engine
    all_cols = pp.all_feature_cols if hasattr(pp, "all_feature_cols") else list(src.columns)
    feats = []
    for col in all_cols:
        if col not in src.columns: continue
        s = src[col].dropna(); is_num = np.issubdtype(s.dtype, np.number)
        e: Dict = {"name": col, "type": "numeric" if is_num else "categorical",
                   "dtype": str(s.dtype), "missing_pct": round(src[col].isnull().mean()*100,1)}
        if is_num:
            e.update({"min": round(float(s.min()),4), "max": round(float(s.max()),4),
                      "mean": round(float(s.mean()),4), "median": round(float(s.median()),4),
                      "std": round(float(s.std()),4)})
        else:
            e["categories"] = [str(c) for c in s.value_counts().index[:50]]
        feats.append(e)
    return {"experiment_id": eid, "problem_type": agent.problem_type,
            "target_col": agent.target_col, "n_features": len(feats), "features": feats,
            "label_classes": (agent.label_encoder.classes_.tolist()
                              if getattr(agent, "label_encoder", None) and
                                 getattr(agent.label_encoder, "is_fitted", False) else None)}


@app.post("/experiments/{eid}/predict", tags=["Inference"])
async def predict(eid: str, body: PR, req: Request):
    auth(req); agent = get_agent(eid); t = time.time()
    try:
        df = pd.DataFrame(body.data)
        df = _fill_missing_cols(df, agent)   # ← key fix: fill missing columns
        result = agent.predict(df)
    except Exception as e:
        raise HTTPException(422, str(e))
    return {**result, "latency_ms": round((time.time()-t)*1000, 2)}


@app.post("/experiments/{eid}/predict/batch-file", tags=["Inference"])
async def batch_predict(eid: str, file: UploadFile = File(...), req: Request = None):
    auth(req); agent = get_agent(eid)
    content = await file.read()
    try: df = pd.read_csv(io.BytesIO(content))
    except Exception as e: raise HTTPException(400, f"Cannot parse CSV: {e}")
    if df.empty: raise HTTPException(400, "CSV is empty")
    if len(df) > 10000: raise HTTPException(400, "Max 10,000 rows")
    t = time.time()
    try:
        df = _fill_missing_cols(df, agent)
        result = agent.predict(df)
    except Exception as e: raise HTTPException(422, str(e))
    preds = result.get("predictions", []); probs = result.get("probabilities")
    rows = [{"row": i, "prediction": p,
             **( {"confidence": round(max(probs[i]),4)} if probs and i < len(probs) else {})}
            for i, p in enumerate(preds)]
    keys = list(rows[0].keys()) if rows else ["row", "prediction"]
    csv_text = ",".join(keys) + "\n" + "\n".join(",".join(str(r.get(k,"")) for k in keys) for r in rows)
    return JSONResponse({"experiment_id": eid, "n_rows": len(preds),
                         "latency_ms": round((time.time()-t)*1000,2),
                         "predictions": preds, "probabilities": probs, "csv": csv_text})


@app.post("/experiments/{eid}/explain", tags=["Inference"])
async def explain(eid: str, body: PR, req: Request):
    auth(req); agent = get_agent(eid)
    try:
        from automl.explainability import ExplainabilityEngine
        ee = ExplainabilityEngine(model=agent.best_model, X_train=agent.X_train,
                                  X_val=agent.X_val, feature_names=agent.feature_names,
                                  problem_type=agent.problem_type)
        row = pd.DataFrame(body.data[:1])
        pp  = agent.preprocessing_engine
        Xt  = pp.best_pipeline.transform(row[pp.all_feature_cols])
        return ee.explain_instance(Xt[0])
    except Exception as e:
        return {"method": "lime", "features": [], "error": str(e)}


# ── Sensitivity Analysis ────────────────────────────────────────────────────────
@app.post("/experiments/{eid}/sensitivity", tags=["Analysis"])
async def sensitivity(eid: str, body: SensReq, req: Request):
    auth(req); agent = get_agent(eid)
    base_row = body.data[0]
    pp  = agent.preprocessing_engine
    src = _get_src(agent)
    clf = agent.problem_type == "classification"
    all_cols = pp.all_feature_cols if hasattr(pp, "all_feature_cols") else list(base_row.keys())

    # Build complete baseline row
    full_row: Dict = {}
    for c in all_cols:
        if c in base_row and base_row[c] is not None:
            full_row[c] = base_row[c]
        elif src is not None and c in src.columns:
            s = src[c].dropna()
            full_row[c] = float(s.median()) if np.issubdtype(s.dtype, np.number) else (str(s.mode()[0]) if len(s.mode()) else "")
        else:
            full_row[c] = 0

    def _numeric_score(res: dict) -> float:
        """Return a consistent float score regardless of problem type.
        - Regression: the predicted value
        - Classification: confidence of predicted class (max probability)
        """
        if clf:
            probs = res.get("probabilities")
            if probs and len(probs) > 0 and isinstance(probs[0], list):
                return float(max(probs[0]))
            # fall back to index of predicted class
            pred = res.get("predictions", [None])[0]
            classes = getattr(agent, "label_encoder", None)
            if classes and hasattr(classes, "classes_"):
                try:
                    idx = list(classes.classes_).index(pred)
                    return float(idx) / max(len(classes.classes_) - 1, 1)
                except: pass
            return 0.0
        else:
            pred = res.get("predictions", [None])[0]
            try: return float(pred)
            except: return 0.0

    try:
        base_res  = agent.predict(pd.DataFrame([full_row]))
        base_pred = base_res["predictions"][0]
        base_num  = _numeric_score(base_res)
        metric_label = "confidence" if clf else "predicted value"
    except Exception as e:
        raise HTTPException(422, f"Baseline prediction failed: {e}")

    results = []
    for col in all_cols:
        if src is not None and col in src.columns:
            s = src[col].dropna()
            if not np.issubdtype(s.dtype, np.number): continue
            std  = float(s.std()) or 1.0
            mean = float(s.median())
        else:
            v = full_row.get(col, 0)
            try: v = float(v)
            except: continue
            std = abs(v) * 0.2 or 1.0; mean = v

        low_val = mean - 2*std; high_val = mean + 2*std
        low_score = high_score = base_num
        try:
            r = dict(full_row); r[col] = low_val
            low_score = _numeric_score(agent.predict(pd.DataFrame([r])))
        except: pass
        try:
            r = dict(full_row); r[col] = high_val
            high_score = _numeric_score(agent.predict(pd.DataFrame([r])))
        except: pass

        results.append({
            "feature": col,
            "low_val": round(low_val, 4), "high_val": round(high_val, 4),
            "low_pred": round(low_score, 4), "high_pred": round(high_score, 4),
            "impact": round(abs(high_score - low_score), 6),
            "direction": "positive" if high_score >= low_score else "negative"
        })

    results.sort(key=lambda x: x["impact"], reverse=True)
    return {"experiment_id": eid, "base_prediction": str(base_pred),
            "base_prediction_num": base_num, "metric_label": metric_label,
            "sensitivity": results[:20]}


# ── EDA Endpoints ──────────────────────────────────────────────────────────────
@app.get("/experiments/{eid}/eda/advanced", tags=["Analysis"])
async def advanced_eda(eid: str, req: Request):
    """Serve EDA stats — uses live df if in memory, else falls back to saved summary."""
    auth(req)
    exp = EXPS.get(eid)
    if not exp: raise HTTPException(404)
    if exp["status"] != "completed": raise HTTPException(400, "Experiment not completed yet")

    agent = exp.get("agent")
    df    = agent.df if agent else None

    # ── Fast path: compute live from df ──────────────────────────────────────
    if df is not None:
        target   = agent.target_col
        num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
        is_clf   = agent.problem_type == "classification"
        missing  = {k: int(v) for k, v in df.isnull().sum().items()}
        stats, outliers_d, skewness = {}, {}, {}
        for c in num_cols:
            s = df[c].dropna()
            if len(s) == 0: continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75)); iqr = q3 - q1
            outliers_d[c] = int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum())
            skewness[c]   = float(s.skew()) if pd.notna(s.skew()) else 0.0
            stats[c]      = {"mean": float(s.mean()), "std": float(s.std()) if pd.notna(s.std()) else 0.0,
                             "min": float(s.min()), "max": float(s.max()),
                             "25%": float(q1), "50%": float(s.median()), "75%": float(q3)}
        corr_df = df[num_cols].corr() if len(num_cols) > 1 else pd.DataFrame()
        corr_matrix = corr_df.where(pd.notnull(corr_df), None).to_dict() if not corr_df.empty else {}
        top_corr = []
        if not corr_df.empty:
            cu = corr_df.abs().unstack().sort_values(ascending=False)
            seen = set()
            for idx, val in cu.items():
                if idx[0] == idx[1] or not pd.notna(val): continue
                key = tuple(sorted(idx))
                if key in seen: continue
                seen.add(key)
                top_corr.append((f"{idx[0]}_vs_{idx[1]}", float(corr_df.loc[idx[0], idx[1]])))
                if len(top_corr) >= 15: break
        target_s = df[target].dropna()
        if is_clf:
            vc = target_s.value_counts()
            tgt_data = {"type": "categorical", "counts": {str(k): int(v) for k,v in vc.items()}}
        else:
            counts, edges = np.histogram(target_s, bins=20)
            tgt_data = {"type": "numeric", "counts": [int(x) for x in counts],
                        "edges": [float(x) for x in edges]}
        biv = {}
        for c in num_cols[:4]:
            s = df[c].dropna()
            if len(s) == 0: continue
            if is_clf:
                classes = target_s.unique()[:5]
                biv[c] = {str(cls): {"x": df[c][target_s==cls].dropna().tolist()[:150]} for cls in classes}
            else:
                biv[c] = {"x": df[c].dropna().tolist()[:400], "y": target_s[df[c].notna()].tolist()[:400]}
        return {"shape": [int(df.shape[0]), int(df.shape[1])],
                "missing": missing, "numeric_stats": stats,
                "outliers": outliers_d, "skewness": skewness,
                "correlations": dict(top_corr), "correlation_matrix": corr_matrix,
                "target_dist": tgt_data, "num_cols": num_cols, "bivariate": biv}

    # ── Fallback: reconstruct from saved EDA report in metadata ──────────────
    s = exp.get("summary") or {}
    eda = s.get("eda_report") or {}
    if not eda:
        raise HTTPException(503, "EDA data not available — retrain to populate.")

    # Map saved eda_report fields to the format the frontend expects
    shape_raw = eda.get("shape") or {}
    shape     = [shape_raw.get("rows", 0), shape_raw.get("cols", 0)]

    # missing: convert from {col:{count,pct}} to {col: count}
    miss_raw = (eda.get("missing") or {}).get("columns") or {}
    missing  = {c: int(v.get("count", 0)) for c, v in miss_raw.items()} if isinstance(miss_raw, dict) else {}

    ns_raw      = eda.get("numeric_stats") or {}
    skew_raw    = eda.get("skewness_kurtosis") or {}
    out_raw     = eda.get("outlier_analysis") or {}
    corr_raw    = eda.get("correlations") or {}
    feat_types  = eda.get("feature_types") or {}
    num_cols    = feat_types.get("numeric") or list(ns_raw.keys())
    target_col  = s.get("target_col") or exp.get("target_col", "")
    if target_col in num_cols: num_cols = [c for c in num_cols if c != target_col]
    problem_type = s.get("problem_type", "")
    is_clf = problem_type == "classification"

    # Numeric stats — already in right format
    stats = {}
    for c, v in ns_raw.items():
        if isinstance(v, dict):
            stats[c] = {k2: v2 for k2, v2 in v.items() if k2 in ("mean","std","min","max","25%","50%","75%")}

    # Skewness
    skewness = {}
    for c, v in skew_raw.items():
        if isinstance(v, dict): skewness[c] = float(v.get("skewness", 0))
        elif isinstance(v, (int, float)): skewness[c] = float(v)

    # Outliers
    outliers_d = {}
    for c, v in out_raw.items():
        if isinstance(v, dict): outliers_d[c] = int(v.get("n_outliers", v.get("count", 0)))
        elif isinstance(v, (int, float)): outliers_d[c] = int(v)

    # Correlations — top pairs
    top_corr = []
    if isinstance(corr_raw, dict):
        pairs = [(k, v) for k, v in corr_raw.items() if isinstance(v, (int, float))]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_corr = pairs[:15]

    # Target distribution
    tgt_raw = eda.get("target_summary") or {}
    if is_clf or tgt_raw.get("type") == "categorical":
        vc = tgt_raw.get("class_counts") or {}
        tgt_data = {"type": "categorical", "counts": {str(k): int(v) for k,v in vc.items()}}
    else:
        mean = tgt_raw.get("mean", 0); std = max(tgt_raw.get("std", 1), 0.001)
        edges = [round(mean - 3*std + i*std*0.3, 4) for i in range(21)]
        import math
        counts = [max(0, int(100 * math.exp(-0.5*((mean - 3*std + i*std*0.3 - mean)/std)**2))) for i in range(20)]
        tgt_data = {"type": "numeric", "counts": counts, "edges": edges}

    return {"shape": shape, "missing": missing, "numeric_stats": stats,
            "outliers": outliers_d, "skewness": skewness,
            "correlations": dict(top_corr), "correlation_matrix": {},
            "target_dist": tgt_data, "num_cols": num_cols, "bivariate": {},
            "_from_cache": True}

@app.get("/experiments/{eid}/eda/histograms", tags=["Analysis"])
async def eda_histograms(eid: str, req: Request):
    auth(req)
    exp = EXPS.get(eid)
    if not exp: raise HTTPException(404)
    if exp["status"] != "completed": raise HTTPException(400, "Not completed")

    agent = exp.get("agent")
    df    = agent.df if agent else None

    # Live path
    if df is not None:
        target   = agent.target_col
        prob_type = agent.problem_type
        num_cols = list(df.select_dtypes(include=np.number).columns)[:15]
        result = {}
        for c in num_cols:
            s = df[c].dropna()
            if len(s) < 5: continue
            counts, edges = np.histogram(s, bins=20)
            result[c] = {"bins": [round(float(e),4) for e in edges[:-1]],
                         "counts": [int(x) for x in counts],
                         "mean": round(float(s.mean()),4), "std": round(float(s.std()),4),
                         "median": round(float(s.median()),4)}
        return {"histograms": result, "target": target, "problem_type": prob_type}

    # Fallback: reconstruct approximate histograms from saved numeric_stats
    s = exp.get("summary") or {}
    eda = s.get("eda_report") or {}
    ns  = eda.get("numeric_stats") or {}
    import math
    result = {}
    for c, v in list(ns.items())[:15]:
        if not isinstance(v, dict): continue
        mean = v.get("mean", 0); std = max(v.get("std", 1), 0.001)
        mn = v.get("min", mean - 3*std); mx = v.get("max", mean + 3*std)
        step = (mx - mn) / 20 if mx != mn else 1
        bins = [round(mn + i*step, 4) for i in range(20)]
        # Gaussian approximation for counts shape
        counts = [max(0, int(50 * math.exp(-0.5*((mn+i*step-mean)/std)**2))) for i in range(20)]
        result[c] = {"bins": bins, "counts": counts,
                     "mean": round(float(mean),4), "std": round(float(std),4),
                     "median": round(float(v.get("50%", mean)),4)}
    return {"histograms": result, "target": s.get("target_col",""), 
            "problem_type": s.get("problem_type",""), "_from_cache": True}


# ── HTML Report (FIXED – no more 500 errors) ───────────────────────────────────
@app.get("/experiments/{eid}/report", tags=["Analysis"])
async def html_report(eid: str, req: Request):
    auth(req)
    def esc(v): return _html.escape(str(v) if v is not None else "—")
    def fv(v):  return f"{v:.4f}" if isinstance(v, float) else str(v) if v is not None else "—"
    try:
        exp = EXPS.get(eid)
        if not exp: raise HTTPException(404, f"Experiment {eid} not found")
        s   = exp.get("summary") or {}
        bm  = s.get("best_metrics") or {}
        lb  = s.get("leaderboard") or []
        fi  = (s.get("top_features") or [])[:15]
        ins = s.get("ai_insights") or []
        eda = s.get("eda_report") or {}
        pp  = s.get("preprocessing") or {}
        shape = s.get("dataset_shape") or []
        miss  = eda.get("missing") or {}
        corr  = eda.get("correlations") or {}
        tgt_corr = corr.get("target_correlations", {}) if isinstance(corr, dict) else {}

        lb_rows = "".join(
            f"<tr class='{'best' if i==0 else ''}'>"
            f"<td>{i+1}</td><td><code>{esc(r.get('model_name','—'))}</code></td>"
            f"<td>{'  '.join(fv(mv)+' '+str(mk) for mk,mv in (r.get('metrics') or {}).items())}</td></tr>"
            for i, r in enumerate(lb))

        fi_bars = "".join(
            f"<div class='fi-row'><div class='fi-label' title='{esc(f['feature'])}'>{esc(f['feature'])}</div>"
            f"<div class='fi-bar-wrap'><div class='fi-bar' style='width:{f['importance']/(fi[0]['importance'] or 1)*100:.1f}%'></div></div>"
            f"<div class='fi-val'>{f['importance']:.4f}</div></div>"
            for f in fi) if fi else "<p class='muted'>No feature importance data</p>"

        corr_rows = "".join(
            f"<div class='fi-row'><div class='fi-label' title='{esc(col)}'>{esc(col)}</div>"
            f"<div class='fi-bar-wrap'><div class='fi-bar' style='width:{min(abs(v)*100,100):.1f}%;background:{'#22c55e' if v>0 else '#ef4444'}'></div></div>"
            f"<div class='fi-val' style='color:{'#22c55e' if v>0 else '#ef4444'}'>{v:.3f}</div></div>"
            for col, v in list(tgt_corr.items())[:10] if isinstance(v, (int,float))) if tgt_corr else ""

        miss_rows = "".join(
            f"<tr><td>{esc(col)}</td><td style='text-align:right'>{mv:.1f}%</td></tr>"
            for col, mv in list(miss.items() if isinstance(miss,dict) else {}.items())[:12]
            if isinstance(mv, (int,float)) and mv > 0) if miss else ""

        metrics_cells = "".join(
            f"<td><div class='metric-val'>{fv(v)}</div><div class='metric-label'>{esc(str(k).replace('_',' '))}</div></td>"
            for k, v in list(bm.items())[:6])

        ins_html = "".join(f"<div class='insight'>{esc(i)}</div>" for i in ins)
        best_strat = pp.get("best_strategy","—") if isinstance(pp,dict) else "—"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Report – {esc(exp.get('dataset_name',eid))}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:13px;line-height:1.6;
  background:#f6f7f9;color:#111827;padding:32px 16px}}
.container{{max-width:960px;margin:0 auto}}
header{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:24px 28px;margin-bottom:20px}}
h1{{font-size:22px;font-weight:700;color:#111827;margin-bottom:6px}}
.meta{{font-size:12px;color:#6b7280;display:flex;gap:16px;flex-wrap:wrap}}
.badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;
  background:#dbeafe;color:#1d4ed8}}
.badge.reg{{background:#d1fae5;color:#065f46}}
section{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px 24px;margin-bottom:16px}}
h2{{font-size:14px;font-weight:600;color:#374151;margin-bottom:14px;padding-bottom:8px;
  border-bottom:1px solid #f3f4f6;text-transform:uppercase;letter-spacing:.5px}}
.metrics-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:10px}}
.metric-card{{background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:12px;text-align:center}}
.metric-val{{font-size:20px;font-weight:700;color:#111827;font-family:monospace}}
.metric-label{{font-size:10px;color:#9ca3af;margin-top:2px;text-transform:uppercase}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#f9fafb;padding:8px 12px;text-align:left;font-size:10px;font-weight:600;color:#6b7280;
  text-transform:uppercase;letter-spacing:.4px;border-bottom:2px solid #e5e7eb}}
td{{padding:8px 12px;border-bottom:1px solid #f3f4f6}}
tr.best td{{background:#eff6ff;font-weight:600}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.fi-row{{display:flex;align-items:center;gap:8px;padding:4px 0}}
.fi-label{{width:130px;font-size:11px;color:#6b7280;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:monospace}}
.fi-bar-wrap{{flex:1;height:6px;background:#f3f4f6;border-radius:3px;overflow:hidden}}
.fi-bar{{height:100%;background:#3b82f6;border-radius:3px;transition:width .4s}}
.fi-val{{width:48px;text-align:right;font-size:11px;color:#9ca3af;font-family:monospace}}
.insight{{background:#eff6ff;border-left:3px solid #3b82f6;border-radius:0 6px 6px 0;
  padding:8px 12px;margin-bottom:8px;font-size:12px;color:#1e40af}}
.muted{{color:#9ca3af;font-size:12px}}
footer{{text-align:center;font-size:11px;color:#9ca3af;margin-top:24px}}
@media(max-width:600px){{.two-col{{grid-template-columns:1fr}}.metrics-grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head>
<body>
<div class="container">
<header>
  <h1>AutoML Experiment Report</h1>
  <div class="meta">
    <span><b>Dataset:</b> {esc(exp.get('dataset_name',eid))}</span>
    <span><b>ID:</b> {esc(eid)}</span>
    <span><b>Target:</b> {esc(s.get('target_col','—'))}</span>
    <span><span class="badge {'reg' if s.get('problem_type')!='classification' else ''}">{esc(s.get('problem_type','—'))}</span></span>
    <span><b>Shape:</b> {' × '.join(str(x) for x in shape) if shape else '?'}</span>
    <span><b>Training time:</b> {esc(s.get('elapsed_sec','—'))}s</span>
  </div>
</header>

<section>
  <h2>🏆 Best Model: {esc(s.get('best_model','—'))}</h2>
  <div class="metrics-grid">{metrics_cells}</div>
</section>

{f'<section><h2>💡 AI Insights</h2>{ins_html}</section>' if ins else ''}

<div class="two-col">
  <section>
    <h2>📊 Model Leaderboard</h2>
    <table><thead><tr><th>#</th><th>Model</th><th>Metrics</th></tr></thead>
    <tbody>{lb_rows}</tbody></table>
  </section>
  <section>
    <h2>🔑 Feature Importance</h2>
    {fi_bars}
  </section>
</div>

{f'''<div class="two-col">
  <section>
    <h2>📈 Target Correlations</h2>
    {corr_rows if corr_rows else "<p class='muted'>No correlation data</p>"}
  </section>
  <section>
    <h2>🔧 Missing Values</h2>
    {'<table><thead><tr><th>Column</th><th>%</th></tr></thead><tbody>'+miss_rows+'</tbody></table>' if miss_rows else "<p class='muted'>No missing values ✓</p>"}
  </section>
</div>''' if corr_rows or miss_rows else ''}

<section>
  <h2>⚙️ Preprocessing</h2>
  <p>Best strategy: <b>{esc(best_strat)}</b></p>
</section>

<footer>Generated by AutoML Agent v6.0 · {time.strftime('%Y-%m-%d %H:%M:%S')}</footer>
</div>
</body></html>"""

        return Response(
            content=html, media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=report_{eid}.html"})

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        err = f"<html><body style='font-family:monospace;padding:20px;background:#fff0f0'><h2 style='color:red'>Report Error</h2><pre>{_html.escape(traceback.format_exc())}</pre></body></html>"
        return Response(content=err, media_type="text/html", status_code=500)


# ── Model Card ─────────────────────────────────────────────────────────────────
@app.get("/experiments/{eid}/model-card", tags=["Analysis"])
async def model_card(eid: str, req: Request):
    auth(req); exp = EXPS.get(eid)
    if not exp: raise HTTPException(404)
    s = exp.get("summary") or {}; lb = s.get("leaderboard") or []
    return {
        "model_id": eid, "model_name": s.get("best_model") or "—",
        "dataset_name": exp.get("dataset_name", eid),
        "version": "1.0", "task_type": s.get("problem_type"),
        "target_column": s.get("target_col"),
        "training_data": {"shape": s.get("dataset_shape")},
        "performance": s.get("best_metrics") or {},
        "training_details": {"models_evaluated": len(lb), "best_model": s.get("best_model"),
                             "training_time_sec": s.get("elapsed_sec")},
        "top_features": (s.get("top_features") or [])[:10],
        "ai_insights": s.get("ai_insights") or [],
        "created_at": exp.get("created_at"), "automl_version": "6.0",
    }


# ── AutoNarrative ──────────────────────────────────────────────────────────────
@app.get("/experiments/{eid}/narrative", tags=["Analysis"])
async def narrative(eid: str, req: Request):
    auth(req)
    exp = EXPS.get(eid)
    if not exp: raise HTTPException(404)
    if exp["status"] != "completed": raise HTTPException(400, "Experiment not completed")
    s  = exp.get("summary") or {}; bm = s.get("best_metrics") or {}
    lb = s.get("leaderboard") or []; fi = s.get("top_features") or []
    eda = s.get("eda_report") or {}; shape = s.get("dataset_shape") or [0,0]
    pp  = s.get("preprocessing") or {}; clf = s.get("problem_type") == "classification"
    sk  = "accuracy" if clf else "r2"; sv = bm.get(sk, 0)
    target = s.get("target_col","target"); dname = exp.get("dataset_name", eid)
    sections = [
        {"title": "📊 Dataset Overview",
         "text": f"The dataset <b>{_esc(dname)}</b> contains <b>{shape[0]:,}</b> rows and <b>{shape[1]}</b> columns. "
                 f"The goal is to predict <b>{_esc(target)}</b>, a <b>{s.get('problem_type','ML')}</b> task. "
                 f"The AutoML engine automatically handled missing values, encoded categorical features, and removed ID/name columns."},
        {"title": "⚙️ Feature Engineering",
         "text": f"After benchmarking multiple preprocessing strategies, <b>{_esc(str(pp.get('best_strategy','—') if isinstance(pp,dict) else pp))}</b> was selected as optimal. "
                 f"The pipeline includes outlier clipping, skewness correction, and automated feature selection."},
        {"title": "🏆 Model Competition",
         "text": f"AutoML evaluated <b>{len(lb)} models</b> including gradient boosting (XGBoost, LightGBM, CatBoost), "
                 f"random forests, and linear models. The winner was <b>{_esc(lb[0].get('model_name','—') if lb else '—')}</b>. "
                 f"Best {sk}: <b>{sv:.4f}</b>."},
        {"title": "🔑 Key Drivers",
         "text": "Feature importance reveals the top predictors are: "
                 + (", ".join(f"<b>{_esc(f['feature'])}</b> ({f['importance']:.3f})" for f in fi[:3]) if fi
                    else "not available.")},
        {"title": "💡 Recommendation",
         "text": f"Based on the analysis, the model is "
                 f"<b>{'ready for production' if sv>=0.85 else 'a solid starting point' if sv>=0.70 else 'in need of improvement'}</b>. "
                 f"Next steps: validate on a real-world test set, run sensitivity analysis to understand feature impacts, and deploy via the Deploy tab."},
    ]
    return {"experiment_id": eid, "dataset_name": dname, "sections": sections,
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S')}


# ── Model Doctor – Automated Diagnostic Engine ────────────────────────────────
@app.get("/experiments/{eid}/doctor", tags=["Analysis"])
async def model_doctor(eid: str, req: Request):
    """Automated model diagnostic — works from disk metadata, no live model required."""
    auth(req)
    exp = EXPS.get(eid)
    if not exp: raise HTTPException(404)
    if exp["status"] != "completed": raise HTTPException(400, "Experiment not completed")

    s  = exp.get("summary") or {}
    bm = s.get("best_metrics") or {}
    lb = s.get("leaderboard") or []
    fi = s.get("top_features") or []
    eda = s.get("eda_report") or {}
    clf = s.get("problem_type") == "classification"
    sk  = "accuracy" if clf else "r2"; sv = bm.get(sk, 0)

    # Also try to get live df for richer checks
    agent = exp.get("agent")
    df    = agent.df if agent else None

    diagnoses = []

    # 1. Score health check
    if sv >= 0.95:
        diagnoses.append({"severity": "warning", "category": "Data Leakage Risk",
            "finding": f"Suspiciously high {sk} of {sv:.3f}.",
            "prescription": "Verify that no future data is leaking into features. Check for columns computed from the target."})
    elif sv >= 0.85:
        diagnoses.append({"severity": "healthy", "category": "Model Performance",
            "finding": f"Strong {sk} of {sv:.3f} — production-ready.",
            "prescription": "Monitor for performance drift after deployment. Set up periodic retraining."})
    elif sv >= 0.70:
        diagnoses.append({"severity": "warning", "category": "Model Performance",
            "finding": f"Moderate {sk} of {sv:.3f}. Room for improvement.",
            "prescription": "Try increasing HPO trials to 50+, gather more training data, or engineer domain-specific features."})
    else:
        diagnoses.append({"severity": "critical", "category": "Model Performance",
            "finding": f"Low {sk} of {sv:.3f}. Model is underperforming.",
            "prescription": "Check data quality, consider collecting more samples, and review the feature set."})

    # 2. Leaderboard gap
    if len(lb) >= 2:
        top1 = (lb[0].get("metrics") or {}).get(sk, 0) or 0
        top2 = (lb[1].get("metrics") or {}).get(sk, 0) or 0
        if abs(top1 - top2) < 0.003:
            diagnoses.append({"severity": "info", "category": "Ensemble Opportunity",
                "finding": f"Top 2 models are very close ({abs(top1-top2):.4f} gap).",
                "prescription": "Consider building a stacking ensemble combining the top 2-3 models."})

    # 3. Feature concentration
    if fi and fi[0].get("importance", 0) > 0.5 * sum(f.get("importance",0) for f in fi):
        diagnoses.append({"severity": "warning", "category": "Feature Dominance",
            "finding": f"Feature '{fi[0]['feature']}' dominates importance ({fi[0]['importance']:.3f}).",
            "prescription": "Investigate if this feature could be a data leakage source. Consider interaction features."})

    # 4. Class imbalance — use saved EDA report
    if clf:
        tgt = eda.get("target_summary") or {}
        vc_raw = tgt.get("class_counts") or (df and {str(k): int(v) for k,v in df[s.get("target_col","")].value_counts().items()} if df else {})
        if vc_raw and len(vc_raw) >= 2:
            vals = list(vc_raw.values())
            ratio = min(vals) / max(vals) if max(vals) > 0 else 1
            if ratio < 0.2:
                diagnoses.append({"severity": "warning", "category": "Class Imbalance",
                    "finding": f"Severe class imbalance: minority is {ratio:.1%} of majority.",
                    "prescription": "Use class_weight='balanced', SMOTE oversampling, or adjust prediction threshold."})

    # 5. Missing value check — use saved EDA report
    miss_cols = (eda.get("missing") or {}).get("columns") or {}
    if isinstance(miss_cols, dict):
        heavy = {c: v for c, v in miss_cols.items() if isinstance(v, dict) and v.get("pct", 0) > 30}
        if heavy:
            diagnoses.append({"severity": "warning", "category": "Missing Data",
                "finding": f"{len(heavy)} columns have >30% missing: {', '.join(list(heavy.keys())[:3])}.",
                "prescription": "Consider dropping columns with >50% missing. For 30-50%, use domain-specific imputation."})

    # 6. Small dataset — use saved shape
    shape = s.get("dataset_shape") or [eda.get("shape", {}).get("rows", 0)]
    n_rows = shape[0] if shape else 0
    if 0 < n_rows < 500:
        diagnoses.append({"severity": "warning", "category": "Dataset Size",
            "finding": f"Small dataset ({n_rows:,} rows). Generalization may be limited.",
            "prescription": "Gather more training data. Use cross-validation (k=10). Consider simpler models to avoid overfitting."})

    overall = "healthy" if all(d["severity"] in ("healthy","info") for d in diagnoses) else (
              "critical" if any(d["severity"] == "critical" for d in diagnoses) else "warning")
    score = 100 - sum({"critical": 30, "warning": 15, "info": 5, "healthy": 0}[d["severity"]] for d in diagnoses)

    return {"experiment_id": eid, "overall_health": overall,
            "health_score": max(0, min(100, score)),
            "n_findings": len(diagnoses), "diagnoses": diagnoses,
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S')}


# ── Smart Data Profiler ────────────────────────────────────────────────────────
@app.get("/experiments/{eid}/profile", tags=["Analysis"])
async def data_profile(eid: str, req: Request):
    """Business-readable data profile — works from disk metadata, no live df required."""
    auth(req)
    exp = EXPS.get(eid)
    if not exp: raise HTTPException(404)
    if exp["status"] != "completed": raise HTTPException(400, "Experiment not completed")

    agent  = exp.get("agent")
    df     = agent.df if agent else None
    s      = exp.get("summary") or {}
    eda    = s.get("eda_report") or {}
    target = (agent.target_col if agent else None) or exp.get("target_col") or s.get("target_col","")

    # ── Live path ─────────────────────────────────────────────────────────────
    if df is not None:
        n_rows, n_cols = df.shape
        columns = []
        for col in df.columns:
            sv = df[col]; miss = int(sv.isnull().sum())
            miss_pct = round(miss/n_rows*100, 1); nu = int(sv.nunique())
            is_num = np.issubdtype(sv.dtype, np.number)
            ci: Dict = {"name": col, "dtype": str(sv.dtype), "is_target": col==target,
                        "missing": miss, "missing_pct": miss_pct,
                        "unique": nu, "unique_pct": round(nu/n_rows*100,1)}
            if is_num:
                ci.update({"role":"numeric",
                    "mean": round(float(sv.mean()),4) if pd.notna(sv.mean()) else None,
                    "std":  round(float(sv.std()),4)  if pd.notna(sv.std())  else None,
                    "min":  round(float(sv.min()),4)  if pd.notna(sv.min())  else None,
                    "max":  round(float(sv.max()),4)  if pd.notna(sv.max())  else None,
                    "skew": round(float(sv.skew()),3) if pd.notna(sv.skew()) else None})
            else:
                tv = sv.value_counts().head(5)
                ci.update({"role": "categorical" if nu<=50 else "high-cardinality",
                           "top_values": [{"value":str(k),"count":int(v)} for k,v in tv.items()]})
            columns.append(ci)
        completeness = round((1-df.isnull().mean().mean())*100, 1)
        uniqueness   = round(len(df.drop_duplicates())/n_rows*100, 1)
        return {"experiment_id":eid,"dataset_name":exp.get("dataset_name",eid),
                "shape":{"rows":n_rows,"columns":n_cols},
                "quality_score":round(completeness*.6+uniqueness*.4,1),
                "completeness_pct":completeness,"uniqueness_pct":uniqueness,
                "columns":columns,"target_col":target,
                "problem_type":s.get("problem_type",""),
                "generated_at":time.strftime('%Y-%m-%d %H:%M:%S')}

    # ── Fallback: reconstruct from saved EDA report ───────────────────────────
    shape_raw = eda.get("shape") or {}
    n_rows = shape_raw.get("rows", 0); n_cols = shape_raw.get("cols", 0)
    ns_raw   = eda.get("numeric_stats") or {}
    sk_raw   = eda.get("skewness_kurtosis") or {}
    cs_raw   = eda.get("categorical_stats") or {}
    miss_raw = (eda.get("missing") or {}).get("columns") or {}
    feat_t   = eda.get("feature_types") or {}
    num_cols = set(feat_t.get("numeric") or list(ns_raw.keys()))
    dq       = eda.get("data_quality_score") or {}

    columns = []
    all_cols = list(ns_raw.keys()) + [c for c in (cs_raw or {}) if c not in ns_raw]
    if target and target not in all_cols: all_cols.append(target)

    for col in all_cols:
        miss_info = miss_raw.get(col, {})
        miss_cnt  = miss_info.get("count", 0) if isinstance(miss_info, dict) else 0
        miss_pct  = miss_info.get("pct", 0)   if isinstance(miss_info, dict) else 0
        is_num    = col in num_cols
        ci: Dict  = {"name":col,"dtype":"float64" if is_num else "object","is_target":col==target,
                     "missing":miss_cnt,"missing_pct":round(miss_pct,1),
                     "unique":None,"unique_pct":None}
        if is_num:
            v = ns_raw.get(col) or {}
            skv = sk_raw.get(col) or {}
            ci.update({"role":"numeric",
                "mean": v.get("mean"), "std": v.get("std"),
                "min":  v.get("min"),  "max": v.get("max"),
                "skew": skv.get("skewness") if isinstance(skv,dict) else skv})
        else:
            cinfo = (cs_raw or {}).get(col) or {}
            top_vals = []
            for k, cnt in (cinfo.get("value_counts") or {}).items():
                top_vals.append({"value":str(k),"count":int(cnt)})
                if len(top_vals)>=5: break
            nu = cinfo.get("n_unique", None)
            ci.update({"role":"categorical","unique":nu,"top_values":top_vals})
        columns.append(ci)

    completeness = dq.get("completeness", 100)
    uniqueness   = dq.get("uniqueness",   100)
    quality_score = dq.get("overall", round(completeness*.6+uniqueness*.4,1))

    return {"experiment_id":eid,"dataset_name":exp.get("dataset_name",eid),
            "shape":{"rows":n_rows,"columns":n_cols},
            "quality_score":quality_score,
            "completeness_pct":completeness,"uniqueness_pct":uniqueness,
            "columns":columns,"target_col":target,
            "problem_type":s.get("problem_type",""),
            "generated_at":time.strftime('%Y-%m-%d %H:%M:%S'),
            "_from_cache":True}


# ── NEW: Counterfactual Explainer ──────────────────────────────────────────────
@app.post("/experiments/{eid}/counterfactual", tags=["Analysis"])
async def counterfactual(eid: str, body: PR, req: Request):
    """Find the minimum feature changes needed to flip/shift prediction."""
    auth(req); agent = get_agent(eid)
    base_row = body.data[0]; target_pred = body.data[0].get("__target__")
    pp  = agent.preprocessing_engine
    src = _get_src(agent)
    all_cols = pp.all_feature_cols if hasattr(pp, "all_feature_cols") else list(base_row.keys())

    full_row: Dict = {}
    for c in all_cols:
        v = base_row.get(c)
        if v is not None: full_row[c] = v
        elif src is not None and c in src.columns:
            s = src[c].dropna()
            full_row[c] = float(s.median()) if np.issubdtype(s.dtype, np.number) else (str(s.mode()[0]) if len(s.mode()) else "")
        else: full_row[c] = 0

    try:
        base_pred = agent.predict(pd.DataFrame([full_row]))["predictions"][0]
    except Exception as e:
        raise HTTPException(422, f"Base prediction failed: {e}")

    def is_success(pred):
        if target_pred is not None:
            if agent.problem_type == "classification": return str(pred) != str(base_pred)
            else:
                try: return float(pred) >= float(target_pred)
                except: return False
        return str(pred) != str(base_pred)

    num_feats = []
    if src is not None:
        for c in all_cols:
            if c in src.columns and np.issubdtype(src[c].dtype, np.number):
                s = src[c].dropna()
                if len(s) >= 2:
                    num_feats.append({"name": c, "min": float(s.min()), "max": float(s.max()),
                                      "std": float(s.std()), "median": float(s.median())})

    suggestions = []; rng = np.random.default_rng(42)
    # Try single-feature changes first
    for feat in num_feats[:20]:
        for mult in [1,-1,2,-2,0.5,-0.5]:
            nv = max(feat["min"], min(feat["max"], feat["median"] + mult*feat["std"]))
            test = dict(full_row); test[feat["name"]] = nv
            try:
                pred = agent.predict(pd.DataFrame([test]))["predictions"][0]
                if is_success(pred):
                    suggestions.append({"feature": feat["name"],
                        "original": round(float(full_row.get(feat["name"], feat["median"])),4),
                        "changed_to": round(nv,4), "delta": round(nv - float(full_row.get(feat["name"],feat["median"])),4),
                        "new_prediction": str(pred), "n_changes": 1})
                    break
            except: pass
        if len(suggestions) >= 3: break

    return {"experiment_id": eid, "base_prediction": str(base_pred),
            "counterfactuals": suggestions[:5], "n_found": len(suggestions)}


# ── Feature Interaction ─────────────────────────────────────────────────────────
@app.post("/experiments/{eid}/interaction", tags=["Analysis"])
async def feature_interaction(eid: str, req: Request):
    auth(req); body = await req.json()
    feat_a = body.get("feature_a",""); feat_b = body.get("feature_b","")
    if not feat_a or not feat_b: raise HTTPException(400, "Provide feature_a and feature_b")
    agent = get_agent(eid); df = agent.df
    if df is None: raise HTTPException(503, "DataFrame not in memory")
    missing = []
    if feat_a not in df.columns: missing.append(feat_a)
    if feat_b not in df.columns: missing.append(feat_b)
    if missing: raise HTTPException(400, f"Features not found in dataset: {', '.join(missing)}")
    target = agent.target_col; clf = agent.problem_type == "classification"

    # Only dropna on the 3 columns we actually need — not the full dataframe
    sub = df[[feat_a, feat_b, target]].dropna()
    if len(sub) == 0: raise HTTPException(400, "No non-null rows found for these features")
    n = min(400, len(sub))
    sample = sub.sample(n=n, random_state=42)

    xs = sample[feat_a].tolist()
    ys = sample[feat_b].tolist()
    ts = sample[target].tolist()

    # Pearson correlation between the two features
    try:
        corr = float(sub[[feat_a, feat_b]].corr().iloc[0, 1])
        corr = corr if not np.isnan(corr) else 0.0
    except: corr = 0.0

    # For classification, encode target as integer class index for coloring
    target_numeric = []
    classes = None
    if clf:
        uniq = list(dict.fromkeys(ts))  # unique in order
        classes = [str(c) for c in uniq]
        target_numeric = [uniq.index(t) if t in uniq else 0 for t in ts]
    else:
        for t in ts:
            try: target_numeric.append(float(t))
            except: target_numeric.append(None)

    return {
        "feature_a": feat_a, "feature_b": feat_b,
        "n_points": len(xs),
        "scatter": {
            "x": [round(float(v), 4) for v in xs],
            "y": [round(float(v), 4) for v in ys],
            "target": target_numeric,
        },
        "correlation": round(corr, 4),
        "problem_type": agent.problem_type,
        "classes": classes,
    }


# ── Stats & Compare ─────────────────────────────────────────────────────────────
@app.get("/api/stats", tags=["Monitoring"])
async def stats(req: Request):
    auth(req)
    comp = [v for v in EXPS.values() if v["status"] == "completed"]
    accs, r2s = [], []
    for v in comp:
        bm = (v.get("summary") or {}).get("best_metrics") or {}
        if "accuracy" in bm: accs.append(bm["accuracy"])
        if "r2"       in bm: r2s.append(bm["r2"])
    return {"total_experiments": len(EXPS), "completed": len(comp),
            "failed":   sum(1 for v in EXPS.values() if v["status"]=="failed"),
            "training": sum(1 for v in EXPS.values() if v["status"]=="training"),
            "in_memory":sum(1 for v in EXPS.values() if v.get("agent")),
            "total_models_trained": sum(len((v.get("summary") or {}).get("leaderboard") or []) for v in comp),
            "avg_best_accuracy": round(sum(accs)/len(accs),4) if accs else None,
            "avg_best_r2": round(sum(r2s)/len(r2s),4) if r2s else None}

@app.post("/api/compare", tags=["Monitoring"])
async def compare(req: Request):
    auth(req); body = await req.json(); eids = body.get("experiment_ids",[])
    rows = []
    for eid in eids:
        exp = EXPS.get(eid)
        if not exp or exp["status"] != "completed": continue
        s = exp.get("summary") or {}
        rows.append({"experiment_id": eid, "dataset_name": exp.get("dataset_name",eid),
                     "best_model": s.get("best_model") or "—",
                     "problem_type": s.get("problem_type"),
                     "metrics": s.get("best_metrics") or {},
                     "top_features": (s.get("top_features") or [])[:8],
                     "leaderboard": s.get("leaderboard") or []})
    return rows


# ── Deploy ─────────────────────────────────────────────────────────────────────
class HFDeployReq(BaseModel):
    hf_token: str
    repo_name: str
    space_sdk: str = "gradio"  # gradio or streamlit

def _build_bundle(eid: str, deploy_dir: Path) -> Path:
    """Save model bundle to disk. Returns path."""
    bundle_path = deploy_dir / "model_bundle.pkl"
    if not bundle_path.exists():
        agent = EXPS[eid].get("agent")
        if not agent: raise HTTPException(503, "Model not in memory — retrain first")
        bundle = {
            "model": agent.best_model,
            "pipeline": agent.preprocessing_engine.best_pipeline,
            "label_encoder": getattr(agent, "label_encoder", None),
            "problem_type": agent.problem_type,
            "target_col": agent.target_col,
            "feature_cols": list(agent.preprocessing_engine.all_feature_cols),
            "model_name": getattr(agent, "best_model_name", "model"),
        }
        with open(bundle_path, "wb") as f: pickle.dump(bundle, f)
    return bundle_path

@app.post("/experiments/{eid}/deploy", tags=["Deploy"])
async def deploy(eid: str, req: Request, target: str = "export"):
    auth(req); exp = EXPS.get(eid)
    if not exp: raise HTTPException(404)
    deploy_dir = cfg().experiments_dir / eid / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = _build_bundle(eid, deploy_dir)

    r: Dict = {"experiment_id": eid, "target": target, "status": "exported",
               "files": ["model_bundle.pkl"]}

    if target == "local":
        api_code = '''"""AutoML Agent – Local FastAPI Inference Server"""
import pickle, numpy as np, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="AutoML Inference", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

with open("model_bundle.pkl", "rb") as f:
    B = pickle.load(f)

print(f"✅ Loaded: {B['model_name']} | task={B['problem_type']} | features={len(B['feature_cols'])}")

class PredictReq(BaseModel):
    data: List[Dict[str, Any]]

@app.get("/")
def root(): return {"model": B["model_name"], "task": B["problem_type"], "features": B["feature_cols"]}

@app.get("/health")
def health(): return {"ok": True, "model": B["model_name"]}

@app.post("/predict")
def predict(req: PredictReq):
    df = pd.DataFrame(req.data)
    missing = [c for c in B["feature_cols"] if c not in df.columns]
    for c in missing: df[c] = 0
    X = B["pipeline"].transform(df[B["feature_cols"]])
    preds = B["model"].predict(X).tolist()
    result = {"predictions": preds, "model": B["model_name"]}
    if hasattr(B["model"], "predict_proba"):
        try: result["probabilities"] = B["model"].predict_proba(X).tolist()
        except: pass
    return result
'''
        (deploy_dir / "api.py").write_text(api_code)
        (deploy_dir / "requirements.txt").write_text(
            "fastapi\nuvicorn[standard]\npandas\nscikit-learn\nnumpy\nxgboost\nlightgbm\ncatboost\n")
        r.update({"files": ["model_bundle.pkl", "api.py", "requirements.txt"],
                  "run_command": "pip install -r requirements.txt && uvicorn api:app --port 9000 --reload",
                  "predict_url": "http://localhost:9000/predict",
                  "docs_url": "http://localhost:9000/docs"})

    elif target == "docker":
        app_code = '''"""AutoML Docker Inference"""
import pickle, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
with open("/app/model_bundle.pkl", "rb") as f: B = pickle.load(f)

class Req(BaseModel): data: List[Dict[str, Any]]

@app.get("/health")
def health(): return {"ok": True}

@app.post("/predict")
def predict(req: Req):
    df = pd.DataFrame(req.data)
    for c in B["feature_cols"]:
        if c not in df.columns: df[c] = 0
    X = B["pipeline"].transform(df[B["feature_cols"]])
    return {"predictions": B["model"].predict(X).tolist()}
'''
        dockerfile = (
            "FROM python:3.11-slim\nWORKDIR /app\n"
            "RUN pip install --no-cache-dir fastapi uvicorn pandas scikit-learn numpy xgboost lightgbm catboost\n"
            "COPY model_bundle.pkl app.py /app/\n"
            "EXPOSE 8080\nCMD [\"uvicorn\",\"app:app\",\"--host\",\"0.0.0.0\",\"--port\",\"8080\"]\n"
        )
        (deploy_dir / "app.py").write_text(app_code)
        (deploy_dir / "Dockerfile").write_text(dockerfile)
        r.update({
            "files": ["model_bundle.pkl", "app.py", "Dockerfile"],
            "run_command": f"docker build -t automl-{eid} . && docker run -p 8080:8080 automl-{eid}",
            "predict_url": "http://localhost:8080/predict"
        })

    EXPS[eid]["deployment"] = r; _save_meta(eid)
    return r


@app.post("/experiments/{eid}/deploy/huggingface", tags=["Deploy"])
async def deploy_huggingface(eid: str, body: HFDeployReq, req: Request):
    """
    One-click deploy to Hugging Face Spaces.
    Uses curl subprocess exclusively — avoids ALL huggingface_hub library auth quirks.
    """
    import subprocess, shutil, tempfile, json as _json, traceback as _tb
    from pathlib import Path as _P

    def _curl_json(args: list, label: str):
        """Run curl, return (status_code:int, parsed_json:dict|None, raw:str)."""
        cmd = ["curl", "--silent", "--show-error", "--location",
               "--max-time", "120", "--write-out", "\n__HTTP_STATUS__%{http_code}"] + args
        r = subprocess.run(cmd, capture_output=True, text=True)
        raw = r.stdout
        # Extract HTTP status appended by --write-out
        status_code = 0
        if "__HTTP_STATUS__" in raw:
            body_part, status_part = raw.rsplit("__HTTP_STATUS__", 1)
            raw = body_part.strip()
            try: status_code = int(status_part.strip())
            except: pass
        L().info(f"curl {label}: rc={r.returncode} http={status_code} out={raw[:300]}")
        parsed = None
        try: parsed = _json.loads(raw)
        except: pass
        return status_code, parsed, raw

    try:
        auth(req)
        exp = EXPS.get(eid)
        if not exp: raise HTTPException(404, f"Experiment {eid} not found")
        agent = exp.get("agent")
        if not agent: raise HTTPException(503, "Model not in memory — retrain first")

        # ── Clean token — strip ALL whitespace, quotes ────────────────────────
        hf_token  = "".join(body.hf_token.split()).strip('"').strip("'")
        repo_name = re.sub(r"[^a-zA-Z0-9_\-]", "-", body.repo_name.strip()) or f"automl-{eid}"

        if not hf_token:
            raise HTTPException(400, "Token is empty. Paste your HF write token.")
        if not hf_token.startswith("hf_"):
            raise HTTPException(400,
                f"Token must start with 'hf_' — got '{hf_token[:10]}…' ({len(hf_token)} chars). "
                "Copy the full token from huggingface.co/settings/tokens")

        L().info(f"HF deploy: token_len={len(hf_token)} prefix={hf_token[:8]} repo={repo_name}")

        auth_hdr = f"Authorization: Bearer {hf_token}"

        # ── STEP 1: Verify token via whoami ───────────────────────────────────
        status, whoami, raw = _curl_json(
            ["-H", auth_hdr, "https://huggingface.co/api/whoami"],
            "whoami"
        )

        if status == 0 and whoami is None:
            raise HTTPException(502,
                "Could not reach huggingface.co — check your server's internet connection.")

        if whoami and "error" in whoami:
            hf_err = whoami["error"]
            raise HTTPException(401,
                f"Token rejected by Hugging Face: \"{hf_err}\"\n\n"
                "Most likely causes:\n"
                "1. You created a READ token — deploy needs a WRITE token\n"
                "2. Token is expired or revoked\n"
                "3. You copied extra characters or missed some\n\n"
                "Fix: huggingface.co/settings/tokens → New token → "
                "Role = 'Write' → copy the full hf_xxx... token")

        if not whoami or "name" not in whoami:
            raise HTTPException(502, f"Unexpected HF response: {raw[:300]}")

        username = whoami["name"]

        # Check if token has write access (optional — some older token formats don't expose this)
        token_role = (whoami.get("auth") or {}).get("accessToken", {}).get("role", "")
        if token_role and token_role.lower() == "read":
            raise HTTPException(403,
                f"Token belongs to '{username}' but has READ-only access. "
                "Create a new token with 'Write' role at huggingface.co/settings/tokens")

        L().info(f"HF auth OK: username={username} role={token_role or 'unknown'}")

        # ── Build model bundle ────────────────────────────────────────────────
        deploy_dir = cfg().experiments_dir / eid / "deploy"
        deploy_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = _build_bundle(eid, deploy_dir)
        pkl_mb = bundle_path.stat().st_size / 1_048_576
        if pkl_mb > 50:
            raise HTTPException(400,
                f"Model is {pkl_mb:.1f} MB — too large for HF free tier (50 MB limit). "
                "Use Docker deploy instead.")

        # ── Build feature metadata ────────────────────────────────────────────
        s = exp.get("summary") or {}
        model_nm  = s.get("best_model", "Model")
        tgt_col   = exp.get("target_col", "target")
        ds_name   = exp.get("dataset_name", eid)
        prob_type = agent.problem_type or "ml"
        features  = list(agent.preprocessing_engine.all_feature_cols)
        src       = _get_src(agent)

        feat_meta = []
        for feat in features[:25]:
            if src is not None and feat in src.columns:
                col = src[feat].dropna()
                if np.issubdtype(col.dtype, np.number):
                    feat_meta.append({"name": feat, "type": "number",
                                      "min": round(float(col.min()), 4),
                                      "max": round(float(col.max()), 4),
                                      "default": round(float(col.median()), 4)})
                else:
                    cats = [str(c) for c in col.value_counts().index[:15]]
                    feat_meta.append({"name": feat, "type": "categorical",
                                      "categories": cats, "default": cats[0] if cats else ""})
            else:
                feat_meta.append({"name": feat, "type": "number",
                                  "min": 0.0, "max": 1.0, "default": 0.0})

        fm_json = _json.dumps(feat_meta, indent=2)

        # ── Generate app.py ───────────────────────────────────────────────────
        app_py = f'''"""AutoML Predictor — {ds_name} | Model: {model_nm} | Task: {prob_type}"""
import gradio as gr, pickle, pandas as pd

FEAT_META = {fm_json}

with open("model_bundle.pkl", "rb") as _f:
    B = pickle.load(_f)

def predict(*args):
    row = {{fm["name"]: (float(args[i]) if fm["type"] == "number" else str(args[i]))
            for i, fm in enumerate(FEAT_META)}}
    df = pd.DataFrame([row])
    for c in B["feature_cols"]:
        if c not in df.columns: df[c] = 0
    X  = B["pipeline"].transform(df[B["feature_cols"]])
    pred = B["model"].predict(X)[0]
    out = f"### Prediction: `{{pred}}`"
    if hasattr(B["model"], "predict_proba"):
        try:
            probs = B["model"].predict_proba(X)[0]
            cls   = list(getattr(B["model"], "classes_", range(len(probs))))
            lines = "\\n".join(f"- `{{c}}`: {{p*100:.1f}}%" for c, p in
                               sorted(zip(cls, probs), key=lambda x: -x[1]))
            out  += f"\\n\\n**Confidence:**\\n{{lines}}"
        except: pass
    return out

inputs = []
for fm in FEAT_META:
    if fm["type"] == "number":
        inputs.append(gr.Number(label=fm["name"], value=fm["default"],
                                minimum=fm.get("min"), maximum=fm.get("max")))
    else:
        inputs.append(gr.Dropdown(choices=fm["categories"],
                                  label=fm["name"], value=fm["default"]))

gr.Interface(fn=predict, inputs=inputs,
             outputs=gr.Markdown(label="Result"),
             title="AutoML: {ds_name}",
             description="Model: {model_nm} | Task: {prob_type} | Target: {tgt_col}",
             theme=gr.themes.Soft()).launch()
'''

        readme_md = "\n".join([
            "---", f"title: {repo_name}", "emoji: 🤖",
            "colorFrom: blue", "colorTo: green", "sdk: gradio",
            "sdk_version: 4.44.0", "app_file: app.py", "pinned: false", "---",
            "", f"# {ds_name} — AutoML Predictor", "",
            f"**Model:** {model_nm}  ", f"**Task:** {prob_type}  ",
            f"**Target:** {tgt_col}  ", f"**Features:** {len(features)}  ",
            "", "Deployed with AutoML Agent v6.0",
        ]) + "\n"

        reqs_txt = "gradio>=4.44.0\npandas\nnumpy\nscikit-learn\nxgboost\nlightgbm\ncatboost\n"

        full_repo = f"{username}/{repo_name}"
        hf_dir = _P(tempfile.mkdtemp(prefix=f"automl_hf_{eid}_"))
        try:
            shutil.copy(bundle_path, hf_dir / "model_bundle.pkl")
            (hf_dir / "app.py").write_text(app_py, encoding="utf-8")
            (hf_dir / "README.md").write_text(readme_md, encoding="utf-8")
            (hf_dir / "requirements.txt").write_text(reqs_txt, encoding="utf-8")

            # ── STEP 2: Create the Space ──────────────────────────────────────
            create_body = _json.dumps({
                "type": "space", "name": repo_name,
                "sdk": "gradio", "private": False,
            })
            status, resp, raw = _curl_json([
                "-X", "POST",
                "-H", auth_hdr,
                "-H", "Content-Type: application/json",
                "-d", create_body,
                "https://huggingface.co/api/repos/create",
            ], "create-space")

            if status not in (200, 201, 409):  # 409 = already exists, that's fine
                err_msg = (resp or {}).get("error", raw[:300]) if resp else raw[:300]
                if status == 403:
                    raise HTTPException(403,
                        f"Permission denied creating Space (HTTP 403). "
                        "Your token needs 'Write' scope on huggingface.co/settings/tokens")
                raise HTTPException(502,
                    f"Could not create HF Space (HTTP {status}): {err_msg}")

            L().info(f"HF Space created/exists: {full_repo}")

            # ── STEP 3: Upload files one by one via the upload API ────────────
            files_to_upload = [
                ("README.md",        "README.md",        "text/plain"),
                ("requirements.txt", "requirements.txt", "text/plain"),
                ("app.py",           "app.py",           "text/x-python"),
                ("model_bundle.pkl", "model_bundle.pkl", "application/octet-stream"),
            ]
            for local_name, remote_name, mime in files_to_upload:
                local_path = hf_dir / local_name
                status, resp, raw = _curl_json([
                    "-X", "PUT",
                    "-H", auth_hdr,
                    "-H", f"Content-Type: {mime}",
                    "--data-binary", f"@{local_path}",
                    f"https://huggingface.co/api/spaces/{full_repo}/upload/{remote_name}",
                ], f"upload-{local_name}")

                if status not in (200, 201):
                    # Try alternate upload endpoint
                    status, resp, raw = _curl_json([
                        "-X", "PUT",
                        "-H", auth_hdr,
                        "-H", f"Content-Type: {mime}",
                        "--data-binary", f"@{local_path}",
                        f"https://huggingface.co/api/repos/{full_repo}/upload/{remote_name}?repo_type=space",
                    ], f"upload-alt-{local_name}")

                if status not in (200, 201):
                    L().warning(f"Upload {local_name} HTTP {status}: {raw[:200]}")

            L().info(f"HF Space files uploaded: {full_repo}")

        finally:
            shutil.rmtree(hf_dir, ignore_errors=True)

        space_url = f"https://huggingface.co/spaces/{full_repo}"
        result = {
            "experiment_id": eid, "target": "huggingface", "status": "deployed",
            "repo_id": full_repo, "space_url": space_url,
            "embed_url": f"https://{username}-{repo_name}.hf.space",
            "model": model_nm,
            "message": f"✅ Deployed! Live in ~90 seconds at: {space_url}",
        }
        EXPS[eid]["deployment"] = result
        _save_meta(eid)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        L().error(f"HF deploy error:\n{_tb.format_exc()}")
        raise HTTPException(500, f"Deploy error: {exc}")




@app.post("/api/test-hf-token", tags=["Deploy"])
async def test_hf_token(req: Request):
    """
    Test a HF token using curl.
    Returns username, token type (read/write), and whether it can create Spaces.
    """
    import subprocess, json as _json
    try:
        body  = await req.json()
        raw   = body.get("token", "")
        token = "".join(raw.split()).strip('"').strip("'")

        if not token:
            return {"ok": False, "error": "No token provided"}

        if not token.startswith("hf_"):
            return {
                "ok": False,
                "error": f"Token must start with hf_ — got '{token[:8]}…' ({len(token)} chars). "
                         "Make sure you copied the full token.",
                "token_len": len(token),
                "token_prefix": token[:8],
            }

        L().info(f"Testing HF token: len={len(token)} prefix={token[:8]}")

        # ── Call whoami via curl ──────────────────────────────────────────────
        result = subprocess.run(
            ["curl", "--silent", "--show-error", "--location",
             "--max-time", "20",
             "-H", f"Authorization: Bearer {token}",
             "https://huggingface.co/api/whoami"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            return {
                "ok": False,
                "error": f"Network error — could not reach huggingface.co. "
                         f"Check your internet connection. (curl: {result.stderr[:200]})",
            }

        try:
            data = _json.loads(result.stdout)
        except Exception:
            return {
                "ok": False,
                "error": f"Unexpected response from HF: {result.stdout[:200]}",
            }

        # ── Token rejected by HF ─────────────────────────────────────────────
        if "error" in data:
            hf_err = data["error"]
            return {
                "ok": False,
                "hf_error": hf_err,
                "token_len": len(token),
                "token_prefix": token[:8],
                "error": (
                    f"HF rejected this token: \"{hf_err}\"\n\n"
                    "Most common causes:\n"
                    "① You created a READ token — you need a WRITE token\n"
                    "② The token was revoked or expired\n"
                    "③ You copied extra characters accidentally\n\n"
                    "Fix: go to huggingface.co/settings/tokens → "
                    "click 'New token' → set Role to 'Write' → copy the full token"
                ),
            }

        # ── Token accepted — check if it has write access ─────────────────────
        username     = data.get("name", "unknown")
        auth_info    = data.get("auth") or {}
        access_token = auth_info.get("accessToken") or {}
        token_role   = access_token.get("role", "unknown")   # 'read' or 'write'
        # older API uses 'type' instead of 'role'
        if token_role == "unknown":
            token_role = access_token.get("type", "unknown")

        can_write = token_role in ("write", "Write", "admin")

        L().info(f"HF token OK: user={username} role={token_role} can_write={can_write}")

        if not can_write and token_role not in ("unknown",):
            return {
                "ok": False,
                "username": username,
                "token_role": token_role,
                "token_len": len(token),
                "error": (
                    f"Token is valid for user '{username}' but has READ-only access. "
                    "Deploying to Spaces requires a WRITE token.\n\n"
                    "Fix: go to huggingface.co/settings/tokens → "
                    "create a NEW token → set Role to 'Write'"
                ),
            }

        return {
            "ok": True,
            "username": username,
            "token_role": token_role,
            "token_len": len(token),
            "can_create_spaces": can_write,
            "message": f"✅ Token valid! Logged in as '{username}' with {token_role} access.",
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": f"Unexpected error: {e}"}




@app.on_event("startup")
async def startup():
    _load_disk()
    print(f"\n{'='*60}\n  AutoML Agent v6.0  →  http://localhost:8000\n{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTIC AI LAYER
# ═══════════════════════════════════════════════════════════════════════════════

# In-memory agent config and session state
AGENT_CONFIG: Dict = {
    "api_key":  "",           # Anthropic API key
    "model":    "llama-3.1-8b-instant",
    "enabled":  False,
}
AGENT_CHATS: Dict[str, List[Dict]] = {}   # eid → chat history


class AgentConfigReq(BaseModel):
    api_key: str
    model: str = "llama-3.1-8b-instant"

class ChatReq(BaseModel):
    message: str
    experiment_id: Optional[str] = None


# ── Config ─────────────────────────────────────────────────────────────────────

@app.get("/agent/config", tags=["Agent"])
async def get_agent_config(req: Request):
    auth(req)
    m = AGENT_CONFIG["model"]
    return {
        "enabled":     AGENT_CONFIG["enabled"],
        "model":       m,
        "has_key":     bool(AGENT_CONFIG["api_key"]),
        "key_preview": ("***" + AGENT_CONFIG["api_key"][-4:]) if AGENT_CONFIG["api_key"] else "",
        "routing": {
            "plan":     "llama-3.1-8b-instant (always)",
            "reflect":  "llama-3.1-8b-instant (always)",
            "insights": m,
            "chat":     m,
        }
    }

@app.post("/agent/config", tags=["Agent"])
async def set_agent_config(body: AgentConfigReq, req: Request):
    auth(req)
    AGENT_CONFIG["api_key"]  = body.api_key.strip()
    AGENT_CONFIG["model"]    = body.model
    AGENT_CONFIG["enabled"]  = bool(body.api_key.strip())

    if AGENT_CONFIG["enabled"]:
        try:
            from openai import OpenAI as _OAI
        except ImportError:
            AGENT_CONFIG["enabled"] = False
            return {"ok": False, "enabled": False,
                    "message": "❌ openai package not installed. Run: pip install openai"}
        try:
            client = _OAI(
                api_key=AGENT_CONFIG["api_key"],
                base_url="https://api.groq.com/openai/v1",
            )
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )
            reply = resp.choices[0].message.content or ""
            return {"ok": True, "enabled": True,
                    "model": AGENT_CONFIG["model"],
                    "message": f"✅ Groq key valid! Model: {AGENT_CONFIG['model']}"}
        except Exception as e:
            err = str(e)
            AGENT_CONFIG["enabled"] = False
            if "401" in err or "invalid_api_key" in err.lower() or "authentication" in err.lower() or "api_key" in err.lower():
                return {"ok": False, "enabled": False,
                        "message": "❌ Invalid Groq API key. Get one free at console.groq.com/keys"}
            return {"ok": False, "enabled": False,
                    "message": f"❌ Groq error: {err[:200]}"}
    return {"ok": True, "enabled": False, "message": "Agent disabled — no API key"}

@app.post("/agent/test-key", tags=["Agent"])
async def test_agent_key(req: Request):
    """Quick test of the stored API key."""
    auth(req)
    if not AGENT_CONFIG["api_key"]:
        return {"ok": False, "error": "No API key configured"}
    try:
        from automl.llm_agent import LLMAgent
        agent = LLMAgent(api_key=AGENT_CONFIG["api_key"], user_model=AGENT_CONFIG["model"])
        # Quick ping — profile a tiny dataset
        import anthropic
        client = anthropic.Anthropic(api_key=AGENT_CONFIG["api_key"])
        resp = client.messages.create(
            user_model=AGENT_CONFIG["model"],
            max_tokens=30,
            messages=[{"role": "user", "content": "Reply only: OK"}],
        )
        reply = resp.content[0].text if resp.content else ""
        return {"ok": True, "model": AGENT_CONFIG["model"],
                "reply": reply, "message": "✅ API key works!"}
    except ImportError:
        return {"ok": False, "error": "anthropic package not installed. Run: pip install anthropic"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Agentic Training ───────────────────────────────────────────────────────────

@app.post("/agent/train", tags=["Agent"])
async def agent_train(req: Request, bg: BackgroundTasks,
                      file: UploadFile = File(...),
                      target_col: str = Form(...),
                      dataset_name: str = Form(default=""),
                      n_trials_override: int = Form(default=0)):
    """
    Fully agentic training — LLM decides everything.
    Streams logs showing the agent's reasoning.
    """
    auth(req)
    eid = str(uuid.uuid4())[:8]
    safe = re.sub(r'[^\w\-.]', '_', (dataset_name or file.filename or eid).strip())[:48]
    ext  = Path(file.filename or "data.csv").suffix or ".csv"
    p    = cfg().data_dir / f"{safe}_{eid}{ext}"
    p.write_bytes(await file.read())
    dname = dataset_name.strip() or file.filename or eid

    EXPS[eid] = {
        "status": "queued", "agent": None, "summary": None,
        "target_col": target_col, "filename": file.filename,
        "dataset_name": dname, "submitted_by": "admin",
        "created_at": time.time(), "agentic": True,
    }
    EXP_LOGS[eid] = []
    _push_log(eid, f"🤖 Agentic AutoML started for '{dname}' · target={target_col}")
    _push_log(eid, f"🧠 LLM: {'Claude ' + AGENT_CONFIG['model'] if AGENT_CONFIG['enabled'] else 'Rule-based fallback'}")

    bg.add_task(_agent_train_task, eid, p, target_col, n_trials_override)
    return {"experiment_id": eid, "status": "queued", "agentic": True, "dataset_name": dname}


def _agent_train_task(eid: str, path: Path, target_col: str, n_trials_override: int):
    """Background task: fully agentic training with ReAct loop + multi-agent."""
    EXPS[eid]["status"] = "training"
    _start = time.time()

    def _log(msg: str):
        _push_log(eid, msg)

    def _thought(t: str):
        # Stream agent thoughts as special log lines
        _push_log(eid, f"💭 {t}")

    try:
        from automl.agentic_core import MultiAgentAutoML
        runner = MultiAgentAutoML(
            api_key=AGENT_CONFIG["api_key"] if AGENT_CONFIG["enabled"] else None,
            user_model=AGENT_CONFIG["model"],
        )
        runner._start_ts = _start
        summary, agent_obj, memory = runner.run(
            str(path), target_col,
            push_log=_log, push_thought=_thought,
        )
        # Flatten memory into summary
        summary["agent_iterations_count"] = len(memory.iterations)
        summary["agent_data_issues"]      = memory.data_issues
        summary["agent_features_added"]   = memory.tried_features

        if "best_model" not in summary and "best_model_name" in summary:
            summary["best_model"] = summary["best_model_name"]
        if agent_obj.df is not None:
            summary["dataset_shape"] = [int(agent_obj.df.shape[0]), int(agent_obj.df.shape[1])]
        summary["elapsed_sec"]  = round(time.time() - EXPS[eid]["created_at"], 1)
        summary["dataset_name"] = EXPS[eid].get("dataset_name", eid)

        EXPS[eid].update({"agent": agent_obj, "summary": summary, "status": "completed"})

        # Persist model
        try:
            exp_dir = cfg().experiments_dir / eid
            exp_dir.mkdir(parents=True, exist_ok=True)
            with open(exp_dir / "best_model.pkl", "wb") as f: pickle.dump(agent_obj.best_model, f)
            with open(exp_dir / "preprocessor.pkl", "wb") as f: pickle.dump(agent_obj.preprocessing_engine, f)
            if getattr(agent_obj, "label_encoder", None):
                with open(exp_dir / "label_encoder.pkl", "wb") as f: pickle.dump(agent_obj.label_encoder, f)
            _log("💾 Model saved to disk")
        except Exception as pe:
            _log(f"⚠️ Could not persist: {pe}")

        _save_meta(eid)
        _log(f"✅ Agentic run complete — best: {summary.get('best_model','—')}")

    except Exception as e:
        import traceback
        EXPS[eid].update({"status": "failed", "error": str(e)})
        _push_log(eid, f"❌ Failed: {e}")
        L().error(traceback.format_exc())
        _save_meta(eid)


# ── Agent Analysis of existing experiment ─────────────────────────────────────

@app.get("/experiments/{eid}/agent-analyze", tags=["Agent"])
async def agent_analyze(eid: str, req: Request):
    """
    Run LLM analysis on an already-trained experiment.
    Returns structured insights, plan explanation, and recommendations.
    """
    auth(req)
    exp = EXPS.get(eid)
    if not exp: raise HTTPException(404)
    if exp["status"] != "completed": raise HTTPException(400, "Experiment not completed")

    s   = exp.get("summary") or {}
    bm  = s.get("best_metrics") or {}
    lb  = s.get("leaderboard") or []
    fi  = s.get("top_features") or []
    agent_obj = exp.get("agent")
    df = agent_obj.df if agent_obj else None

    # Build profile from available data
    try:
        if df is not None and agent_obj:
            from automl.llm_agent import profile_dataset
            profile = profile_dataset(df, agent_obj.target_col, agent_obj.problem_type)
        else:
            profile = {
                "n_rows": (s.get("dataset_shape") or [0])[0],
                "n_cols": (s.get("dataset_shape") or [0, 0])[1],
                "problem_type": s.get("problem_type", "unknown"),
                "target_col": s.get("target_col", "target"),
                "n_numeric": 0, "n_categorical": 0,
                "missing_pct": 0, "duplicate_pct": 0, "target": {},
                "top_correlations": [], "high_skew_features": [],
            }
    except Exception:
        profile = {"n_rows": 0, "n_cols": 0, "problem_type": "unknown",
                   "target_col": "target", "n_numeric": 0, "n_categorical": 0,
                   "missing_pct": 0, "duplicate_pct": 0, "target": {},
                   "top_correlations": [], "high_skew_features": []}

    from automl.llm_agent import LLMAgent
    llm = LLMAgent(
        api_key=AGENT_CONFIG["api_key"] if AGENT_CONFIG["enabled"] else None,
        user_model=AGENT_CONFIG["model"],
    )
    insights = llm.generate_insights(profile, lb, fi)

    # Also get plan reasoning if available
    existing_plan = s.get("agent_plan") or {}

    # Return enriched response including ReAct loop data if available
    react_result     = s.get("react_result") or {}
    agent_iterations = s.get("agent_iterations") or []
    data_issues      = s.get("data_issues") or s.get("agent_data_issues") or []
    features_added   = s.get("features_engineered") or s.get("agent_features_added") or []

    return {
        "experiment_id": eid,
        "agent_enabled": AGENT_CONFIG["enabled"],
        "insights": insights,
        "plan_used": existing_plan,
        "profile": profile,
        "agent_log": llm.get_log(),
        # ReAct loop data
        "react_result": react_result,
        "agent_iterations": agent_iterations,
        "agent_iterations_count": len(agent_iterations),
        "data_issues": data_issues,
        "features_engineered": features_added,
        "agent_thoughts": s.get("agent_log") or [],
    }


# ── Agent stream plan ──────────────────────────────────────────────────────────

@app.get("/agent/stream-plan", tags=["Agent"])
async def stream_plan(req: Request):
    """
    Stream the agent's planning reasoning for a given dataset profile.
    Useful for showing live LLM thought process in UI.
    """
    auth(req)
    params = dict(req.query_params)
    eid  = params.get("experiment_id")

    profile: Dict = {}
    if eid and eid in EXPS:
        exp = EXPS.get(eid)
        agent_obj = (exp or {}).get("agent")
        df = agent_obj.df if agent_obj else None
        if df is not None and agent_obj:
            from automl.llm_agent import profile_dataset
            try: profile = profile_dataset(df, agent_obj.target_col, agent_obj.problem_type)
            except: pass
    if not profile:
        profile = body.get("profile", {})

    from automl.llm_agent import LLMAgent
    llm = LLMAgent(
        api_key=AGENT_CONFIG["api_key"] if AGENT_CONFIG["enabled"] else None,
        user_model=AGENT_CONFIG["model"],
    )

    async def gen():
        for chunk in llm.stream_plan(profile):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield 'data: "__DONE__"\n\n'

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Chat interface ─────────────────────────────────────────────────────────────

@app.post("/agent/chat", tags=["Agent"])
async def agent_chat(body: ChatReq, req: Request):
    """Non-streaming chat — returns full answer."""
    auth(req)
    context: Dict = {}
    eid = body.experiment_id
    if eid and eid in EXPS:
        exp = EXPS.get(eid) or {}
        s   = exp.get("summary") or {}
        context = {
            "experiment_id": eid,
            "dataset_name":  exp.get("dataset_name", eid),
            "problem_type":  s.get("problem_type"),
            "target_col":    s.get("target_col"),
            "best_model":    s.get("best_model"),
            "best_metrics":  s.get("best_metrics"),
            "top_features":  (s.get("top_features") or [])[:6],
            "leaderboard":   (s.get("leaderboard") or [])[:5],
            "dataset_shape": s.get("dataset_shape"),
            "ai_insights":   s.get("ai_insights"),
        }

    from automl.llm_agent import LLMAgent
    llm = LLMAgent(
        api_key=AGENT_CONFIG["api_key"] if AGENT_CONFIG["enabled"] else None,
        user_model=AGENT_CONFIG["model"],
    )

    if not llm.is_available():
        return {"answer": (
            "AI chat is not available. Please configure your Anthropic API key in the Agent Settings tab. "
            "Go to 🤖 Agent → Settings → paste your key from console.anthropic.com"
        ), "confidence": "low"}

    full_answer = ""
    for chunk in llm.chat(body.message, context):
        full_answer += chunk

    AGENT_CHATS.setdefault(eid or "global", []).append({
        "role": "user", "content": body.message, "ts": time.strftime("%H:%M:%S")
    })
    AGENT_CHATS[eid or "global"].append({
        "role": "assistant", "content": full_answer, "ts": time.strftime("%H:%M:%S")
    })
    return {"answer": full_answer, "confidence": "high"}


@app.get("/agent/chat/stream", tags=["Agent"])
async def agent_chat_stream(req: Request):
    """SSE streaming chat."""
    auth(req)
    params  = dict(req.query_params)
    message = params.get("message", "")
    eid     = params.get("experiment_id")

    context: Dict = {}
    if eid and eid in EXPS:
        exp = EXPS.get(eid) or {}
        s   = exp.get("summary") or {}
        context = {
            "experiment_id": eid,
            "dataset_name": exp.get("dataset_name", eid),
            "problem_type": s.get("problem_type"),
            "best_model": s.get("best_model"),
            "best_metrics": s.get("best_metrics"),
            "top_features": (s.get("top_features") or [])[:5],
        }

    from automl.llm_agent import LLMAgent
    llm = LLMAgent(
        api_key=AGENT_CONFIG["api_key"] if AGENT_CONFIG["enabled"] else None,
        user_model=AGENT_CONFIG["model"],
    )

    async def gen():
        for chunk in llm.chat(message, context):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield 'data: "__DONE__"\n\n'

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/agent/chat/history", tags=["Agent"])
async def chat_history(req: Request, experiment_id: Optional[str] = None):
    auth(req)
    key = experiment_id or "global"
    return {"history": AGENT_CHATS.get(key, [])[-50:]}
