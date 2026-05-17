"""
AutoML LLM Agent Brain — Groq-powered via OpenAI-compatible client.
Forces tool calls so LLM never falls back to plain text.
Falls back gracefully to rule-based planner on any failure.
"""
from __future__ import annotations
import json, time, re
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np
import pandas as pd

try:
    from openai import OpenAI as _OpenAI
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# ─────────────────────────────────────────────────────────────────────────────
# Model routing
# ─────────────────────────────────────────────────────────────────────────────

TASK_MODELS: Dict[str, tuple] = {
    "plan":     ("llama-3.1-8b-instant",     600),
    "reflect":  ("llama-3.1-8b-instant",     300),
    "insights": ("llama-3.1-8b-instant",     700),
    "chat":     ("llama-3.1-8b-instant",     512),
    "stream":   ("llama-3.1-8b-instant",     450),
}
ESCALATE_TASKS = {"chat", "insights", "stream"}

# ─────────────────────────────────────────────────────────────────────────────
# Compact dataset profiler
# ─────────────────────────────────────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame, target_col: str, problem_type: str) -> Dict:
    n, c = df.shape
    num_cols = [col for col in df.select_dtypes(include=np.number).columns if col != target_col]
    cat_cols = [col for col in df.select_dtypes(include="object").columns  if col != target_col]
    miss = round(df.isnull().mean().mean() * 100, 1)
    dup  = round(df.duplicated().mean()   * 100, 1)
    tgt  = df[target_col].dropna()

    if problem_type == "classification":
        vc = tgt.value_counts()
        target_info: Dict = {
            "classes": int(vc.shape[0]),
            "imbalance": round(float(vc.min() / vc.max()), 2) if vc.max() else 1.0,
            "min_class_n": int(vc.min()),
        }
    else:
        target_info = {
            "mean": round(float(tgt.mean()), 3),
            "std":  round(float(tgt.std()),  3),
            "skew": round(float(tgt.skew()), 2),
        }

    top_corr: List[str] = []
    if problem_type == "regression" and num_cols:
        corrs = df[num_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        top_corr = [f"{col}={v:.2f}" for col, v in corrs.head(4).items()]

    high_skew = []
    for col in num_cols[:15]:
        try:
            sk = float(df[col].skew())
            if abs(sk) > 1.5: high_skew.append(f"{col}({sk:.1f})")
        except: pass

    return {
        "rows": n, "cols": c,
        "num_feats": len(num_cols), "cat_feats": len(cat_cols),
        "missing_pct": miss, "dup_pct": dup,
        "task": problem_type, "target": target_col,
        "target_info": target_info,
        "top_corr":   top_corr,
        "high_skew":  high_skew[:4],
        "sample_num": num_cols[:5],
        "sample_cat": cat_cols[:3],
    }


def _compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))

def _trim_lb(leaderboard: List[Dict], n: int = 4) -> List[Dict]:
    out = []
    for r in leaderboard[:n]:
        m = dict(list((r.get("metrics") or {}).items())[:3])
        out.append({"m": r.get("model_name", "?"), "s": m})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYS_PLAN = (
    "You are an AutoML expert. Analyze the dataset profile and call the "
    "set_training_plan function. You MUST call the function — do not reply in plain text."
)
_SYS_REFLECT = (
    "You review AutoML results. If more training would significantly help, "
    "call request_more_training. Otherwise reply 'OK'."
)
_SYS_INSIGHTS = (
    "You summarize ML results for non-technical users. "
    "You MUST call the generate_final_insights function — do not reply in plain text."
)
_SYS_CHAT = (
    "You are a helpful ML assistant. Answer concisely using the experiment context. No preamble."
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas — OpenAI / Groq function-calling format
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_training_plan",
            "description": "Set the complete AutoML training strategy for this dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "1-2 sentence explanation of your strategy choices."
                    },
                    "recommended_models": {
                        "type": "array",
                        "description": "Models to train, ordered by priority. Must be from the allowed list.",
                        "items": {
                            "type": "string",
                            "enum": ["xgboost","lightgbm","catboost","random_forest",
                                     "extra_trees","logistic_regression","ridge",
                                     "svm","knn","neural_net","adaboost","gradient_boosting"]
                        }
                    },
                    "n_trials": {
                        "type": "integer",
                        "description": "Optuna HPO trials per model. Use 10-15 for small data, 25-40 for large."
                    },
                    "preprocessing_strategy": {
                        "type": "string",
                        "enum": ["standard","robust","minmax","power","auto"],
                        "description": "Scaling strategy. Use robust if missing>5% or high skew."
                    },
                    "feature_selection": {
                        "type": "string",
                        "enum": ["mutual_info","model_based","none"]
                    },
                    "handle_class_imbalance": {
                        "type": "boolean",
                        "description": "True if imbalance ratio < 0.4."
                    },
                    "priority_metric": {
                        "type": "string",
                        "description": "Metric to optimise: accuracy, f1_weighted, roc_auc, r2, rmse."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Your confidence in this plan, 0.0-1.0."
                    }
                },
                "required": ["reasoning","recommended_models","n_trials",
                             "preprocessing_strategy","priority_metric","confidence"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_more_training",
            "description": "Request a second training pass if initial results are unsatisfactory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning":         {"type": "string"},
                    "additional_models": {"type": "array", "items": {"type": "string"}},
                    "additional_trials": {"type": "integer"},
                    "focus_on":          {"type": "string"}
                },
                "required": ["reasoning","additional_models","additional_trials"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_final_insights",
            "description": "Generate plain-language insights about the completed AutoML run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "executive_summary": {"type": "string"},
                    "model_explanation": {"type": "string"},
                    "key_findings":      {"type": "array", "items": {"type": "string"}},
                    "risks_and_caveats": {"type": "array", "items": {"type": "string"}},
                    "next_steps":        {"type": "array", "items": {"type": "string"}}
                },
                "required": ["executive_summary","model_explanation",
                             "key_findings","risks_and_caveats","next_steps"]
            }
        }
    }
]

# One-tool subsets used when forcing a specific tool call
_TOOLS_PLAN     = [TOOLS[0]]
_TOOLS_REFLECT  = TOOLS[1:2]  # empty list means no tools — we use auto
_TOOLS_INSIGHTS = [TOOLS[2]]


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback planner (zero API calls, always works)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_plan(p: Dict) -> Dict:
    n, clf = p["rows"], p["task"] == "classification"
    n_cat   = p["cat_feats"]
    miss    = p["missing_pct"]
    imb     = p.get("target_info", {}).get("imbalance", 1.0)
    n_cls   = p.get("target_info", {}).get("classes", 2)
    high_sk = len(p.get("high_skew", []))

    if n < 500:
        models = (["logistic_regression","random_forest","svm","knn"]
                  if clf else ["ridge","random_forest","svm"])
        trials = 12
    elif n < 5000:
        models = (["xgboost","lightgbm","random_forest","logistic_regression"]
                  if clf else ["xgboost","lightgbm","ridge","random_forest"])
        trials = 25
    else:
        models = (["lightgbm","xgboost","catboost","random_forest"]
                  if clf else ["lightgbm","xgboost","catboost","random_forest"])
        trials = 40

    if n_cat > 3:
        models = ["catboost"] + [m for m in models if m != "catboost"]

    pp     = "robust" if (miss > 5 or high_sk > 2) else ("power" if high_sk else "standard")
    metric = ("roc_auc" if (clf and n_cls == 2) else "f1_weighted" if clf else "r2")

    return {
        "recommended_models": models[:4],
        "n_trials": trials,
        "preprocessing_strategy": pp,
        "feature_selection": "mutual_info",
        "handle_class_imbalance": clf and imb < 0.4,
        "priority_metric": metric,
        "confidence": 0.75,
        "reasoning": (
            f"Rule-based: {n} rows, {p['num_feats']} numeric + {n_cat} categorical, "
            f"{miss}% missing. {'GB stack' if n > 500 else 'Small-data stack'}."
        ),
        "_source": "rules",
    }


def _parse_json_from_text(text: str, required_key: str) -> Optional[Dict]:
    """
    Last-resort parser: extract JSON from LLM plain-text response.
    Handles cases where LLM wraps JSON in markdown fences.
    """
    if not text:
        return None
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Try full parse
    try:
        obj = json.loads(text)
        if required_key in obj:
            return obj
    except Exception:
        pass
    # Try to find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if required_key in obj:
                return obj
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────────────────────────────────────────

class LLMAgent:
    """
    Groq-powered agent via OpenAI-compatible client.
    Forces tool calls for structured tasks so the LLM can't dodge into plain text.
    Always falls back to rule-based planner on failure.
    """

    def __init__(self, api_key: Optional[str] = None,
                 user_model: str = "llama-3.1-8b-instant"):
        self.api_key    = api_key
        self.user_model = user_model
        self.client: Optional[Any] = None
        self._log: List[Dict] = []
        self._tokens_used = {"input": 0, "output": 0}

        if HAS_GROQ and api_key:
            try:
                self.client = _OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
            except Exception as e:
                self.client = None
                self._add_log("warn", f"Client init failed: {e}")

    def is_available(self) -> bool:
        return self.client is not None

    def token_usage(self) -> Dict:
        return dict(self._tokens_used)

    def _add_log(self, role: str, content: str, tool: Optional[str] = None):
        self._log.append({"role": role, "content": content,
                          "tool": tool, "ts": time.strftime("%H:%M:%S")})

    def get_log(self) -> List[Dict]:
        return self._log

    def _model_for(self, task: str) -> tuple:
        default_model, max_tok = TASK_MODELS.get(task, ("llama-3.1-8b-instant", 500))
        if task in ESCALATE_TASKS and self.user_model != "llama-3.1-8b-instant":
            return self.user_model, max_tok + 200
        return default_model, max_tok

    def _call(self, system: str, messages: List[Dict],
              tools: Optional[List], task: str,
              force_tool: Optional[str] = None) -> Optional[Any]:
        """
        Call Groq via OpenAI client.
        force_tool: if set, force the model to call this specific function.
        """
        if not self.client:
            return None

        model, max_tok = self._model_for(task)
        full_messages  = [{"role": "system", "content": system}] + messages

        # Force a specific tool call so LLM can't respond with plain text
        tool_choice: Any = "auto"
        if force_tool and tools:
            tool_choice = {"type": "function", "function": {"name": force_tool}}

        for attempt in range(3):
            try:
                kwargs: Dict = {
                    "model":       model,
                    "max_tokens":  max_tok,
                    "messages":    full_messages,
                    "temperature": 0.1,
                }
                if tools:
                    kwargs["tools"]       = tools
                    kwargs["tool_choice"] = tool_choice

                resp = self.client.chat.completions.create(**kwargs)

                if hasattr(resp, "usage") and resp.usage:
                    self._tokens_used["input"]  += resp.usage.prompt_tokens or 0
                    self._tokens_used["output"] += resp.usage.completion_tokens or 0

                return resp

            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    wait = 60 if attempt == 0 else 30
                    self._add_log("warn", f"Rate limit hit — waiting {wait}s…")
                    time.sleep(wait)
                    continue
                if attempt == 2:
                    self._add_log("error", f"API call failed: {err[:200]}")
                    return None
                time.sleep(2 ** attempt)
        return None

    def _extract_tool(self, resp: Any, name: str) -> Optional[Dict]:
        """Extract tool call from response. Falls back to text JSON parsing."""
        if not resp:
            return None
        try:
            msg = resp.choices[0].message

            # Primary: proper tool_calls field
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function.name == name:
                        try:
                            return json.loads(tc.function.arguments)
                        except Exception:
                            pass

            # Fallback: model returned JSON in text content
            text = msg.content or ""
            if text:
                result = _parse_json_from_text(text, "recommended_models" if name == "set_training_plan"
                                               else "executive_summary" if name == "generate_final_insights"
                                               else "additional_models")
                if result:
                    self._add_log("warn", f"Tool {name}: parsed from text fallback")
                    return result

        except Exception as e:
            self._add_log("error", f"_extract_tool({name}): {e}")

        return None

    def _text(self, resp: Any) -> str:
        if not resp: return ""
        try: return resp.choices[0].message.content or ""
        except: return ""

    # ── Phase 1: Plan ─────────────────────────────────────────────────────────

    def plan(self, profile: Dict, push_log: Optional[Callable] = None) -> Dict:
        def _log(m):
            self._add_log("agent", m)
            if push_log: push_log(m)

        if not self.client:
            _log("🧠 Rule-based planner (no API key configured)")
            plan = _rule_based_plan(profile)
            _log(f"📋 {plan['reasoning']}")
            return plan

        _log(f"🧠 LLM planning strategy ({self._model_for('plan')[0]})…")

        prompt = (
            f"Dataset profile:\n{_compact(profile)}\n\n"
            f"Analyze this and call set_training_plan with the best strategy. "
            f"Pick models suited to the data size, types and task. "
            f"Keep reasoning to 1-2 sentences."
        )

        resp = self._call(
            _SYS_PLAN,
            [{"role": "user", "content": prompt}],
            tools=_TOOLS_PLAN,
            task="plan",
            force_tool="set_training_plan",   # ← force: no plain-text escape
        )

        plan = self._extract_tool(resp, "set_training_plan")

        if plan:
            plan["_source"] = "llm"
            _log(f"✅ LLM plan: models={plan.get('recommended_models',[])} "
                 f"trials={plan.get('n_trials')} pp={plan.get('preprocessing_strategy')} "
                 f"conf={plan.get('confidence',0):.0%}")
        else:
            _log(f"⚠️ LLM did not return a plan (text: {self._text(resp)[:80]}). Using rules.")
            plan = _rule_based_plan(profile)

        plan["_reasoning"] = plan.get("reasoning", "")
        return plan

    # ── Phase 2: Reflect ──────────────────────────────────────────────────────

    def reflect(self, profile: Dict, plan: Dict, leaderboard: List[Dict],
                push_log: Optional[Callable] = None) -> Optional[Dict]:
        def _log(m):
            self._add_log("agent", m)
            if push_log: push_log(m)

        if not self.client or not leaderboard:
            return None

        top    = leaderboard[0]
        prompt = (
            f"Task:{profile['task']} rows:{profile['rows']} "
            f"best_model:{top.get('model_name')} "
            f"metrics:{_compact(dict(list((top.get('metrics') or {}).items())[:3]))} "
            f"leaderboard:{_compact(_trim_lb(leaderboard, 3))}\n"
            f"Is performance satisfactory? If yes, reply 'OK'. "
            f"If a second pass would significantly help, call request_more_training."
        )
        _log("🔍 LLM reflecting on results…")
        resp  = self._call(
            _SYS_REFLECT,
            [{"role": "user", "content": prompt}],
            tools=[TOOLS[1]],
            task="reflect",
            force_tool=None,   # allow free choice: text "OK" or tool call
        )
        extra = self._extract_tool(resp, "request_more_training")
        text  = self._text(resp)

        if extra:
            _log(f"🔄 More training requested: {extra.get('reasoning','')}")
        else:
            _log(f"💭 Reflection: {text[:100]}")
        return extra

    # ── Phase 3: Insights ─────────────────────────────────────────────────────

    def generate_insights(self, profile: Dict, leaderboard: List[Dict],
                          top_features: List[Dict],
                          push_log: Optional[Callable] = None) -> Dict:
        def _log(m):
            self._add_log("agent", m)
            if push_log: push_log(m)

        if not leaderboard:
            return {}

        if not self.client:
            top = leaderboard[0]; bm = top.get("metrics", {})
            clf = profile["task"] == "classification"
            sk  = "accuracy" if clf else "r2"; sv = bm.get(sk, 0)
            feats = [f["feature"] for f in top_features[:3]] if top_features else []
            return {
                "executive_summary": (
                    f"{top.get('model_name')} achieved {sk}={sv:.4f}. "
                    f"{'Strong — production-ready.' if sv > 0.85 else 'Solid baseline.'}"
                ),
                "model_explanation": f"{top.get('model_name')} performed best across all candidates.",
                "key_findings": [
                    f"Best {sk}: {sv:.4f}",
                    f"Top features: {', '.join(feats)}" if feats else "Features computed",
                    f"{profile['rows']:,} rows × {profile['cols']} columns",
                ],
                "risks_and_caveats": ["Validate on fresh holdout data before production."],
                "next_steps": ["Run sensitivity analysis.", "Deploy via the Deploy tab."],
            }

        _log("✍️ LLM generating insights…")
        fi_compact = [{"f": f["feature"], "i": round(f.get("importance", 0), 4)}
                      for f in top_features[:5]]
        prompt = (
            f"task:{profile['task']} target:{profile['target']} "
            f"rows:{profile['rows']} cols:{profile['cols']}\n"
            f"leaderboard:{_compact(_trim_lb(leaderboard, 4))}\n"
            f"top_features:{_compact(fi_compact)}\n"
            f"Call generate_final_insights. Each field max 2 sentences."
        )
        resp = self._call(
            _SYS_INSIGHTS,
            [{"role": "user", "content": prompt}],
            tools=_TOOLS_INSIGHTS,
            task="insights",
            force_tool="generate_final_insights",
        )
        insights = self._extract_tool(resp, "generate_final_insights") or {}

        if not insights:
            top = leaderboard[0]
            insights = {
                "executive_summary": f"Best model: {top.get('model_name','model')}. Training complete.",
                "model_explanation": "Selected by validation performance across all candidates.",
                "key_findings": [], "risks_and_caveats": [], "next_steps": []
            }

        tok = self._tokens_used
        _log(f"✅ Insights done. Tokens: in={tok['input']} out={tok['output']}")
        return insights

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat(self, question: str, context: Dict) -> Generator[str, None, None]:
        if not self.client:
            yield "No API key configured. Go to 🤖 Agent → ⚙ Settings."
            return

        ctx_slim = {
            "ds":      context.get("dataset_name", ""),
            "task":    context.get("problem_type", ""),
            "target":  context.get("target_col", ""),
            "model":   context.get("best_model", ""),
            "metrics": context.get("best_metrics", {}),
            "top_f":   [f["feature"] for f in (context.get("top_features") or [])[:5]],
            "shape":   context.get("dataset_shape", []),
        }
        model, max_tok = self._model_for("chat")
        try:
            stream = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tok,
                messages=[
                    {"role": "system", "content": _SYS_CHAT},
                    {"role": "user",   "content": f"Context:{_compact(ctx_slim)}\n\nQ: {question}"},
                ],
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"\n[Error: {e}]"

    # ── Streaming plan reasoning ───────────────────────────────────────────────

    def stream_plan(self, profile: Dict) -> Generator[str, None, None]:
        if not self.client:
            plan = _rule_based_plan(profile)
            yield "**Rule-based plan** (no Groq key configured)\n\n"
            yield f"Models: {', '.join(plan['recommended_models'])}\n"
            yield f"Trials: {plan['n_trials']} | Preprocessing: {plan['preprocessing_strategy']}\n"
            yield f"Reason: {plan['reasoning']}\n"
            return

        model, max_tok = self._model_for("stream")
        prompt = (
            f"Dataset: {_compact(profile)}\n\n"
            f"Think step by step about this dataset: what do you notice? "
            f"What ML challenges exist? Which models would you try and why? "
            f"What preprocessing makes sense? Keep it under 150 words."
        )
        try:
            stream = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tok,
                messages=[
                    {"role": "system", "content": _SYS_PLAN},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"\n[Error: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Agentic runner
# ─────────────────────────────────────────────────────────────────────────────

class AgenticAutoML:
    """LLM-controlled AutoML pipeline using Groq."""

    def __init__(self, api_key: Optional[str],
                 model: str = "llama-3.1-8b-instant"):
        self.llm = LLMAgent(api_key=api_key, user_model=model)

    def run(self, file_path: str, target_col: str,
            push_log: Optional[Callable] = None) -> tuple:
        from automl.agent import AutoMLAgent
        from automl.data_handler import load_dataset, detect_problem_type

        def _log(m):
            if push_log: push_log(f"🤖 {m}")

        _log("Loading dataset…")
        df = load_dataset(file_path)
        problem_type = detect_problem_type(df, target_col, None)
        _log(f"Detected: {problem_type} · {df.shape[0]:,} rows × {df.shape[1]} cols")

        profile = profile_dataset(df, target_col, problem_type)
        _log(f"Profiled: {profile['num_feats']} numeric, {profile['cat_feats']} categorical, "
             f"{profile['missing_pct']}% missing")

        # Phase 1 — Plan
        plan = self.llm.plan(profile, push_log=push_log)
        source = plan.get("_source", "rules")
        _log(f"Plan source: {source} | models: {plan.get('recommended_models',[])} | "
             f"trials: {plan.get('n_trials')}")

        n_trials = min(int(plan.get("n_trials") or 25), 80)
        feat_sel = plan.get("feature_selection", "mutual_info")

        # Phase 2 — Execute AutoML with LLM-chosen params
        agent = AutoMLAgent(
            target_col=target_col,
            problem_type=problem_type,
            feature_selection=feat_sel,
            handle_outliers=True,
            n_trials=n_trials,
        )
        agent._llm_plan = plan   # trainer reads this to filter models
        _log(f"Launching AutoML: {plan.get('recommended_models',[])} × {n_trials} trials…")
        summary  = agent.run(file_path)
        leaderboard = summary.get("leaderboard", [])

        # Phase 3 — Reflect
        _log("LLM reviewing results…")
        extra = self.llm.reflect(profile, plan, leaderboard, push_log=push_log)
        if extra and extra.get("additional_trials", 0) > 0:
            _log(f"[Second pass planned: {extra.get('focus_on','improvements')}]")

        # Phase 4 — Insights
        _log("Generating insights…")
        top_features = summary.get("top_features", [])
        insights = self.llm.generate_insights(profile, leaderboard, top_features, push_log=push_log)

        tok = self.llm.token_usage()
        _log(f"Total tokens used: {tok['input']+tok['output']:,} "
             f"(in={tok['input']:,} out={tok['output']:,})")

        summary["agent_plan"]      = plan
        summary["agent_insights"]  = insights
        summary["agent_log"]       = self.llm.get_log()
        summary["agent_tokens"]    = tok
        summary["ai_insights"]     = _format_ai_insights(insights, summary)
        summary["dataset_profile"] = profile

        _log("✅ Agentic run complete!")
        return summary, agent


def _format_ai_insights(insights: Dict, summary: Dict) -> List[str]:
    result = []
    if insights.get("executive_summary"):
        result.append(f"📊 {insights['executive_summary']}")
    for f in (insights.get("key_findings") or [])[:3]:
        result.append(f"🔑 {f}")
    for r in (insights.get("risks_and_caveats") or [])[:2]:
        result.append(f"⚠️ {r}")
    bm = summary.get("best_metrics", {})
    if bm:
        k, v = next(iter(bm.items()), (None, None))
        if k: result.append(f"🏆 {k}: {v:.4f} | {summary.get('best_model','—')}")
    return result
