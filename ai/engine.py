from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass(frozen=True)
class AIProjectOutput:
    project_id: str
    forecast_risk_score_14d: float
    root_causes: List[str]
    recommendations: List[str]


def _safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def _infer_root_causes(row: pd.Series) -> List[str]:
    causes: List[str] = []

    days = float(row.get("days_since_update", 0) or 0)
    ratio = float(row.get("completion_ratio", 0) or 0)

    if days >= 60:
        causes.append("Long communication silence")
    elif days >= 30:
        causes.append("Communication cadence slowing")

    if ratio < 0.3:
        causes.append("Low milestone completion")
    elif ratio < 0.6:
        causes.append("Completion behind plan")

    activity = _safe_str(row.get("activity_level", ""))
    if activity == "Low":
        causes.append("Low activity level")

    # Optional business signals
    payment_delay = row.get("payment_delay_days")
    if payment_delay is not None and str(payment_delay) != "nan":
        try:
            if float(payment_delay) >= 30:
                causes.append("Payment delay")
        except ValueError:
            pass

    deps = row.get("dependency_count")
    if deps is not None and str(deps) != "nan":
        try:
            if float(deps) >= 3:
                causes.append("High external dependency load")
        except ValueError:
            pass

    approval_stage = _safe_str(row.get("approval_stage"))
    if approval_stage in {"Procurement", "UAT"}:
        causes.append(f"Stalled around {approval_stage}")

    return causes[:5] if causes else ["No dominant cause detected"]


def _recommendations_from_causes(causes: List[str]) -> List[str]:
    recs: List[str] = []

    if any("silence" in c.lower() or "cadence" in c.lower() for c in causes):
        recs.append("Schedule a status checkpoint within 48 hours")
        recs.append("Request a written progress update with blockers")

    if any("completion" in c.lower() or "milestone" in c.lower() for c in causes):
        recs.append("Re-baseline milestones and confirm next 2 deliverables")
        recs.append("Add weekly milestone acceptance review")

    if any("payment" in c.lower() for c in causes):
        recs.append("Escalate payment clearance and confirm disbursement date")

    if any("dependency" in c.lower() for c in causes):
        recs.append("Create dependency owner list and set due dates")

    if any("procurement" in c.lower() for c in causes):
        recs.append("Review procurement timeline and vendor readiness")

    if any("uat" in c.lower() for c in causes):
        recs.append("Define UAT entry/exit criteria and assign test owners")

    return recs[:6] if recs else ["Monitor and keep update cadence stable"]


def _forecast_risk_14d(row: pd.Series) -> float:
    current = float(row.get("risk_score", 0) or 0)
    days = float(row.get("days_since_update", 0) or 0)
    activity = _safe_str(row.get("activity_level", ""))

    drift = 0.0
    if days >= 45:
        drift += 8
    elif days >= 25:
        drift += 4

    if activity == "Low":
        drift += 5
    elif activity == "High":
        drift -= 2

    return max(0.0, min(100.0, current + drift))


def run_ai_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Optional AI-like inference (rule-based placeholder).

    This is intentionally deterministic and explainable.
    Later you can replace internals with ML/LLM while keeping the same output schema.
    """
    required = {"project_id", "risk_score", "days_since_update", "completion_ratio", "activity_level"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"AI inference requires columns: {missing}")

    ml_forecast = _try_model_forecast(df, model_file="ai/model.json")

    outputs: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        causes = _infer_root_causes(row)
        recs = _recommendations_from_causes(causes)
        pid = _safe_str(row.get("project_id"))
        forecast = ml_forecast.get(pid)
        if forecast is None:
            forecast = _forecast_risk_14d(row)
        outputs.append(
            {
                "project_id": pid,
                "forecast_risk_score_14d": float(max(0.0, min(100.0, forecast))),
                "root_causes": causes,
                "recommendations": recs,
            }
        )

    return pd.DataFrame(outputs)


def _try_model_forecast(df: pd.DataFrame, model_file: str) -> Dict[str, float]:
    """Try to forecast using a trained model artifact.

    Returns mapping project_id -> forecast, or empty dict if unavailable.
    """
    path = Path(model_file)
    if not path.exists():
        return {}

    try:
        model = json.loads(path.read_text())
    except Exception:
        return {}

    if model.get("kind") != "ridge_regression_onehot":
        return {}

    feature_cols: List[str] = model.get("feature_columns", [])
    weights: List[float] = model.get("weights", [])
    if not feature_cols or not weights or len(feature_cols) != len(weights):
        return {}

    # Build features similarly to train.py (minimal copy to keep isolation).
    x = df.copy()

    numeric = [
        "days_since_update",
        "completion_ratio",
        "risk_score",
        "payment_delay_days",
        "dependency_count",
        "team_size",
    ]
    categoricals = [
        "activity_level",
        "sector",
        "org_type",
        "region",
        "priority",
        "approval_stage",
        "update_channel",
    ]

    for c in numeric:
        if c not in x.columns:
            x[c] = 0.0
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    for c in categoricals:
        if c not in x.columns:
            x[c] = ""
        x[c] = x[c].fillna("").astype(str)

    base = x[numeric + categoricals]
    encoded = pd.get_dummies(base, columns=categoricals, dummy_na=False)

    # Align columns with model
    encoded_cols = encoded.columns.tolist()
    for col in feature_cols:
        if col == "__bias__":
            continue
        if col not in encoded_cols:
            encoded[col] = 0.0
    encoded = encoded[[c for c in feature_cols if c != "__bias__"]]

    import numpy as np

    X = encoded.to_numpy(dtype=float)
    X = np.hstack([np.ones((X.shape[0], 1), dtype=float), X])
    w = np.array(weights, dtype=float)
    yhat = X @ w

    preds: Dict[str, float] = {}
    for pid, pred in zip(df["project_id"].astype(str).tolist(), yhat.tolist()):
        preds[pid] = float(pred)
    return preds
