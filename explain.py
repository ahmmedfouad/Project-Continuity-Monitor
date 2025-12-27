from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class Explanation:
    total_score: float
    parts: List[Tuple[str, float]]


def explain_risk(row: pd.Series) -> Explanation:
    """Explain risk score as additive, human-readable components.

    Mirrors the current logic in `risk_engine.calculate_risk_score`.
    """
    parts: List[Tuple[str, float]] = []
    score = 0.0

    days = float(row.get("days_since_update", 0) or 0)
    ratio = float(row.get("completion_ratio", 0) or 0)
    activity = str(row.get("activity_level", ""))

    # Silence
    if days > 60:
        parts.append(("Silence > 60 days", 40.0))
        score += 40.0
    elif days > 30:
        parts.append(("Silence > 30 days", 25.0))
        score += 25.0

    # Completion
    if ratio < 0.3:
        parts.append(("Completion ratio < 0.3", 30.0))
        score += 30.0
    elif ratio < 0.6:
        parts.append(("Completion ratio < 0.6", 15.0))
        score += 15.0

    # Activity
    if activity == "Low":
        parts.append(("Activity level = Low", 20.0))
        score += 20.0
    elif activity == "Medium":
        parts.append(("Activity level = Medium", 10.0))
        score += 10.0

    return Explanation(total_score=min(score, 100.0), parts=parts)


def explain_df(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        expl = explain_risk(r)
        rows.append(
            {
                "project_id": r.get("project_id"),
                "project_name": r.get("project_name"),
                "risk_score": r.get("risk_score"),
                "explained_score": expl.total_score,
                "top_factors": ", ".join([name for name, _ in expl.parts]) if expl.parts else "No factors",
            }
        )
    return pd.DataFrame(rows)
