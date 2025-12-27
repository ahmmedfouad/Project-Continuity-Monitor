from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainConfig:
    history_entities_file: str = "history_entities_log.json"
    output_model_file: str = "ai/model.json"
    horizon_days: int = 14
    min_samples: int = 50
    allow_synthetic_targets: bool = True


FEATURES_NUMERIC = [
    "days_since_update",
    "completion_ratio",
    "risk_score",
    "payment_delay_days",
    "dependency_count",
    "team_size",
]

FEATURES_CATEGORICAL = [
    "activity_level",
    "sector",
    "org_type",
    "region",
    "priority",
    "approval_stage",
    "update_channel",
]


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _build_training_pairs(history: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Create supervised pairs: X at time t -> y = risk_score at t+horizon.

    Uses per-project snapshots from `history_entities_log.json`.
    """
    if len(history) == 0:
        return pd.DataFrame()

    h = history.copy()
    h["timestamp"] = _to_datetime(h["timestamp"])
    h = h.dropna(subset=["timestamp", "project_id"])
    h = h.sort_values(["project_id", "timestamp"])

    rows = []
    for project_id, grp in h.groupby("project_id"):
        times = grp["timestamp"].to_numpy()
        # For each row i, find first j where time_j >= time_i + horizon
        for i in range(len(grp)):
            t0 = grp.iloc[i]["timestamp"]
            target_time = t0 + pd.Timedelta(days=horizon_days)
            j_candidates = grp.index[grp["timestamp"] >= target_time]
            if len(j_candidates) == 0:
                continue
            j = j_candidates[0]
            x = grp.loc[grp.index[i]].to_dict()
            y = grp.loc[j]["risk_score"]
            if pd.isna(y):
                continue
            x["target_risk_score"] = float(y)
            rows.append(x)

    return pd.DataFrame(rows)


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    for c in FEATURES_NUMERIC:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    for c in FEATURES_CATEGORICAL:
        if c not in x.columns:
            x[c] = ""
        x[c] = x[c].fillna("").astype(str)

    # Drop identifiers / non-features and columns that can leak target timestamps
    drop_cols = [
        "timestamp",
        "project_id",
        "project_name",
        "last_update",
        "status",
        # Never include target as input feature
        "target_risk_score",
    ]
    x = x.drop(columns=[col for col in drop_cols if col in x.columns])
    return x


def _fit_linear_model(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> Tuple[np.ndarray, float]:
    """Ridge regression closed-form: w = (X^T X + l2 I)^-1 X^T y."""
    XtX = X.T @ X
    I = np.eye(XtX.shape[0])
    w = np.linalg.solve(XtX + l2 * I, X.T @ y)
    return w, l2


def _evaluate(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    yhat = X @ w
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    return {"mae": mae, "rmse": rmse}


def _synthetic_target(row: pd.Series) -> float:
    current = float(pd.to_numeric(row.get("risk_score", 0), errors="coerce") or 0)
    days = float(pd.to_numeric(row.get("days_since_update", 0), errors="coerce") or 0)
    activity = str(row.get("activity_level", ""))

    drift = 0.0
    if days >= 45:
        drift += 8
    elif days >= 25:
        drift += 4

    if activity == "Low":
        drift += 5
    elif activity == "High":
        drift -= 2

    return float(max(0.0, min(100.0, current + drift)))


def train(config: TrainConfig) -> Dict:
    history_path = Path(config.history_entities_file)
    if not history_path.exists():
        raise FileNotFoundError(
            f"{config.history_entities_file} not found. Run the app and click 'Save Snapshot' a few times first."
        )

    history = pd.read_json(history_path)
    pairs = _build_training_pairs(history, horizon_days=config.horizon_days)
    used_synthetic = False
    if len(pairs) < config.min_samples:
        if not config.allow_synthetic_targets:
            raise ValueError(
                f"Not enough supervised samples ({len(pairs)}). Need at least {config.min_samples}. "
                "Create more history snapshots over time, or set allow_synthetic_targets=True."
            )

        # Demo fallback: create synthetic labels using a simple drift heuristic.
        # This keeps the training pipeline runnable in MVP while you collect real history.
        used_synthetic = True
        pairs = history.copy()
        pairs["target_risk_score"] = pairs.apply(_synthetic_target, axis=1)

    y = pd.to_numeric(pairs["target_risk_score"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y)
    if mask.sum() == 0:
        raise ValueError("Training target contains no finite values.")
    pairs = pairs.loc[mask].copy()
    y = y[mask]
    X_df = _clean_features(pairs)

    # One-hot encode categoricals (simple and portable)
    X_encoded = pd.get_dummies(X_df, columns=FEATURES_CATEGORICAL, dummy_na=False)

    # Fill NaNs deterministically (median where possible, else 0)
    med = X_encoded.median(numeric_only=True)
    X_encoded = X_encoded.fillna(med).fillna(0.0)

    X = X_encoded.to_numpy(dtype=float)

    # Add bias term
    X = np.hstack([np.ones((X.shape[0], 1), dtype=float), X])

    w, l2 = _fit_linear_model(X, y, l2=1.0)
    metrics = _evaluate(X, y, w)

    model = {
        "kind": "ridge_regression_onehot",
        "version": 1,
        "horizon_days": config.horizon_days,
        "l2": float(l2),
        "feature_columns": ["__bias__"] + X_encoded.columns.tolist(),
        "weights": w.tolist(),
        "training": {
            **asdict(config),
            "n_samples": int(X.shape[0]),
            "used_synthetic_targets": bool(used_synthetic),
            "metrics": metrics,
        },
    }

    out_path = Path(config.output_model_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # JSON must not contain NaN/Infinity
    out_path.write_text(json.dumps(model, indent=2, allow_nan=False))
    return model


def main() -> None:
    cfg = TrainConfig()
    model = train(cfg)
    print(
        f"Trained model v{model['version']} (horizon={model['horizon_days']}d) "
        f"on {model['training']['n_samples']} samples -> {cfg.output_model_file}"
    )


if __name__ == "__main__":
    main()
