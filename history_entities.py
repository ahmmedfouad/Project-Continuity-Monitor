import json
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


def log_entities_state(df: pd.DataFrame, filename: str = "history_entities_log.json") -> None:
    """Append a per-project snapshot to history.

    Keeps minimal, stable schema suitable for later ML training.
    """
    ts = datetime.now().isoformat()

    cols = [
        "project_id",
        "project_name",
        "sector",
        "last_update",
        "days_since_update",
        "completion_ratio",
        "activity_level",
        "risk_score",
        "status",
    ]

    snapshot_rows = []
    for _, row in df.iterrows():
        item: Dict[str, Any] = {"timestamp": ts}
        for c in cols:
            item[c] = row.get(c)
        snapshot_rows.append(item)

    try:
        with open(filename, "r") as f:
            data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        data = []

    data.extend(snapshot_rows)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)
