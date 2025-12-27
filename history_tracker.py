import json
from datetime import datetime

def log_project_state(df, filename="history_log.json"):
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "at_risk": int(len(df[df["status"] == "At Risk"])),
            "average_risk": float(df["risk_score"].mean())
        }
    }

    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    data.append(snapshot)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
