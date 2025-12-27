from datetime import datetime
import pandas as pd

from risk_engine import calculate_risk_score


def analyze_projects(df):
    df["last_update"] = pd.to_datetime(df["last_update"])
    today = datetime.today()

    df["days_since_update"] = (today - df["last_update"]).dt.days
    df["planned_milestones"] = pd.to_numeric(df["planned_milestones"], errors="coerce")
    df["completed_milestones"] = pd.to_numeric(df["completed_milestones"], errors="coerce")

    planned = df["planned_milestones"].replace(0, pd.NA)
    df["completion_ratio"] = (df["completed_milestones"] / planned).fillna(0.0)

    if "activity_level" not in df.columns:
        df["activity_level"] = "Medium"

    def get_status(row):
        if row["days_since_update"] > 45 and row["completion_ratio"] < 0.5:
            return "At Risk"
        elif 20 <= row["days_since_update"] <= 45:
            return "Needs Attention"
        else:
            return "On Track"

    df["status"] = df.apply(get_status, axis=1)

    df["risk_score"] = df.apply(calculate_risk_score, axis=1)
    return df
