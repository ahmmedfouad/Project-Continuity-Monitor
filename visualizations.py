import plotly.express as px
import pandas as pd


def load_history(filename="history_log.json"):
    try:
        return pd.read_json(filename)
    except ValueError:
        return pd.DataFrame()
    except FileNotFoundError:
        return pd.DataFrame()


def _normalize_history_df(history_df):
    if history_df is None or len(history_df) == 0:
        return pd.DataFrame(columns=["timestamp", "at_risk", "average_risk"])

    df = history_df.copy()
    if "summary" in df.columns:
        summary = pd.json_normalize(df["summary"])
        df = pd.concat([df.drop(columns=["summary"]), summary], axis=1)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "at_risk" not in df.columns:
        df["at_risk"] = pd.NA
    if "average_risk" not in df.columns:
        df["average_risk"] = pd.NA

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df

def status_pie_chart(df):
    fig = px.pie(
        df,
        names="status",
        title="Project Status Distribution",
        hole=0.4
    )
    return fig

def sector_bar_chart(df):
    sector_counts = df.groupby("sector").size().reset_index(name="count")
    fig = px.bar(
        sector_counts,
        x="sector",
        y="count",
        title="Projects by Sector",
        text="count"
    )
    return fig

def silence_timeline(df):
    fig = px.histogram(
        df,
        x="days_since_update",
        nbins=10,
        title="Project Silence Duration (Days)"
    )
    return fig

def completion_scatter(df):
    fig = px.scatter(
        df,
        x="days_since_update",
        y="completion_ratio",
        color="status",
        title="Completion vs Silence Pattern",
        hover_data=["project_name"]
    )
    return fig


def history_at_risk_trend(history_df):
    df = _normalize_history_df(history_df)
    fig = px.line(
        df,
        x="timestamp",
        y="at_risk",
        markers=True,
        title="At-Risk Projects Over Time"
    )
    return fig


def history_average_risk_trend(history_df):
    df = _normalize_history_df(history_df)
    fig = px.line(
        df,
        x="timestamp",
        y="average_risk",
        markers=True,
        title="Average Risk Score Over Time"
    )
    return fig
