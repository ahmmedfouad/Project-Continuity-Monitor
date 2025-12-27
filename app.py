from data_loader import load_data
from analysis import analyze_projects
from datetime import datetime
import json
from visualizations import (
    status_pie_chart,
    sector_bar_chart,
    silence_timeline,
    completion_scatter,
    load_history,
    history_at_risk_trend,
    history_average_risk_trend
)
from project_details import show_project_details
from history_tracker import log_project_state
from history_entities import log_entities_state
from explain import explain_df

import streamlit as st

st.set_page_config(page_title="Project Continuity Monitor", layout="wide")

st.markdown(
        """
        <style>
            /* Prevent top header from being clipped */
            .block-container { padding-top: 5.25rem; }
            div[data-testid="stMetric"] {
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 14px;
                padding: 14px;
            }
            .pcm-hero {
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 18px;
                padding: 18px 18px 8px 18px;
                background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(16,185,129,0.10));
            }
            .pcm-muted { opacity: 0.85; }
            .pcm-badge {
                display: inline-block;
                padding: 3px 10px;
                border-radius: 999px;
                border: 1px solid rgba(255,255,255,0.16);
                background: rgba(255,255,255,0.06);
                font-size: 0.85rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
)


def _safe_for_streamlit_table(df):
    if df is None:
        return df
    safe = df.copy()
    for col in safe.columns:
        series = safe[col]
        if str(series.dtype).startswith("datetime"):
            safe[col] = series.astype(str)
            continue

        # Handle object columns that may contain Timestamp / datetime objects.
        if series.dtype == "object":
            sample = series.dropna().head(25)
            if len(sample) == 0:
                continue
            if any(hasattr(v, "isoformat") for v in sample):
                safe[col] = series.apply(lambda v: v.isoformat() if hasattr(v, "isoformat") else v)
    return safe

st.sidebar.markdown("## Controls")
source = st.sidebar.selectbox(
        "Data Source",
        ["csv", "db"],
        help="Choose where to load project data from",
)

df = analyze_projects(load_data(source))

st.markdown(
        """
        <div class="pcm-hero">
            <div style="display:flex; align-items:center; justify-content:space-between; gap: 12px;">
                <div>
                    <h2 style="margin: 0; padding: 0;">Project Continuity Monitor</h2>
                    <div class="pcm-muted" style="margin-top: 4px;">Early warning signals for project stagnation — decision support, not judgment.</div>
                </div>
                <div class="pcm-badge">Read-only • MVP</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)

st.write("")

st.sidebar.markdown("---")
sectors = sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns else []
selected_sectors = st.sidebar.multiselect("Sector filter", sectors, default=sectors)

status_options = ["On Track", "Needs Attention", "At Risk"]
selected_status = st.sidebar.multiselect("Status filter", status_options, default=status_options)

min_risk, max_risk = int(df["risk_score"].min()), int(df["risk_score"].max())
risk_range = st.sidebar.slider("Risk score range", 0, 100, (min_risk, max_risk))

filtered_df = df.copy()
if selected_sectors:
    filtered_df = filtered_df[filtered_df["sector"].isin(selected_sectors)]
if selected_status:
    filtered_df = filtered_df[filtered_df["status"].isin(selected_status)]
filtered_df = filtered_df[(filtered_df["risk_score"] >= risk_range[0]) & (filtered_df["risk_score"] <= risk_range[1])]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing {len(filtered_df)} of {len(df)} projects")

total = len(filtered_df)
at_risk = int((filtered_df["status"] == "At Risk").sum())
needs_attention = int((filtered_df["status"] == "Needs Attention").sum())
avg_risk = float(filtered_df["risk_score"].mean()) if total else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Projects", total)
k2.metric("At Risk", at_risk)
k3.metric("Needs Attention", needs_attention)
k4.metric("Avg Risk", f"{avg_risk:.1f} / 100")

top_risky = filtered_df.sort_values("risk_score", ascending=False).head(5)

tab_dashboard, tab_details, tab_history, tab_audit = st.tabs(
    ["📊 Dashboard", "🔎 Project Details", "📚 History", "🧾 Audit"],
)

with tab_dashboard:
    st.subheader("Overview")
    overview_df = filtered_df[[
            "project_name",
            "sector",
            "days_since_update",
            "completion_ratio",
            "activity_level",
            "risk_score",
            "status",
        ]]
    st.dataframe(
        _safe_for_streamlit_table(overview_df),
        width="stretch",
        hide_index=True,
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Visual Insights")
        v1, v2 = st.columns(2)
        v1.plotly_chart(status_pie_chart(filtered_df), width="stretch")
        v2.plotly_chart(sector_bar_chart(filtered_df), width="stretch")

        v3, v4 = st.columns(2)
        v3.plotly_chart(silence_timeline(filtered_df), width="stretch")
        v4.plotly_chart(completion_scatter(filtered_df), width="stretch")

    with c2:
        st.subheader("Highest Risk")
        st.dataframe(
            _safe_for_streamlit_table(top_risky[["project_name", "risk_score", "status"]]),
            width="stretch",
            hide_index=True,
        )

with tab_details:
    st.subheader("Drill-down")
    show_project_details(filtered_df if len(filtered_df) else df)

    st.subheader("Explainability")
    st.caption("Explain the current risk score using transparent rule components.")
    try:
        expl = explain_df(filtered_df if len(filtered_df) else df)
        st.dataframe(_safe_for_streamlit_table(expl), width="stretch", hide_index=True)
    except Exception as e:
        st.warning(f"Explainability unavailable: {e}")




with tab_history:
    st.subheader("Historical Tracking")
    st.caption("Track system-level signals across time using read-only snapshots.")

    actions1, actions2 = st.columns([1, 2])
    with actions1:
        if st.button("Save Snapshot", type="primary"):
            try:
                log_project_state(df, filename="history_log.json")
                log_entities_state(df, filename="history_entities_log.json")
                st.success("History snapshot saved.")
            except Exception as e:
                st.warning(f"Could not save history snapshot: {e}")

    history_df = load_history("history_log.json")
    if history_df is None or len(history_df) == 0:
        st.info("No history snapshots yet. Click 'Save Snapshot' to start.")
    else:
        h1, h2 = st.columns(2)
        h1.plotly_chart(history_at_risk_trend(history_df), width="stretch")
        h2.plotly_chart(history_average_risk_trend(history_df), width="stretch")

        st.subheader("Raw history")
        try:
            with open("history_log.json") as f:
                st.json(json.load(f))
        except FileNotFoundError:
            st.info("history_log.json not found.")

with tab_audit:
    st.subheader("Integrity Audit Log")
    st.caption("Immutable-style logging for traceability.")

    if st.button("Log Current Status Snapshot"):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_projects": int(len(df)),
                "at_risk": int((df["status"] == "At Risk").sum()),
                "needs_attention": int((df["status"] == "Needs Attention").sum()),
            },
        }

        try:
            with open("audit_log.json", "r+") as f:
                data = json.load(f)
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=2)
            st.success("Snapshot logged successfully.")
        except FileNotFoundError:
            with open("audit_log.json", "w") as f:
                json.dump([log_entry], f, indent=2)
            st.success("Snapshot logged successfully.")

    try:
        with open("audit_log.json") as f:
            st.json(json.load(f))
    except FileNotFoundError:
        st.info("No audit log yet. Click 'Log Current Status Snapshot' to create one.")


st.markdown("---")
st.subheader("AI Insights (Optional)")
st.caption("This section is optional. The app runs even if the AI module is missing.")

try:
    from ai.engine import run_ai_inference

    ai_enabled = True
except Exception:
    ai_enabled = False

if not ai_enabled:
    st.info("AI module not available. You can deploy/run the app without it.")
else:
    try:
        import json as _json
        from pathlib import Path as _Path

        model_path = _Path("ai/model.json")
        if model_path.exists():
            try:
                model_meta = _json.loads(model_path.read_text()).get("training", {})
                metrics = model_meta.get("metrics")
                if metrics:
                    st.caption(f"Model-backed forecast enabled (MAE={metrics.get('mae'):.2f}, RMSE={metrics.get('rmse'):.2f}).")
                else:
                    st.caption("Model-backed forecast enabled.")
            except Exception:
                st.caption("Model-backed forecast enabled.")
        else:
            st.caption("No trained model found. Forecast uses rule-based fallback.")

        ai_df = run_ai_inference(df)
        t1, t2, t3 = st.columns(3)
        t1.metric("AI Coverage", f"{len(ai_df)} projects")
        t2.metric("Avg Forecast (14d)", f"{ai_df['forecast_risk_score_14d'].mean():.1f} / 100")

        st.dataframe(
            _safe_for_streamlit_table(
                ai_df.sort_values("forecast_risk_score_14d", ascending=False).head(15)
            ),
            width="stretch",
            hide_index=True,
        )

        st.caption("Open a row to see root causes and recommendations.")
        selected = st.selectbox(
            "AI details for project",
            ai_df["project_id"].tolist(),
        )
        row = ai_df[ai_df["project_id"] == selected].iloc[0]
        st.write("Root causes:")
        st.write(row["root_causes"])
        st.write("Recommendations:")
        st.write(row["recommendations"])
    except Exception as e:
        st.warning(f"AI inference failed safely: {e}")

