import streamlit as st


def _json_safe_value(v):
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return v


def _row_to_json_safe_dict(row):
    d = row.to_dict()
    return {k: _json_safe_value(v) for k, v in d.items()}

def show_project_details(df):
    project = st.selectbox(
        "Select a project to inspect",
        df["project_name"].unique()
    )

    row = df[df["project_name"] == project].iloc[0]

    st.markdown("### 📌 Project Overview")
    st.json(_row_to_json_safe_dict(row))

    st.markdown("### ⚠️ Risk Interpretation")
    if row["risk_score"] > 70:
        st.error("High risk – immediate attention recommended")
    elif row["risk_score"] > 40:
        st.warning("Moderate risk – monitoring required")
    else:
        st.success("Low risk – project is stable")
    