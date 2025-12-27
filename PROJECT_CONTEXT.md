# Project Context – Project Continuity Monitor

## 1. Project Purpose
This is a real, working MVP (not a paper project).
The goal is to detect early signs of project stagnation **before failure occurs**.
It is a decision-support system, not an enforcement or evaluation system.

## 2. Core Philosophy
- Simple logic, strong insight
- Read-only analysis (no actions, no edits)
- Human decision remains final
- Early warning, not judgment
- Modular design to avoid breaking the app

## 3. Current Tech Stack
- Python
- Streamlit
- Pandas
- Plotly
- Optional SQLite
- JSON-based immutable logs (instead of blockchain for now)

## 4. Project Structure
# Project Context – Project Continuity Monitor

## 1. Project Purpose
This is a real, working MVP (not a paper project).
The goal is to detect early signs of project stagnation **before failure occurs**.
It is a decision-support system, not an enforcement or evaluation system.

## 2. Core Philosophy
- Simple logic, strong insight
- Read-only analysis (no actions, no edits)
- Human decision remains final
- Early warning, not judgment
- Modular design to avoid breaking the app

## 3. Current Tech Stack
- Python
- Streamlit
- Pandas
- Plotly
- Optional SQLite
- JSON-based immutable logs (instead of blockchain for now)

## 4. Project Structure
├── app.py # Main UI (do not break)
├── data_loader.py # CSV / DB loader
├── analysis.py # Core analysis logic
├── risk_engine.py # Risk scoring engine (0–100)
├── visualizations.py # All charts (Plotly)
├── project_details.py # Drill-down view per project
├── history_tracker.py # Historical snapshots (JSON)
├── simulation.py # What-if simulation (future step)
├── projects_dataset.csv # Demo data
├── audit_log.json
└── requirements.txt



## 5. Implemented Features (DO NOT REWRITE)
- Status classification: On Track / Needs Attention / At Risk
- Risk Score (0–100) based on:
  - Silence duration
  - Completion ratio
  - Activity level
- Dashboard with charts
- Project drill-down view
- All features are modular and isolated

## 5.1 Dataset Contract (CSV)
The app requires these columns to run:
- `project_name`
- `sector`
- `last_update`
- `planned_milestones`
- `completed_milestones`
- `activity_level`

Optional columns may exist (used for future steps like richer risk factors and AI) and must not break the app if missing:
- `org_type`, `region`, `priority`
- `budget_total`, `budget_spent`
- `payment_delay_days`, `dependency_count`, `team_size`
- `approval_stage`, `approval_last_moved`
- `update_channel`, `risk_notes`

## 6. Design Rules (VERY IMPORTANT)
- Do NOT put new logic inside app.py
- Any new feature must be in a separate file
- app.py only wires things together
- If a feature is optional, it must fail safely
- No breaking changes

## 7. Current Stage
STEP 2 completed successfully:
- Risk Scoring Engine
- Project Details (Drill-down)

Next planned steps:
- STEP 3: Historical Tracking visualization
- STEP 4: What-if Simulation UI
- STEP 5: Exportable Executive Report

## 8. How the AI Agent Should Help
- Extend the system step by step
- Explain WHY each design choice is made
- Assume the user may be questioned technically
- Prefer clarity over cleverness
- Treat this as a real product, not a demo

## 9. Non-Goals
- No enforcement
- No financial transactions
- No sensitive government data
- No overengineering
