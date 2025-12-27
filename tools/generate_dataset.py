import random
from datetime import date, timedelta

import pandas as pd


def _choice_weighted(values, weights):
    return random.choices(values, weights=weights, k=1)[0]


def generate_projects_dataset(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)

    sectors = [
        "Education",
        "Health",
        "Infrastructure",
        "IT",
        "Manufacturing",
        "Retail",
        "Energy",
        "Agriculture",
        "Logistics",
        "Finance",
        "Tourism",
        "Environment",
        "Training",
        "SME",
        "Culture",
        "Sports",
        "Public Services",
        "Water & Sanitation",
    ]

    org_types = ["Government", "Private", "NGO", "University"]
    regions = [
        "Cairo",
        "Giza",
        "Alexandria",
        "Delta",
        "Suez Canal",
        "Upper Egypt",
        "Red Sea",
        "Sinai",
    ]

    approval_stages = [
        "Initiation",
        "Requirements",
        "Procurement",
        "Execution",
        "UAT",
        "Handover",
        "Closed",
    ]

    update_channels = ["Meeting", "Email", "System", "Field Visit", "Phone"]

    today = date(2025, 12, 26)

    rows = []
    for i in range(1, n_rows + 1):
        project_id = f"P{i:03d}"
        sector = random.choice(sectors)

        # Activity correlates with latency to update
        activity_level = _choice_weighted(
            ["Low", "Medium", "High"],
            [0.35, 0.45, 0.20],
        )

        if activity_level == "High":
            days_since_update = random.randint(0, 28)
        elif activity_level == "Medium":
            days_since_update = random.randint(7, 60)
        else:
            days_since_update = random.randint(20, 140)

        last_update = today - timedelta(days=days_since_update)

        # Start date earlier than last update
        start_date = last_update - timedelta(days=random.randint(30, 420))

        planned = random.randint(3, 18)

        # Completed milestones correlated with activity and days since update
        if activity_level == "High":
            completed = random.randint(max(0, planned - 4), planned)
        elif activity_level == "Medium":
            completed = random.randint(max(0, planned - 10), min(planned, planned - 1))
        else:
            completed = random.randint(0, max(1, planned - 12))

        # Optional enterprise-style fields (safe for current app)
        org_type = _choice_weighted(org_types, [0.25, 0.45, 0.20, 0.10])
        region = _choice_weighted(regions, [0.35, 0.20, 0.12, 0.12, 0.07, 0.08, 0.04, 0.02])
        priority = _choice_weighted(["Low", "Medium", "High"], [0.25, 0.55, 0.20])

        budget_total = float(_choice_weighted([0.5, 1, 3, 5, 10, 25, 50], [2, 4, 5, 4, 3, 2, 1]))
        budget_total *= 1_000_000
        progress_ratio = completed / planned if planned else 0
        noise = random.uniform(-0.15, 0.20)
        budget_spent = max(0.0, min(budget_total, budget_total * max(0.0, progress_ratio + noise)))

        payment_delay_days = int(_choice_weighted([0, 7, 14, 30, 60, 90], [6, 3, 3, 2, 1, 0.5]))
        dependency_count = int(_choice_weighted([0, 1, 2, 3, 5, 8], [4, 4, 3, 2, 1, 0.5]))
        team_size = int(_choice_weighted([3, 5, 8, 12, 20, 35], [2, 4, 4, 3, 2, 1]))

        approval_stage = random.choice(approval_stages)
        approval_last_moved = last_update - timedelta(days=random.randint(0, 45))

        update_channel = random.choice(update_channels)

        risk_notes = _choice_weighted(
            [
                "",
                "Awaiting approval",
                "Vendor delay",
                "Payment backlog",
                "Scope clarification needed",
                "Resource constraints",
                "External dependency blocking",
                "Seasonality impact",
            ],
            [5, 2, 2, 1.5, 1.5, 1.3, 1.0, 0.7],
        )

        project_name = f"{sector} Initiative #{random.randint(10, 999)}"

        rows.append(
            {
                "project_id": project_id,
                "project_name": project_name,
                "sector": sector,
                "start_date": start_date.isoformat(),
                "last_update": last_update.isoformat(),
                "planned_milestones": planned,
                "completed_milestones": completed,
                "activity_level": activity_level,
                # Optional extra fields
                "org_type": org_type,
                "region": region,
                "priority": priority,
                "budget_total": round(budget_total, 2),
                "budget_spent": round(budget_spent, 2),
                "payment_delay_days": payment_delay_days,
                "dependency_count": dependency_count,
                "team_size": team_size,
                "approval_stage": approval_stage,
                "approval_last_moved": approval_last_moved.isoformat(),
                "update_channel": update_channel,
                "risk_notes": risk_notes,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = generate_projects_dataset(n_rows=200, seed=42)
    df.to_csv("projects_dataset.csv", index=False)
    print(f"Wrote {len(df)} rows to projects_dataset.csv")


if __name__ == "__main__":
    main()
