def simulate_improvement(row, days_reduced=20, milestones_added=1):
    new_days = max(row["days_since_update"] - days_reduced, 0)
    new_completion = min(
        (row["completed_milestones"] + milestones_added) /
        row["planned_milestones"], 1
    )

    return {
        "simulated_days_since_update": new_days,
        "simulated_completion_ratio": new_completion
    }
