def calculate_risk_score(row):
    """
    Calculates a risk score (0–100) based on:
    - Silence duration
    - Completion ratio
    - Activity level
    """

    score = 0

    # 1. Silence factor (time since last update)
    if row["days_since_update"] > 60:
        score += 40
    elif row["days_since_update"] > 30:
        score += 25

    # 2. Completion factor
    if row["completion_ratio"] < 0.3:
        score += 30
    elif row["completion_ratio"] < 0.6:
        score += 15

    # 3. Activity factor
    if row["activity_level"] == "Low":
        score += 20
    elif row["activity_level"] == "Medium":
        score += 10

    return min(score, 100)
