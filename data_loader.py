import pandas as pd
import sqlite3
from pathlib import Path

def load_data(source="csv"):
    if source == "db" and Path("projects.db").exists():
        conn = sqlite3.connect("projects.db")
        df = pd.read_sql("SELECT * FROM projects", conn)
        conn.close()
        return df
    else:
        return pd.read_csv("projects_dataset.csv")
