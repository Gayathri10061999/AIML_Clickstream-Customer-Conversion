import pandas as pd


def create_features(df):
    df = df.copy()

    if "exit_time" in df.columns and "entry_time" in df.columns:
        df["session_duration"] = df["exit_time"] - df["entry_time"]

    if "page_views" in df.columns:
        df["pages_per_minute"] = df["page_views"] / (df.get("session_duration", 1) + 1)
        df["bounce_flag"] = (df["page_views"] == 1).astype(int)

    if "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin(["Sat", "Sun"]).astype(int)

    return df
