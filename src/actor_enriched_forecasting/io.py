from __future__ import annotations

from pathlib import Path
import pandas as pd


REQUIRED_TARGETS = ["Avg_Elapsed_Time", "Avg_Remaining_Time"]


def load_daily_series(csv_path: str | Path, date_column: str = "date") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found. Columns: {list(df.columns)}")
    missing = [col for col in REQUIRED_TARGETS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    df[date_column] = pd.to_datetime(df[date_column], errors="raise")
    return df.sort_values(date_column).set_index(date_column)


def trim_trailing_inactive_days(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    present = [col for col in REQUIRED_TARGETS if col in output.columns]
    while len(output) > 1 and output[present].iloc[-1].fillna(0).sum() == 0:
        output = output.iloc[:-1]
    return output


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    n_test = max(1, int(len(df) * test_fraction))
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()
