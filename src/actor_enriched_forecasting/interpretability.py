from __future__ import annotations

import numpy as np
import pandas as pd


def mean_absolute_shap_table(shap_values, feature_names: list[str], target: str, horizon: int, seed: int | None = None):
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[..., 0]
    importance = np.mean(np.abs(values), axis=0)
    return pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": importance,
        "Target": target,
        "Horizon": horizon,
        "Seed": seed,
    })


def aggregate_shap_importance(shap_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        shap_rows.groupby("Feature", as_index=False)["MeanAbsSHAP"]
        .mean()
        .sort_values("MeanAbsSHAP", ascending=False)
        .reset_index(drop=True)
    )


def feature_group(feature_name: str) -> str:
    actor_markers = ("Cum_Count_", "Cum_Time_")
    return "Actor behavior" if feature_name.startswith(actor_markers) else "Process state"
