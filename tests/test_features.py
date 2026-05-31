import numpy as np
import pandas as pd

from actor_enriched_forecasting.features import feature_engineering, FeatureConfig


def sample_frame():
    n = 40
    return pd.DataFrame({
        "Avg_Elapsed_Time": np.arange(n, dtype=float),
        "Avg_Remaining_Time": np.arange(n, 0, -1, dtype=float),
        "Cum_Count_C": np.arange(n, dtype=float),
        "Cum_Count_I": np.ones(n),
        "Cum_Count_HI": np.ones(n) * 2,
        "Cum_Count_HB": np.ones(n) * 3,
        "Cum_Time_C_seconds": np.arange(n, dtype=float),
        "Cum_Time_I_seconds": np.ones(n),
        "Cum_Time_HI_seconds": np.ones(n) * 2,
        "Cum_Time_HB_seconds": np.ones(n) * 3,
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))


def test_remaining_time_not_used_as_input():
    frame, baseline, actor = feature_engineering(sample_frame(), "Avg_Remaining_Time", 1, FeatureConfig())
    assert not any("Avg_Remaining_Time" in name for name in baseline + actor)
    assert frame["target_mode"].eq("level").all()


def test_elapsed_time_uses_change_target():
    frame, _, _ = feature_engineering(sample_frame(), "Avg_Elapsed_Time", 1, FeatureConfig())
    assert frame["target_mode"].eq("change").all()
    assert (frame["target"] == 1.0).all()
