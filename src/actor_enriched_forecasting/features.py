from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd

from .io import trim_trailing_inactive_days


ACTOR_VARS = [
    "Cum_Count_C", "Cum_Count_I", "Cum_Count_HI", "Cum_Count_HB",
    "Cum_Time_C_seconds", "Cum_Time_I_seconds",
    "Cum_Time_HI_seconds", "Cum_Time_HB_seconds",
]


@dataclass(frozen=True)
class FeatureConfig:
    target_lags: tuple[int, ...] = tuple(range(1, 8))
    actor_lags: tuple[int, ...] = tuple(range(1, 8))
    rolling_windows: tuple[int, ...] = (3, 7, 14)
    include_causal_peak_flag: bool = False


def _add_history_features(
    frame: pd.DataFrame,
    variable: str,
    lags: Iterable[int],
    windows: Iterable[int],
    include_current: bool,
    include_causal_peak_flag: bool = False,
) -> list[str]:
    features: list[str] = []

    if include_current:
        features.append(variable)

    for lag in lags:
        name = f"{variable}_lag{lag}"
        frame[name] = frame[variable].shift(lag)
        features.append(name)

    for window in windows:
        prior = frame[variable].shift(1)
        mean_name = f"{variable}_rolling_mean{window}"
        std_name = f"{variable}_rolling_std{window}"
        max_name = f"{variable}_rolling_max{window}"
        frame[mean_name] = prior.rolling(window).mean()
        frame[std_name] = prior.rolling(window).std()
        frame[max_name] = prior.rolling(window).max()
        features.extend([mean_name, std_name, max_name])

    prior = frame[variable].shift(1)
    z_name = f"{variable}_zscore7"
    frame[z_name] = (prior - prior.rolling(7).mean()) / prior.rolling(7).std()
    features.append(z_name)

    if include_causal_peak_flag:
        peak_name = f"{variable}_causal_peak_flag7"
        rolling_max = prior.rolling(7).max()
        frame[peak_name] = (prior == rolling_max).astype(float)
        features.append(peak_name)

    return features


def feature_engineering(
    df: pd.DataFrame,
    target_var: str,
    horizon: int,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create leakage-aware tabular features for one target and horizon.

    - Avg_Elapsed_Time is the only process-state input.
    - For Avg_Elapsed_Time, the supervised target is the future change because
      the present elapsed value is observable.
    - For Avg_Remaining_Time, the supervised target is the future level because
      present realized remaining time is not available at prediction time.
    """
    if target_var not in {"Avg_Elapsed_Time", "Avg_Remaining_Time"}:
        raise ValueError(f"Unsupported target variable: {target_var}")
    if horizon < 1:
        raise ValueError("horizon must be positive")

    config = config or FeatureConfig()
    frame = trim_trailing_inactive_days(df)

    if target_var == "Avg_Elapsed_Time":
        frame["target"] = frame[target_var].shift(-horizon) - frame[target_var]
        frame["evaluation_anchor"] = frame[target_var]
        frame["target_mode"] = "change"
    else:
        frame["target"] = frame[target_var].shift(-horizon)
        frame["evaluation_anchor"] = np.nan
        frame["target_mode"] = "level"

    baseline_features = _add_history_features(
        frame=frame,
        variable="Avg_Elapsed_Time",
        lags=config.target_lags,
        windows=config.rolling_windows,
        include_current=False,
        include_causal_peak_flag=config.include_causal_peak_flag,
    )

    actor_features: list[str] = []
    for variable in ACTOR_VARS:
        if variable in frame.columns:
            actor_features.extend(
                _add_history_features(
                    frame=frame,
                    variable=variable,
                    lags=config.actor_lags,
                    windows=config.rolling_windows,
                    include_current=True,
                    include_causal_peak_flag=False,
                )
            )

    required = ["target"] + baseline_features + actor_features
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()

    return frame, list(dict.fromkeys(baseline_features)), list(dict.fromkeys(actor_features))


def reconstruct_prediction(
    model_output: np.ndarray | pd.Series,
    engineered_frame: pd.DataFrame,
    target_var: str,
) -> np.ndarray:
    output = np.asarray(model_output, dtype=float)
    if target_var == "Avg_Elapsed_Time":
        return engineered_frame["evaluation_anchor"].to_numpy(dtype=float) + output
    return output


def actual_future_target(engineered_frame: pd.DataFrame, target_var: str) -> np.ndarray:
    if target_var == "Avg_Elapsed_Time":
        return (
            engineered_frame["evaluation_anchor"].to_numpy(dtype=float)
            + engineered_frame["target"].to_numpy(dtype=float)
        )
    return engineered_frame["target"].to_numpy(dtype=float)
