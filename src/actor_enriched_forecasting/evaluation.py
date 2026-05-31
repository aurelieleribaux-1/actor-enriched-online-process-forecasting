from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_forecast(y_true, y_pred) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def diebold_mariano_hac_pvalue(
    y_true,
    pred_baseline,
    pred_actor,
    loss: str = "squared",
    max_lag: int = 1,
) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    baseline = np.asarray(pred_baseline, dtype=float)
    actor = np.asarray(pred_actor, dtype=float)

    if loss == "absolute":
        differential = np.abs(y_true - baseline) - np.abs(y_true - actor)
    elif loss == "squared":
        differential = (y_true - baseline) ** 2 - (y_true - actor) ** 2
    else:
        raise ValueError("loss must be 'absolute' or 'squared'.")

    differential = differential[~np.isnan(differential)]
    n = len(differential)
    if n < 5 or np.std(differential) == 0:
        return np.nan, np.nan

    centered = differential - np.mean(differential)
    variance = np.mean(centered * centered)
    lag_limit = min(max_lag, n - 1)
    for lag in range(1, lag_limit + 1):
        weight = 1 - lag / (lag_limit + 1)
        gamma = np.mean(centered[lag:] * centered[:-lag])
        variance += 2 * weight * gamma

    if variance <= 0:
        return np.nan, np.nan
    statistic = np.mean(differential) / np.sqrt(variance / n)
    return float(statistic), float(1 - norm.cdf(statistic))


def paired_error_tests(y_true, pred_baseline, pred_actor, horizon: int) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    baseline = np.asarray(pred_baseline, dtype=float)
    actor = np.asarray(pred_actor, dtype=float)
    abs_b = np.abs(y_true - baseline)
    abs_a = np.abs(y_true - actor)
    sq_b = (y_true - baseline) ** 2
    sq_a = (y_true - actor) ** 2

    _, p_mae = wilcoxon(abs_b, abs_a, alternative="greater")
    _, p_rmse = wilcoxon(sq_b, sq_a, alternative="greater")
    dm_sq, dm_sq_p = diebold_mariano_hac_pvalue(y_true, baseline, actor, "squared", max(1, horizon))
    dm_abs, dm_abs_p = diebold_mariano_hac_pvalue(y_true, baseline, actor, "absolute", max(1, horizon))
    return {
        "p_RMSE": float(p_rmse),
        "p_MAE": float(p_mae),
        "DM_Squared_Loss_stat": dm_sq,
        "DM_Squared_Loss_p_value": dm_sq_p,
        "DM_Absolute_Loss_stat": dm_abs,
        "DM_Absolute_Loss_p_value": dm_abs_p,
    }


def baseline_actor_result_row(model: str, dataset: str, target: str, horizon: int, y_true, baseline, actor) -> dict:
    metrics_b = evaluate_forecast(y_true, baseline)
    metrics_a = evaluate_forecast(y_true, actor)
    tests = paired_error_tests(y_true, baseline, actor, horizon)
    return {
        "Dataset": dataset,
        "Target": target,
        "Model": model,
        "Horizon": horizon,
        "RMSE_Baseline": metrics_b["RMSE"],
        "RMSE_Actor": metrics_a["RMSE"],
        "Delta_RMSE": metrics_b["RMSE"] - metrics_a["RMSE"],
        "MAE_Baseline": metrics_b["MAE"],
        "MAE_Actor": metrics_a["MAE"],
        "Delta_MAE": metrics_b["MAE"] - metrics_a["MAE"],
        "R2_Baseline": metrics_b["R2"],
        "R2_Actor": metrics_a["R2"],
        "Delta_R2": metrics_a["R2"] - metrics_b["R2"],
        **tests,
        "RMSE_Significant_Improvement": (
            metrics_b["RMSE"] > metrics_a["RMSE"] and tests["p_RMSE"] < 0.05
        ),
        "MAE_Significant_Improvement": (
            metrics_b["MAE"] > metrics_a["MAE"] and tests["p_MAE"] < 0.05
        ),
    }
