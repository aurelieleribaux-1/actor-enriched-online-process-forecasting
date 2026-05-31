from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..features import FeatureConfig, actual_future_target, feature_engineering, reconstruct_prediction


def _create_model(model_name: str, params: dict, seed: int):
    if model_name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, objective="reg:squarederror", **params)
    if model_name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, verbosity=-1, **params)
    raise ValueError(f"Unsupported tree model: {model_name}")


def run_tree_holdout(
    dataset: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    parameters: dict,
    seeds: list[int],
    targets: list[str],
    horizons: list[int],
    feature_config: FeatureConfig | None = None,
    context_rows: int = 40,
) -> pd.DataFrame:
    rows = []
    for target in targets:
        for horizon in horizons:
            train_fe, baseline_features, actor_features = feature_engineering(train_df, target, horizon, feature_config)
            test_fe, _, _ = feature_engineering(pd.concat([train_df.tail(context_rows), test_df]), target, horizon, feature_config)
            test_fe = test_fe.loc[test_df.index.intersection(test_fe.index)]
            y_train = train_fe["target"]
            y_true = actual_future_target(test_fe, target)

            for seed in seeds:
                row = {"Dataset": dataset, "Target": target, "Model": model_name, "Horizon": horizon, "Seed": seed}
                for label, columns in {"Baseline": baseline_features, "Actor": baseline_features + actor_features}.items():
                    model = _create_model(model_name, parameters, seed)
                    model.fit(train_fe[columns], y_train)
                    model_output = model.predict(test_fe[columns])
                    prediction = reconstruct_prediction(model_output, test_fe, target)
                    row[f"RMSE_{label}"] = np.sqrt(mean_squared_error(y_true, prediction))
                    row[f"MAE_{label}"] = mean_absolute_error(y_true, prediction)
                    row[f"R2_{label}"] = r2_score(y_true, prediction)
                row["Delta_RMSE"] = row["RMSE_Baseline"] - row["RMSE_Actor"]
                row["Delta_MAE"] = row["MAE_Baseline"] - row["MAE_Actor"]
                row["Delta_R2"] = row["R2_Actor"] - row["R2_Baseline"]
                rows.append(row)
    return pd.DataFrame(rows)
