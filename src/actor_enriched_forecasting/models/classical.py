from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..evaluation import baseline_actor_result_row
from ..features import FeatureConfig, actual_future_target, feature_engineering, reconstruct_prediction


@dataclass(frozen=True)
class ClassicalConfig:
    sarimax_order: tuple[int, int, int] = (1, 0, 0)
    sarimax_seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    max_features: int = 25


def screen_features(X_train, y_train, X_test, max_features: int):
    train = X_train.replace([np.inf, -np.inf], np.nan).copy()
    test = X_test.replace([np.inf, -np.inf], np.nan).copy()
    usable = [c for c in train if train[c].notna().any() and train[c].nunique(dropna=True) > 1]
    train, test = train[usable], test[usable]
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    train_scaled = scaler.fit_transform(imputer.fit_transform(train))
    test_scaled = scaler.transform(imputer.transform(test))
    k = min(max_features, train_scaled.shape[1])
    scores, _ = f_regression(train_scaled, y_train)
    selected = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1][:k]
    names = [usable[i] for i in selected]
    return (
        pd.DataFrame(train_scaled[:, selected], columns=names, index=X_train.index),
        pd.DataFrame(test_scaled[:, selected], columns=names, index=X_test.index),
        names,
    )


def fit_sarimax(X_train, y_train, X_test, config: ClassicalConfig):
    fitted = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=config.sarimax_order,
        seasonal_order=config.sarimax_seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=300)
    return np.asarray(fitted.forecast(steps=len(X_test), exog=X_test)), fitted


def fit_arx_hac(X_train, y_train, X_test, horizon: int):
    train = sm.add_constant(X_train, has_constant="add")
    test = sm.add_constant(X_test, has_constant="add")
    fitted = sm.OLS(y_train, train).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, horizon)})
    return np.asarray(fitted.predict(test)), fitted


def run_classical_holdout(
    dataset: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list[str],
    horizons: list[int],
    model_config: ClassicalConfig,
    feature_config: FeatureConfig | None = None,
    context_rows: int = 40,
) -> pd.DataFrame:
    rows: list[dict] = []

    for target in targets:
        for horizon in horizons:
            train_fe, baseline_features, actor_features = feature_engineering(
                train_df, target, horizon, feature_config
            )
            context = pd.concat([train_df.tail(context_rows), test_df])
            test_fe, _, _ = feature_engineering(context, target, horizon, feature_config)
            test_fe = test_fe.loc[test_df.index.intersection(test_fe.index)].copy()

            y_train = train_fe["target"]
            y_true = actual_future_target(test_fe, target)
            predictions: dict[tuple[str, str], np.ndarray] = {}

            for feature_set, feature_names in {
                "Baseline": baseline_features,
                "Actor": baseline_features + actor_features,
            }.items():
                X_train, X_test, _ = screen_features(
                    train_fe[feature_names], y_train, test_fe[feature_names], model_config.max_features
                )
                output, _ = fit_sarimax(X_train, y_train, X_test, model_config)
                predictions[("SARIMAX", feature_set)] = reconstruct_prediction(output, test_fe, target)

                output, _ = fit_arx_hac(X_train, y_train, X_test, horizon)
                predictions[("ARX-HAC", feature_set)] = reconstruct_prediction(output, test_fe, target)

            for model in ("SARIMAX", "ARX-HAC"):
                rows.append(
                    baseline_actor_result_row(
                        model, dataset, target, horizon, y_true,
                        predictions[(model, "Baseline")], predictions[(model, "Actor")]
                    )
                )

    return pd.DataFrame(rows)
