from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from actor_enriched_forecasting.config import load_dataset_config, load_experiment_config
from actor_enriched_forecasting.features import FeatureConfig
from actor_enriched_forecasting.io import load_daily_series, chronological_split
from actor_enriched_forecasting.models.classical import ClassicalConfig, run_classical_holdout

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["BPIC2017", "BPIC2012", "BPIC2011"])
args = parser.parse_args()

dataset_cfg = load_dataset_config(args.dataset, ROOT / "configs/datasets.yaml")
exp_cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
df = load_daily_series(ROOT / dataset_cfg["processed_csv"], exp_cfg["experiment"]["date_column"])
train, test = chronological_split(df, exp_cfg["experiment"]["test_fraction"])

features = exp_cfg["features"]
feature_cfg = FeatureConfig(
    target_lags=tuple(features["target_lags"]),
    actor_lags=tuple(features["actor_lags"]),
    rolling_windows=tuple(features["rolling_windows"]),
    include_causal_peak_flag=features["include_causal_peak_flag"],
)
model_cfg = ClassicalConfig(
    sarimax_order=tuple(dataset_cfg["sarimax_order"]),
    sarimax_seasonal_order=tuple(dataset_cfg["sarimax_seasonal_order"]),
    max_features=features["max_selected_regressors"],
)

results = run_classical_holdout(
    args.dataset, train, test,
    exp_cfg["experiment"]["targets"],
    exp_cfg["experiment"]["horizons"],
    model_cfg, feature_cfg
)
output = ROOT / "results/generated" / args.dataset
output.mkdir(parents=True, exist_ok=True)
results.to_csv(output / "classical_actor_vs_baseline_results.csv", index=False)
print(results.to_string(index=False))
print(f"\nSaved: {output / 'classical_actor_vs_baseline_results.csv'}")
