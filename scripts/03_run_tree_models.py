from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from actor_enriched_forecasting.config import load_dataset_config, load_experiment_config
from actor_enriched_forecasting.features import FeatureConfig
from actor_enriched_forecasting.io import load_daily_series, chronological_split
from actor_enriched_forecasting.models.tree import run_tree_holdout

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["BPIC2017", "BPIC2012", "BPIC2011"])
parser.add_argument("--model", required=True, choices=["XGBoost", "LightGBM"])
args = parser.parse_args()

dataset_cfg = load_dataset_config(args.dataset, ROOT / "configs/datasets.yaml")
exp_cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
df = load_daily_series(ROOT / dataset_cfg["processed_csv"], exp_cfg["experiment"]["date_column"])
train, test = chronological_split(df, exp_cfg["experiment"]["test_fraction"])

# Replace these by the final configuration selected through training-only CV.
params = {
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}
results = run_tree_holdout(
    args.dataset, train, test, args.model, params,
    exp_cfg["experiment"]["seeds"],
    exp_cfg["experiment"]["targets"],
    exp_cfg["experiment"]["horizons"],
    FeatureConfig(
        target_lags=tuple(exp_cfg["features"]["target_lags"]),
        actor_lags=tuple(exp_cfg["features"]["actor_lags"]),
        rolling_windows=tuple(exp_cfg["features"]["rolling_windows"]),
    )
)
output = ROOT / "results/generated" / args.dataset
output.mkdir(parents=True, exist_ok=True)
path = output / f"{args.model.lower()}_multi_seed_holdout.csv"
results.to_csv(path, index=False)
print(f"Saved: {path}")
