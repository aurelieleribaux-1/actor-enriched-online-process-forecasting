from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import joblib
import pandas as pd
import shap

from actor_enriched_forecasting.interpretability import mean_absolute_shap_table, aggregate_shap_importance

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--features", required=True)
parser.add_argument("--target", default="unspecified")
parser.add_argument("--horizon", type=int, default=1)
args = parser.parse_args()

model = joblib.load(args.model)
X = pd.read_csv(args.features)
explainer = shap.Explainer(model, X)
values = explainer(X, check_additivity=False)
local = mean_absolute_shap_table(values.values, list(X.columns), args.target, args.horizon)
global_importance = aggregate_shap_importance(local)
out = ROOT / "results/generated/lightgbm_shap_importance.csv"
out.parent.mkdir(parents=True, exist_ok=True)
global_importance.to_csv(out, index=False)
print(global_importance.head(20).to_string(index=False))
print(f"Saved: {out}")
