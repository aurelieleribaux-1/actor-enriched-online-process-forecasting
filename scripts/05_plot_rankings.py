from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import pandas as pd
from actor_enriched_forecasting.plotting import add_ranking_columns, make_ranking_figure

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

df = pd.read_csv(ROOT / args.input)
ranked, summary = add_ranking_columns(df)
output_dir = ROOT / "figures/generated"
output_dir.mkdir(parents=True, exist_ok=True)
ranked.to_csv(ROOT / "results/generated/final_model_results_with_ranks.csv", index=False)
summary.to_csv(ROOT / "results/generated/model_rank_by_horizon.csv", index=False)
make_ranking_figure(summary, output_dir / "ranking_and_significance_by_horizon.png")
print(summary.to_string(index=False))
print(f"Saved figure outputs to {output_dir}")
