from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

"""Entry point for final recurrent attention model runs.

The reusable model architecture is provided in
src/actor_enriched_forecasting/models/rnn.py. Populate this runner with the final
training-only selected configuration before generating publication results; the
output-stripped development notebooks in notebooks/archive/ contain the original
cross-validation and multi-seed orchestration.
"""
import argparse
from actor_enriched_forecasting.config import load_experiment_config

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["BPIC2017", "BPIC2012", "BPIC2011"])
args = parser.parse_args()
config = load_experiment_config(ROOT / "configs/experiment.yaml")
print(f"RNN final-run scaffold for {args.dataset}.")
print("Final recurrent configuration grid is documented in configs/experiment.yaml.")
print("Synchronize selected trained configuration here before publication result generation.")
