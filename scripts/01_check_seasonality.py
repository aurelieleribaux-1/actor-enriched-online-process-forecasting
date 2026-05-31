from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf

from actor_enriched_forecasting.config import load_dataset_config, load_experiment_config
from actor_enriched_forecasting.io import load_daily_series, chronological_split

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["BPIC2017", "BPIC2012", "BPIC2011"])
args = parser.parse_args()

dataset_cfg = load_dataset_config(args.dataset, ROOT / "configs/datasets.yaml")
exp_cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
df = load_daily_series(ROOT / dataset_cfg["processed_csv"], exp_cfg["experiment"]["date_column"])
train, _ = chronological_split(df, exp_cfg["experiment"]["test_fraction"])
out = ROOT / "figures/generated" / "seasonality" / args.dataset
out.mkdir(parents=True, exist_ok=True)

for target in exp_cfg["experiment"]["targets"]:
    series = train[target].dropna()
    result = STL(series, period=7, robust=True).fit()
    strength = max(0.0, 1 - np.var(result.resid) / np.var(result.seasonal + result.resid))
    print(f"{args.dataset} | {target} | weekly seasonal strength={strength:.3f} | lag7 ACF={series.autocorr(7):.3f}")
    fig = result.plot()
    fig.suptitle(f"{args.dataset}: weekly STL decomposition of {target}")
    fig.savefig(out / f"{target}_stl.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(series, lags=min(35, len(series)//2 - 1), ax=ax)
    for lag in (7, 14, 21, 28):
        ax.axvline(lag, linestyle="--", alpha=0.5)
    fig.savefig(out / f"{target}_acf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
