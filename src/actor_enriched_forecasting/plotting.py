from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_ORDER = ["SARIMAX", "ARX-HAC", "XGBoost", "LightGBM", "GRU+Attn", "LSTM+Attn"]


def add_ranking_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = df.copy()
    output["RMSE_Significant_Improvement"] = (output["Delta_RMSE"] > 0) & (output["p_RMSE"] < 0.05)
    output["MAE_Significant_Improvement"] = (output["Delta_MAE"] > 0) & (output["p_MAE"] < 0.05)
    output["Rank_RMSE_Actor"] = output.groupby(["Dataset", "Target", "Horizon"])["RMSE_Actor"].rank(ascending=True)
    output["Rank_MAE_Actor"] = output.groupby(["Dataset", "Target", "Horizon"])["MAE_Actor"].rank(ascending=True)
    summary = output.groupby(["Model", "Horizon"], as_index=False).agg(
        Mean_Rank_RMSE=("Rank_RMSE_Actor", "mean"),
        Mean_Rank_MAE=("Rank_MAE_Actor", "mean"),
        Significant_RMSE_Count=("RMSE_Significant_Improvement", "sum"),
        Total_Comparisons=("RMSE_Significant_Improvement", "count"),
    )
    summary["Significant_RMSE_Share"] = summary["Significant_RMSE_Count"] / summary["Total_Comparisons"]
    return output, summary


def make_ranking_figure(summary: pd.DataFrame, output_path: str | Path) -> None:
    models = [model for model in MODEL_ORDER if model in set(summary["Model"])]
    markers = dict(zip(MODEL_ORDER, ["o", "s", "^", "D", "P", "X"]))
    linestyles = dict(zip(MODEL_ORDER, ["-", "--", "-.", ":", "-", "--"]))
    offsets = dict(zip(MODEL_ORDER, [-0.15, -0.09, -0.03, 0.03, 0.09, 0.15]))

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    for model in models:
        group = summary[summary["Model"] == model].sort_values("Horizon")
        x = group["Horizon"].to_numpy() + offsets[model]
        axes[0].plot(x, group["Mean_Rank_RMSE"], marker=markers[model], linestyle=linestyles[model],
                     linewidth=2.8, markersize=9, label=model)
        axes[1].plot(x, group["Significant_RMSE_Share"] * 100, marker=markers[model],
                     linestyle=linestyles[model], linewidth=2.8, markersize=9, label=model)

    for ax in axes:
        ax.set_xticks([1, 3, 7])
        ax.tick_params(labelsize=14)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlabel("Forecast horizon", fontsize=16)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Average RMSE rank", fontsize=16)
    axes[0].set_title("(a) Average RMSE rank of actor-enriched models", fontsize=18)
    axes[1].set_ylim(-3, 103)
    axes[1].set_ylabel("Significant RMSE improvements (%)", fontsize=16)
    axes[1].set_title("(b) Significant actor-enrichment gains", fontsize=18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=len(models), frameon=False, fontsize=13)
    fig.text(0.5, 0.01,
             r"Lower average rank is better. Significant gains require positive $\Delta$RMSE and Wilcoxon $p<0.05$. "
             "Horizontal offsets display overlapping trajectories.",
             ha="center", fontsize=12)
    plt.tight_layout(rect=[0.02, 0.07, 0.98, 0.90])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
