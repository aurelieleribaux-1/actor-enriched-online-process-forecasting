# Notebooks

This folder contains the notebook-based framework used in the study. It covers two stages: generating the daily time series and running the forecasting experiments.

## Time-series generation pipeline

The time-series generation notebooks construct the daily process-performance and actor-coordination series for BPIC2011, BPIC2012, and BPIC2017.

Run the notebook for the relevant dataset using its original event log and corresponding actor-enriched event data. The resulting daily CSV can then be used in the forecasting experiments. The three daily CSVs used in the study are also supplied in the time-series subfolder under `data/`.

See `data/README.md` for the original event-log sources and the actor-enrichment preparation step.

## Forecasting experiments

The experiment notebooks implement the forecasting framework separately for each dataset:

- `Experiments_Actor_Enriched_Forecasting_Framework_BPIC2011.ipynb`
- `Experiments_Actor_Enriched_Forecasting_Framework_BPIC2012.ipynb`
- `Experiments_Actor_Enriched_Forecasting_Framework_BPIC2017.ipynb`

They compare performance-only, workload-aware, and actor-enriched inputs for forecasting future average elapsed time and average remaining time at horizons of 1, 3, and 7 days.

The model groups include classical time-series models, tree-based models, recurrent neural networks with and without attention, and frozen time-series foundation models with trained Ridge prediction layers.

## How to run

1. Open the relevant notebook in Google Colab and follow its setup instructions.
2. To rebuild the daily data, run the time-series generation pipeline using the matching raw and actor-enriched event data.
3. Open the dataset's experiment notebook and upload either the generated daily CSV or the supplied study CSV.
4. Run the common data and feature setup before executing the selected model group.
5. Download the experiment result bundles and retain them in `results/`.

Model groups can be run separately after the common experiment setup. Consult the main repository README and the instructions within each notebook for evaluation settings, input definitions, and required supporting files.

