# When Coordination Becomes Predictive
## Actor-Enriched Multi-Horizon Forecasting of Process Performance

This repository contains the notebooks, study data, and result exports for actor-enriched forecasting of daily process-performance indicators. The framework transforms event logs into daily time series and evaluates whether actor-coordination features improve forecasts beyond observable performance and workload history.

The study uses BPIC2011, BPIC2012, and BPIC2017. Actor features describe continuation, interruption, handover-idle, and handover-busy dynamics.

## Repository layout

| Folder | Contents |
| --- | --- |
| `data/` | Data files required for the study, including the daily time series used by the experiment notebooks. |
| `notebooks/` | The framework notebooks: time-series generation, experiment execution, and supporting analyses and validation. |
| `results/` | Saved experiment outputs and analysis results. |

The reproducibility workflow is notebook-based. Start with time-series generation when rebuilding the data, then run the corresponding dataset's experiment notebook.

## Datasets

The original Business Process Intelligence Challenge event logs are available from:

- **BPIC2011:** https://doi.org/10.4121/uuid:d9769f3d-0ab0-4fb8-803b-0d1120ffcf54
- **BPIC2012:** https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f
- **BPIC2017:** https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b

Use the matching dataset throughout time-series generation, validation, and forecasting. The daily inputs used for the experiments belong in `data/`. Obtain the original event log from the source above when it is needed to regenerate or validate those inputs.

## Running the framework

### 1. Generate the daily time series

Open the time-series generation notebooks in `notebooks/` and follow their instructions to load the original event logs, construct actor-coordination features, and aggregate the process and actor information into daily series.

Use the dataset-specific date conventions and elapsed-time units implemented in those notebooks. Retain the generated CSV files for the experiment and validation steps.

### 2. Run the forecasting experiments

Open the experiment notebook for the relevant dataset:

- `Experiments_Actor_Enriched_Forecasting_Framework_BPIC2011.ipynb`
- `Experiments_Actor_Enriched_Forecasting_Framework_BPIC2012.ipynb`
- `Experiments_Actor_Enriched_Forecasting_Framework_BPIC2017.ipynb`

The notebooks are designed for **Google Colab**. Run their setup and installation cells, upload the matching daily time-series CSV when prompted, and run Section 1 before selecting a model group.

The dataset-characteristics section additionally requests the original event log and the enriched actor-behavior CSV used to construct the daily series.

The experiment notebooks include:

- **Classical models:** SARIMAX and ARX-HAC.
- **Tree-based models:** XGBoost and LightGBM.
- **Recurrent models:** LSTM and GRU, with and without attention.
- **Time-series foundation models:** TimesFM-2.5 and Chronos-2.

After the common data and feature setup, model groups can be run separately. Follow the notebook instructions to download their result bundles and retain the outputs in `results/`.

### 3. Inspect and compare results

Use the saved predictions, metrics, and supporting analysis outputs to compare model families, forecasting horizons, and input variants. Result bundles also retain model and preprocessing artifacts where implemented by the corresponding notebook.

## Forecasting task

The forecasting targets are the direct future levels of:

- `Avg_Elapsed_Time`
- `Avg_Remaining_Time`

Each target is evaluated at horizons of **1, 3, and 7 days**.

The experiments compare three input variants:

| Variant | Inputs |
| --- | --- |
| **PerformanceOnly** | Observable elapsed-time history and its engineered features. |
| **WorkloadAware** | Performance inputs plus observable process-state and workload information. |
| **WorkloadPlusActor** | Workload-aware inputs plus actor-coordination features. |

Process-state inputs include cases with activity, events during the day, completed transitions, and active resources. Actor inputs include cumulative behavior counts and durations, behavior proportions, and model-specific engineered features.

`Avg_Remaining_Time` is a forecasting target and is **never used as an input predictor**.

## Daily input data

The experiment notebooks use daily CSV files with a date index and the following core columns:

```text
date
Avg_Elapsed_Time
Avg_Remaining_Time
Cases_With_Activity
Events_During_Day
Transitions_Completed_During_Day
Active_Resources_During_Day
Cum_Count_C
Cum_Count_I
Cum_Count_HI
Cum_Count_HB
Cum_Time_C_seconds
Cum_Time_I_seconds
Cum_Time_HI_seconds
Cum_Time_HB_seconds
```

Behavior labels follow the dataset-specific generation notebook. Some earlier materials use `HD` instead of `HB` for the fourth behavior class. Keep count and duration column names consistent with the matching experiment notebook.

Elapsed-time and remaining-time units follow the dataset-specific input files; behavior-duration columns ending in `_seconds` are measured in seconds.

## Evaluation and temporal validity

The notebooks use three expanding-window evaluation blocks: 70% training followed by 10% testing, 80% followed by 10%, and 90% followed by the final 10%. Training and test origins are restricted so that their target dates remain within the corresponding fold boundaries.

Reported forecasting metrics include RMSE, MAE, R², and MASE.

Predictors are constructed from information available at the forecast cutoff. Training-dependent preprocessing and fitted prediction layers use the corresponding training partition.

Supporting cutoff diagnostics independently reconstruct actor labels and cumulative actor features after hiding events beyond each daily cutoff, then compare them with the full-log reference through that date. Separate process reconstruction checks compare daily process predictors with the saved CSV. These checks use the dataset's date conventions and measurement units.

## Foundation-model configuration

TimesFM-2.5 and Chronos-2 use frozen pretrained backbones. Their forecasts of permitted input series feed a supervised Ridge prediction layer that predicts the future KPI level.

The shared settings are:

- Maximum context: **512 daily observations**.
- Minimum available history: **32 observations**.
- Forecast horizons: **1, 3, and 7 days**.
- Inference batch size: **64**.
- Ridge regularization: **α = 1**.

A separate prediction layer is fitted for each foundation model, fold, target, horizon, and input variant. Median imputation and standardization are fitted on training data. The full model settings are recorded in the experiment notebooks.

## Citation

Please cite the accompanying manuscript when using this framework and cite the original event-log sources when using the datasets.

