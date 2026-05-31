# When Coordination Becomes Predictive  
## Actor-Enriched Multi-Horizon Forecasting of Process Performance

This repository contains the implementation scaffold and reproducibility materials for a study of actor-enriched forecasting of daily open-case process-performance indicators. The experiments compare baseline inputs derived from observable process-state history with actor-enriched inputs that additionally capture continuation, interruption, handover-idle, and handover-busy dynamics.

## Repository layout

```text
.
├── configs/                      # Dataset-specific and experiment settings
├── data/
│   ├── raw/                      # Raw event logs (not distributed here)
│   └── processed/                # Daily actor-enriched time-series CSV files
├── docs/                         # Implementation and reproducibility notes
├── figures/                      # Publication-ready figures
├── notebooks/
│   └── archive/                  # Output-stripped development notebooks
├── results/
│   └── interim/                  # Current result exports and ranking inputs
├── scripts/                      # Executable experiment entry points
├── src/actor_enriched_forecasting/
│   ├── models/                   # Classical, tree-based and recurrent models
│   ├── features.py               # Leakage-aware temporal feature construction
│   ├── evaluation.py             # Metrics and paired significance tests
│   ├── interpretability.py       # SHAP aggregation helpers
│   ├── io.py                     # Loading and chronological splits
│   └── plotting.py               # Ranking/significance figure generation
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Datasets

The study uses the following Business Process Intelligence Challenge (BPIC) event logs:

- BPIC2017: DOI `10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b`
- BPIC2012: DOI `10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f`
- BPIC2011: DOI `10.4121/uuid:d9769f3d-0ab0-4fb8-803b-0d1120ffcf54`

The repository does not redistribute raw event logs. Place the constructed daily time-series files in `data/processed/` using the filenames specified in `configs/datasets.yaml`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Expected processed data columns

Each daily input CSV should include:

```text
date
Avg_Elapsed_Time
Avg_Remaining_Time
Cum_Count_C
Cum_Count_I
Cum_Count_HI
Cum_Count_HB
Cum_Time_C_seconds
Cum_Time_I_seconds
Cum_Time_HI_seconds
Cum_Time_HB_seconds
```

Only `Avg_Elapsed_Time` is used as an observable process-state input. `Avg_Remaining_Time` is treated as a forecasting target and is not included as an input predictor.

## Running the pipeline

### 1. Seasonality diagnostics

```bash
python scripts/01_check_seasonality.py --dataset BPIC2017
python scripts/01_check_seasonality.py --dataset BPIC2012
python scripts/01_check_seasonality.py --dataset BPIC2011
```

The selected SARIMAX specifications are stored in `configs/datasets.yaml`:

- BPIC2017: weekly seasonal order `(1,0,0,7)`
- BPIC2012: non-seasonal order `(0,0,0,0)`
- BPIC2011: non-seasonal order `(0,0,0,0)`

### 2. Classical time-series models

```bash
python scripts/02_run_classical_models.py --dataset BPIC2017
python scripts/02_run_classical_models.py --dataset BPIC2012
python scripts/02_run_classical_models.py --dataset BPIC2011
```

### 3. Tree-based models

```bash
python scripts/03_run_tree_models.py --dataset BPIC2017
```

Repeat for each dataset after confirming final tuned parameters in `configs/experiment.yaml`.

### 4. Recurrent attention models

```bash
python scripts/04_run_rnn_models.py --dataset BPIC2017
```

### 5. Ranking and significance plot

```bash
python scripts/05_plot_rankings.py \
  --input results/interim/final_model_results_for_ranking_current.csv
```

### 6. LightGBM SHAP interpretation

```bash
python scripts/06_explain_lightgbm_shap.py --model path/to/model.joblib --features path/to/features.csv
```

## Important implementation note

The modular code in `src/` follows the online-compatible formulation used in the manuscript:

- `Avg_Remaining_Time` is never used as an input feature.
- Future elapsed time may be modeled as a change from observable current elapsed time.
- Future remaining time is modeled directly as a future level, because its current realized value depends on future case completion.
- Non-causal peak detection is not used in the modular feature builder.

The notebooks in `notebooks/archive/` are output-stripped development records and may include earlier experimental cells. Use the modular scripts for final reproducible experiments.

## Citation

A `CITATION.cff` file is provided. Replace the placeholder manuscript DOI after publication.
