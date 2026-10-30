# Data

This folder contains the data and preparation code used in the study, organized into three subfolders.

## Raw event logs

The raw BPIC event logs are not redistributed in this repository. Download them from their official DOI records and place them in the raw-data subfolder:

- **BPIC2011:** https://doi.org/10.4121/uuid:d9769f3d-0ab0-4fb8-803b-0d1120ffcf54
- **BPIC2012:** https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f
- **BPIC2017:** https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b

## Actor-enriched event logs

This subfolder contains the code for constructing actor-enriched versions of the original event logs. The enrichment identifies continuation, interruption, handover-idle, and handover-busy behavior, as described in the manuscript.

Use the matching dataset's enrichment code with its original event log to generate the actor-enriched event data required by the time-series generation pipeline.

## Daily time series

This subfolder contains the three daily time-series CSV files used in the study: one each for BPIC2011, BPIC2012, and BPIC2017. These files are the inputs to the forecasting experiment notebooks.

The daily CSVs can also be regenerated using the time-series generation pipeline under `notebooks/`, together with the corresponding raw and actor-enriched event logs.

Each daily CSV contains a date index, the two KPI series (`Avg_Elapsed_Time` and `Avg_Remaining_Time`), observable process-state indicators, and cumulative actor-behavior counts and durations. See the main repository README and the dataset-specific notebooks for column definitions and measurement units.

## Workflow

1. Download the original event log.
2. Run the enrichment code for that dataset.
3. Run the time-series generation notebook to reconstruct its daily CSV, or use the supplied daily CSV.
4. Load the daily CSV in the corresponding forecasting experiment notebook.

