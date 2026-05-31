# Data

Raw BPIC event logs are not redistributed in this repository. Download the logs from their official DOI records and construct the daily actor-enriched series as described in the manuscript.

Place the final daily time-series CSV files in `data/processed/`:

```text
BPIC2017_final_online_daily_time_series.csv
BPIC2012_final_online_daily_time_series.csv
BPIC2011_final_online_daily_time_series.csv
```

At minimum, each processed CSV must contain a `date` column, the two outcome series, and the eight actor-behavior count/duration series listed in the main `README.md`.
