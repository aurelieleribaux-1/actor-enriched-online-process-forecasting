# Data

The folder shows the three forms of data needed. First, the raw BPIC event logs are not redistributed in this repository. PLease download the original logs from their official DOI records. 

Second, the folder 'actor enriched event logs' contains the code to construct the daily actor-enriched series as described in the manuscript.

Finally, 

Place the final daily time-series CSV files in `data/processed/`:

```text
BPIC2017_final_online_daily_time_series.csv
BPIC2012_final_online_daily_time_series.csv
BPIC2011_final_online_daily_time_series.csv
```

At minimum, each processed CSV must contain a `date` column, the two outcome series, and the eight actor-behavior count/duration series listed in the main `README.md`.
