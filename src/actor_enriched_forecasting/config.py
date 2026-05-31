from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset_config(dataset: str, config_path: str | Path = "configs/datasets.yaml") -> dict[str, Any]:
    config = load_yaml(config_path)
    datasets = config.get("datasets", {})
    if dataset not in datasets:
        raise KeyError(f"Unknown dataset '{dataset}'. Available datasets: {sorted(datasets)}")
    return datasets[dataset]


def load_experiment_config(config_path: str | Path = "configs/experiment.yaml") -> dict[str, Any]:
    return load_yaml(config_path)
