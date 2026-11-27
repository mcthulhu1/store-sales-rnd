"""ДЗ №3: общие функции. Существующий код ДЗ №1-2 не меняется, только импортируется."""

import os
from pathlib import Path

import mlflow
import pandas as pd

from train import git_commit
from utils import load_config

# колонки, которые сохраняем вместе с прогнозами для анализа ошибок и срезов
PRED_COLUMNS = [
    "fold",
    "date",
    "store_nbr",
    "family",
    "type",
    "cluster",
    "onpromotion",
    "holiday",
    "event",
    "dayofweek",
    "long_weekend",
    "dayoff_block_len",
    "local_holiday",
    "days_since_payday",
    "lag_16",
    "roll_mean_7",
    "roll_mean_28",
    "sales",
]


def read_config(path):
    config = load_config(path)
    # ключи seeds из yaml читаются как int, hw2_runs - тоже; приводим к int на всякий случай
    config["hw2_runs"] = {int(k): v for k, v in config.get("hw2_runs", {}).items()}
    return config


def fold_bounds(last_date, horizon, n_folds):
    """Те же фолды, что в train.py: подряд по 16 дней, последний кончается в конце train."""
    bounds = []
    for fold in range(n_folds):
        k = n_folds - 1 - fold
        cutoff = last_date - pd.Timedelta(days=horizon * (k + 1) - 1)
        bounds.append((fold, cutoff, cutoff + pd.Timedelta(days=horizon - 1)))
    return bounds


def train_start_for_fraction(config, cutoff, fraction):
    """Начало train при доле fraction: последние fraction * (cutoff - train_start) дней."""
    full_days = (cutoff - pd.Timestamp(config["train_start"])).days
    return cutoff - pd.Timedelta(days=round(fraction * full_days))


def setup_mlflow(config):
    """Отдельная база MLflow ДЗ №3; артефакты в outputs/hw3/mlartifacts."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.makedirs(config["output_dir"], exist_ok=True)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    name = config["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        artifact_dir = Path(config["mlflow"]["artifact_dir"]).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(name)


def base_tags(config, seed, check):
    return {
        "model": "lightgbm",
        "experiment_id": config["experiment_id"],
        "check": check,
        "git_commit": git_commit(),
        "hw2_run_id": config["hw2_runs"].get(seed, ""),
        "hw2_experiment": "store-sales-hw2 (mlflow.db)",
    }
