"""Обучение и валидация baseline на нескольких временных фолдах.

Запуск: python src/train.py --config configs/baseline.yaml
"""

import argparse
import hashlib
import json
import os
import subprocess

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml

from utils import (
    CAT_FEATURES,
    add_features,
    bootstrap_ci,
    bootstrap_diff_ci,
    feature_list,
    load_data,
    rmsle,
    squared_log_error_by_series,
)


def seasonal_naive(df, cutoff, val, days):
    # прогноз = продажи в тот же день недели, последнее такое значение перед cutoff
    last_week = df[(df["date"] >= cutoff - pd.Timedelta(days=days)) & (df["date"] < cutoff)]
    # если окно больше 7 дней, каждый день недели встретится несколько раз - берем самый свежий
    last_week = last_week.sort_values("date").drop_duplicates(
        ["store_nbr", "family", "dayofweek"], keep="last"
    )
    last_week = last_week[["store_nbr", "family", "dayofweek", "sales"]]
    last_week = last_week.rename(columns={"sales": "pred"})
    res = val.merge(last_week, on=["store_nbr", "family", "dayofweek"], how="left")
    return res["pred"].values


def moving_average(df, cutoff, val, days):
    # прогноз = средние продажи ряда за последние N дней
    last = df[(df["date"] >= cutoff - pd.Timedelta(days=days)) & (df["date"] < cutoff)]
    mean_sales = last.groupby(["store_nbr", "family"], observed=True)["sales"].mean()
    mean_sales = mean_sales.rename("pred").reset_index()
    res = val.merge(mean_sales, on=["store_nbr", "family"], how="left")
    return res["pred"].values


def train_lgbm(train, val, features, config):
    train_set = lgb.Dataset(
        train[features], np.log1p(train["sales"]), categorical_feature=CAT_FEATURES
    )
    params = dict(config["lgbm_params"], seed=config["seed"])
    model = lgb.train(params, train_set, num_boost_round=config["num_boost_round"])
    pred = np.expm1(model.predict(val[features]))
    return np.clip(pred, 0, None), model


def file_md5(path):
    # хеш файла с данными = версия данных
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def log_to_mlflow(config, config_path, results, out_dir):
    """Один run в MLflow на каждую модель: параметры, метрики по фолдам, артефакты."""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment"])
    data_version = file_md5(os.path.join(config["data_dir"], "train.csv"))
    commit = git_commit()
    for model_name, model_res in results.groupby("model"):
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tags({"model": model_name, "git_commit": commit})
            mlflow.log_params(
                {
                    "model": model_name,
                    "horizon": config["horizon"],
                    "n_folds": config["n_folds"],
                    "train_start": config["train_start"],
                    "seed": config["seed"],
                    "train_csv_md5": data_version,
                }
            )
            if model_name == "lightgbm":
                mlflow.log_params(config["lgbm_params"])
                mlflow.log_params(
                    {
                        "num_boost_round": config["num_boost_round"],
                        "lags": config["lags"],
                        "roll_windows": config["roll_windows"],
                    }
                )
            for _, row in model_res.iterrows():
                step = int(row["fold"])
                mlflow.log_metric("rmsle", row["rmsle"], step=step)
                mlflow.log_metric("mae", row["mae"], step=step)
                mlflow.log_metric("rmsle_ci_low", row["ci_low"], step=step)
                mlflow.log_metric("rmsle_ci_high", row["ci_high"], step=step)
            mlflow.log_metrics(
                {
                    "rmsle_mean": model_res["rmsle"].mean(),
                    "rmsle_std": model_res["rmsle"].std(),
                    "mae_mean": model_res["mae"].mean(),
                }
            )
            mlflow.log_artifact(config_path)
            mlflow.log_artifact(f"{out_dir}/results_by_fold.csv")
            if model_name == "lightgbm":
                mlflow.log_artifact(f"{out_dir}/feature_importance.csv")
                mlflow.log_artifact(f"{out_dir}/lgbm_vs_baselines.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    df = add_features(load_data(config["data_dir"]), config)
    features = feature_list(config)
    horizon = config["horizon"]
    last_date = df["date"].max()
    n_boot, seed = config["n_bootstrap"], config["seed"]

    results, comparisons, importances = [], [], []
    for fold in range(config["n_folds"]):
        # фолды идут подряд, последний заканчивается в последний день train
        k = config["n_folds"] - 1 - fold
        cutoff = last_date - pd.Timedelta(days=horizon * (k + 1) - 1)
        val_end = cutoff + pd.Timedelta(days=horizon - 1)

        train = df[
            (df["date"] >= config["train_start"]) & (df["date"] < cutoff) & df["use_for_train"]
        ]
        val = df[(df["date"] >= cutoff) & (df["date"] <= val_end)].reset_index(drop=True)

        preds = {
            "seasonal_naive_7": seasonal_naive(df, cutoff, val, config["seasonal_naive_days"]),
            "moving_average_28": moving_average(df, cutoff, val, config["moving_average_days"]),
        }
        preds["lightgbm"], model = train_lgbm(train, val, features, config)
        importances.append(pd.Series(model.feature_importance("gain"), index=features))

        errors = {name: squared_log_error_by_series(val, pred) for name, pred in preds.items()}
        period = f"{cutoff.date()}..{val_end.date()}"
        for name, pred in preds.items():
            ci_low, ci_high = bootstrap_ci(errors[name], n_boot, seed)
            results.append(
                {
                    "fold": fold,
                    "period": period,
                    "model": name,
                    "rmsle": rmsle(val["sales"], pred),
                    "mae": np.mean(np.abs(val["sales"] - pred)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
        # парное сравнение: на сколько LightGBM лучше каждого наивного прогноза
        for name in ["moving_average_28", "seasonal_naive_7"]:
            diff_low, diff_high = bootstrap_diff_ci(errors[name], errors["lightgbm"], n_boot, seed)
            comparisons.append(
                {
                    "fold": fold,
                    "baseline": name,
                    "rmsle_diff": rmsle(val["sales"], preds[name])
                    - rmsle(val["sales"], preds["lightgbm"]),
                    "diff_ci_low": diff_low,
                    "diff_ci_high": diff_high,
                }
            )
        print(
            f"fold {fold} {period}: "
            + ", ".join(f"{r['model']}={r['rmsle']:.4f}" for r in results[-3:])
        )

    # прогнозы последнего фолда - для анализа ошибок в evaluate.py
    val_out = val[["date", "store_nbr", "family", "sales"]].copy()
    for name, pred in preds.items():
        val_out[name] = pred
    val_out.to_csv(f"{out_dir}/val_predictions_last_fold.csv", index=False)

    # важность признаков, усредненная по всем фолдам
    importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    importance.round(1).to_csv(f"{out_dir}/feature_importance.csv", header=["gain_mean"])

    results = pd.DataFrame(results)
    results.round(4).to_csv(f"{out_dir}/results_by_fold.csv", index=False)
    comparisons = pd.DataFrame(comparisons)
    comparisons.round(4).to_csv(f"{out_dir}/lgbm_vs_baselines.csv", index=False)
    summary = (
        results.groupby("model")
        .agg(
            rmsle_mean=("rmsle", "mean"),
            rmsle_std=("rmsle", "std"),
            mae_mean=("mae", "mean"),
        )
        .sort_values("rmsle_mean")
    )
    summary.round(4).to_csv(f"{out_dir}/results_summary.csv")
    print(summary.round(4))
    print(comparisons.round(4).to_string(index=False))
    with open(f"{out_dir}/config_used.json", "w") as f:
        json.dump(config, f, indent=2)

    log_to_mlflow(config, args.config, results, out_dir)
    print("MLflow: mlflow ui --backend-store-uri " + config["mlflow"]["tracking_uri"])


if __name__ == "__main__":
    main()
