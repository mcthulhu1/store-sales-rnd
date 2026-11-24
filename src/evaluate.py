"""Анализ ошибок на последнем фолде (файл с прогнозами делает train.py).

Запуск: python src/evaluate.py --config configs/baseline.yaml
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from utils import load_config, rmsle

MODELS = ["lightgbm", "moving_average_28", "seasonal_naive_7"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    out = config["output_dir"]
    os.makedirs(f"{out}/plots", exist_ok=True)

    df = pd.read_csv(f"{out}/val_predictions_last_fold.csv", parse_dates=["date"])
    df["day_of_horizon"] = (df["date"] - df["date"].min()).dt.days + 1

    # ошибка по категориям
    by_family = df.groupby("family")[["sales"] + MODELS].apply(
        lambda g: pd.Series(
            {m: rmsle(g["sales"], g[m]) for m in MODELS} | {"mean_sales": g["sales"].mean()}
        )
    )
    by_family = by_family.sort_values("lightgbm", ascending=False).round(4)
    by_family.to_csv(f"{out}/errors_by_family.csv")

    # ошибка по рядам (магазин + категория)
    by_series = df.groupby(["store_nbr", "family"])[["sales", "lightgbm"]].apply(
        lambda g: pd.Series(
            {
                "rmsle_lgbm": rmsle(g["sales"], g["lightgbm"]),
                "mean_sales": g["sales"].mean(),
                "mean_pred": g["lightgbm"].mean(),
            }
        )
    )
    by_series = by_series.sort_values("rmsle_lgbm", ascending=False).round(3)
    by_series.to_csv(f"{out}/errors_by_series.csv")

    # ошибка и смещение по дням: сумма прогноза по всей сети против суммы факта
    by_day = df.groupby("date")[["sales", "lightgbm"]].apply(
        lambda g: pd.Series(
            {
                "rmsle_lgbm": rmsle(g["sales"], g["lightgbm"]),
                "total_fact": g["sales"].sum(),
                "total_pred": g["lightgbm"].sum(),
            }
        )
    )
    by_day["bias_pct"] = 100 * (by_day["total_pred"] / by_day["total_fact"] - 1)
    by_day["dayofweek"] = by_day.index.dayofweek
    by_day.round(3).to_csv(f"{out}/errors_by_day.csv")

    plt.figure(figsize=(8, 3.5))
    plt.plot(by_day.index, by_day["rmsle_lgbm"], marker="o")
    plt.ylabel("RMSLE")
    plt.title("Ошибка LightGBM по дням последнего фолда")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{out}/plots/rmsle_by_day.png", dpi=110)
    plt.close()

    # графики 3 худших рядов
    worst = by_series.head(3).index
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    for ax, (store, family) in zip(axes, worst, strict=True):
        s = df[(df["store_nbr"] == store) & (df["family"] == family)]
        ax.plot(s["date"], s["sales"], color="black", label="факт")
        ax.plot(s["date"], s["lightgbm"], label="lightgbm")
        ax.plot(s["date"], s["seasonal_naive_7"], "--", label="seasonal naive")
        ax.set_title(f"магазин {store}, {family}", fontsize=9)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out}/plots/worst_series.png", dpi=110)
    plt.close()

    print("Худшие категории:\n", by_family.head(5))
    print("Худшие ряды:\n", by_series.head(5))
    print("Ряды без продаж в фолде:", int((by_series["mean_sales"] == 0).sum()))
    print("По дням:\n", by_day.round(3))


if __name__ == "__main__":
    main()
