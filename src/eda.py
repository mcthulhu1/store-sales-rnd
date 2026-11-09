"""Короткий EDA: основные цифры в outputs/eda.json и пара графиков.

Запуск: python src/eda.py --config configs/baseline.yaml
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_dir, out = config["data_dir"], config["output_dir"]
    os.makedirs(f"{out}/plots", exist_ok=True)

    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["date"])
    test = pd.read_csv(f"{data_dir}/test.csv", parse_dates=["date"])
    stores = pd.read_csv(f"{data_dir}/stores.csv")
    oil = pd.read_csv(f"{data_dir}/oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(f"{data_dir}/holidays_events.csv", parse_dates=["date"])

    all_days = pd.date_range(train["date"].min(), train["date"].max())
    missing_days = sorted(set(all_days) - set(train["date"].unique()))
    last_56 = train[train["date"] > train["date"].max() - pd.Timedelta(days=56)]
    zero_series = (last_56.groupby(["store_nbr", "family"])["sales"].sum() == 0).sum()

    # магазины, которые открылись позже начала данных
    first_sale = train[train["sales"] > 0].groupby("store_nbr")["date"].min()
    late_stores = first_sale[first_sale > train["date"].min() + pd.Timedelta(days=30)]

    sales_by_weekday = train.groupby(train["date"].dt.dayofweek)["sales"].mean()
    sales_by_promo = train.groupby(train["onpromotion"] > 0)["sales"].mean()
    top_families = train.groupby("family")["sales"].sum().sort_values(ascending=False).head(5)
    fractional = (train["sales"] % 1 != 0).mean()

    eda = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_period": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
        "n_stores": int(train["store_nbr"].nunique()),
        "n_families": int(train["family"].nunique()),
        "missing_days_in_train": [str(d.date()) for d in missing_days],
        "nan_in_train": int(train.isna().sum().sum()),
        "oil_nan_share": round(float(oil["dcoilwtico"].isna().mean()), 3),
        "zero_sales_share": round(float((train["sales"] == 0).mean()), 3),
        "fractional_sales_share": round(float(fractional), 3),
        "series_all_zero_last_56d": int(zero_series),
        "stores_opened_late": {int(s): str(d.date()) for s, d in late_stores.items()},
        "sales_quantiles": {
            str(q): round(float(train["sales"].quantile(q)), 2) for q in [0.5, 0.9, 0.99, 1.0]
        },
        "mean_sales_by_weekday_mon0": sales_by_weekday.round(1).tolist(),
        "share_rows_on_promo": round(float((train["onpromotion"] > 0).mean()), 3),
        "mean_sales_no_promo": round(float(sales_by_promo[False]), 1),
        "mean_sales_promo": round(float(sales_by_promo[True]), 1),
        "top_families_by_sales": top_families.round(0).to_dict(),
        "holiday_types": holidays["type"].value_counts().to_dict(),
        "store_types": stores["type"].value_counts().to_dict(),
    }
    with open(f"{out}/eda.json", "w") as f:
        json.dump(eda, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(10, 3.5))
    train.groupby("date")["sales"].sum().plot(lw=0.6)
    plt.title("Продажи всей сети по дням")
    plt.tight_layout()
    plt.savefig(f"{out}/plots/eda_total_sales.png", dpi=110)
    plt.close()

    plt.figure(figsize=(6, 3))
    sales_by_weekday.plot(kind="bar")
    plt.title("Средние продажи по дням недели (0 = пн)")
    plt.tight_layout()
    plt.savefig(f"{out}/plots/eda_by_weekday.png", dpi=110)
    plt.close()

    print(json.dumps(eda, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
