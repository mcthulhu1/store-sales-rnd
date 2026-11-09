import numpy as np
import pandas as pd

CAT_FEATURES = ["store_nbr", "family", "type", "cluster"]


def feature_list(config):
    lags = [f"lag_{lag}" for lag in config["lags"]]
    rolls = [f"roll_mean_{w}" for w in config["roll_windows"]]
    return (
        CAT_FEATURES
        + ["onpromotion", "holiday", "event", "dayofweek", "day", "month"]
        + lags
        + rolls
    )


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def squared_log_error_by_series(val, pred):
    """Средняя квадратичная лог-ошибка для каждого ряда (магазин + категория)."""
    err = (np.log1p(np.clip(pred, 0, None)) - np.log1p(val["sales"].values)) ** 2
    return (
        pd.Series(err)
        .groupby([val["store_nbr"].values, val["family"].values], observed=True)
        .mean()
        .values
    )


def bootstrap_ci(errors, n_bootstrap, seed):
    """95% интервал RMSLE: много раз выбираем ряды с повторами и пересчитываем метрику."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(errors), size=len(errors))
        scores.append(np.sqrt(errors[idx].mean()))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def bootstrap_diff_ci(errors_a, errors_b, n_bootstrap, seed):
    """95% интервал разницы RMSLE(a) - RMSLE(b) на одних и тех же выборках рядов."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(errors_a), size=len(errors_a))
        diffs.append(np.sqrt(errors_a[idx].mean()) - np.sqrt(errors_b[idx].mean()))
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def load_data(data_dir):
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["date"])
    stores = pd.read_csv(f"{data_dir}/stores.csv")
    holidays = pd.read_csv(f"{data_dir}/holidays_events.csv", parse_dates=["date"])

    # 25 декабря магазины закрыты, этих дней нет в train - добавим их с нулями
    all_dates = pd.date_range(train["date"].min(), train["date"].max())
    full_index = pd.MultiIndex.from_product(
        [all_dates, train["store_nbr"].unique(), train["family"].unique()],
        names=["date", "store_nbr", "family"],
    )
    df = train.set_index(["date", "store_nbr", "family"]).reindex(full_index).reset_index()
    df["closed_day"] = df["sales"].isna()
    df["sales"] = df["sales"].fillna(0)
    df["onpromotion"] = df["onpromotion"].fillna(0)

    # магазин еще не открылся: нули до первой продажи - это не спрос
    first_sale = df[df["sales"] > 0].groupby("store_nbr")["date"].min().rename("first_sale")
    df = df.merge(first_sale, on="store_nbr", how="left")
    # такие строки нужны для лагов, но в обучение как таргет не идут
    df["use_for_train"] = ~df["closed_day"] & (df["date"] >= df["first_sale"])
    df = df.drop(columns=["closed_day", "first_sale"])

    df = df.merge(stores[["store_nbr", "type", "cluster"]], on="store_nbr", how="left")

    # национальные выходные дни (без перенесенных) и отдельно события вроде Black Friday;
    # Work Day - это рабочая суббота, ее не берем
    national = holidays[(holidays["locale"] == "National") & (~holidays["transferred"])]
    days_off = national[national["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"])]
    events = national[national["type"] == "Event"]
    df["holiday"] = df["date"].isin(days_off["date"]).astype(int)
    df["event"] = df["date"].isin(events["date"]).astype(int)

    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    return df


def add_features(df, config):
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    # все лаги >= горизонта (16): на последний день прогноза они еще известны
    min_lag = config["horizon"]
    assert min(config["lags"]) >= min_lag, "лаг меньше горизонта = утечка"

    log_sales = np.log1p(df["sales"])
    keys = [df["store_nbr"], df["family"]]
    for lag in config["lags"]:
        df[f"lag_{lag}"] = log_sales.groupby(keys).shift(lag)
    shifted = log_sales.groupby(keys).shift(min_lag)
    for window in config["roll_windows"]:
        df[f"roll_mean_{window}"] = shifted.groupby(keys).transform(
            lambda s, w=window: s.rolling(w).mean()
        )

    for col in CAT_FEATURES:
        df[col] = df[col].astype("category")
    return df
