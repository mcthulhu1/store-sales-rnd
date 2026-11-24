import os

import numpy as np
import pandas as pd
import yaml

CAT_FEATURES = ["store_nbr", "family", "type", "cluster"]

# группы признаков ДЗ №2; включаются списком extra_features в конфиге
EXTRA_FEATURES = {
    # годовая сезонность: продажи год назад в тот же день недели (364 = 52 недели)
    "yoy": ["lag_364", "yoy_roll_7"],
    # длинные выходные: выходной день внутри блока из 3+ нерабочих дней подряд
    "long_weekend": ["long_weekend", "dayoff_block_len"],
    # праздники города или провинции магазина
    "local_holiday": ["local_holiday"],
    # дни после зарплаты: в Эквадоре госсектор получает зарплату 15-го и в последний день месяца
    "payday": ["days_since_payday"],
}


def feature_list(config):
    lags = [f"lag_{lag}" for lag in config["lags"]]
    rolls = [f"roll_mean_{w}" for w in config["roll_windows"]]
    extra = [col for group in config.get("extra_features", []) for col in EXTRA_FEATURES[group]]
    return (
        CAT_FEATURES
        + ["onpromotion", "holiday", "event", "dayofweek", "day", "month"]
        + lags
        + rolls
        + extra
    )


def load_config(path):
    """YAML-конфиг; ключ base - путь к родительскому конфигу, его значения перекрываются."""
    with open(path) as f:
        config = yaml.safe_load(f)
    base = config.pop("base", None)
    if base is None:
        return config
    merged = load_config(os.path.join(os.path.dirname(path), base))
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


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

    df = df.merge(
        stores[["store_nbr", "type", "cluster", "city", "state"]], on="store_nbr", how="left"
    )

    # национальные выходные дни (без перенесенных) и отдельно события вроде Black Friday;
    # Work Day - это рабочая суббота, ее не берем
    off_types = ["Holiday", "Transfer", "Additional", "Bridge"]
    actual = holidays[~holidays["transferred"]]
    national = actual[actual["locale"] == "National"]
    days_off = national[national["type"].isin(off_types)]
    events = national[national["type"] == "Event"]
    df["holiday"] = df["date"].isin(days_off["date"]).astype(int)
    df["event"] = df["date"].isin(events["date"]).astype(int)

    # ДЗ №2: праздник в городе (Local) или провинции (Regional) магазина
    local = actual[actual["type"].isin(off_types) & actual["locale"].isin(["Local", "Regional"])]
    local_keys = set(zip(local["date"], local["locale_name"], strict=True))
    df["local_holiday"] = [
        int((d, c) in local_keys or (d, s) in local_keys)
        for d, c, s in zip(df["date"], df["city"], df["state"], strict=True)
    ]
    df = df.drop(columns=["city", "state"])

    # ДЗ №2: длинные выходные - блок подряд идущих нерабочих дней (сб, вс, национальный выходной)
    calendar = pd.DataFrame({"date": all_dates})
    # Work Day - перенесенный рабочий день (обычно суббота): в блок нерабочих дней не входит
    work_days = national.loc[national["type"] == "Work Day", "date"]
    weekend = (calendar["date"].dt.dayofweek >= 5) & ~calendar["date"].isin(work_days)
    is_off = weekend | calendar["date"].isin(days_off["date"])
    block_id = (is_off != is_off.shift()).cumsum()
    block_len = is_off.groupby(block_id).transform("size")
    calendar["dayoff_block_len"] = np.where(is_off, block_len, 0)
    calendar["long_weekend"] = (calendar["dayoff_block_len"] >= 3).astype(int)

    # ДЗ №2: дней с последней зарплаты (15-е число и последний день месяца)
    day = calendar["date"].dt.day
    payday = calendar["date"].dt.is_month_end | (day == 15)
    # до 15-го считаем от конца прошлого месяца (это day дней), после 15-го - от 15-го
    calendar["days_since_payday"] = np.where(payday, 0, np.where(day < 15, day, day - 15))
    df = df.merge(calendar, on="date", how="left")

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

    # ДЗ №2, годовая сезонность: тот же день недели год назад и среднее за неделю вокруг него
    # (дни t-367..t-361); все сдвиги больше горизонта, утечки нет
    if "yoy" in config.get("extra_features", []):
        df["lag_364"] = log_sales.groupby(keys).shift(364)
        shifted_year = log_sales.groupby(keys).shift(361)
        df["yoy_roll_7"] = shifted_year.groupby(keys).transform(lambda s: s.rolling(7).mean())

    for col in CAT_FEATURES:
        df[col] = df[col].astype("category")
    return df
