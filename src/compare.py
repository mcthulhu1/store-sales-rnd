"""ДЗ №2: сводная таблица экспериментов, сравнение на одинаковых seed, абляции, графики.

Каждый эксперимент запускается с 3 seed (configs/hw2/<эксперимент>__seed<N>.yaml).
Сравнение с baseline и с полной моделью делается на одном и том же seed, затем усредняется.

Читает outputs/hw2/<эксперимент>/seed<N>/ (результат train.py и evaluate.py) и пишет:
- outputs/hw2/runs.csv - все запуски (эксперимент x seed);
- outputs/hw2/experiments.csv и .md - сводка по экспериментам (среднее по seed);
- outputs/hw2/ablations.csv - абляции против полной модели;
- outputs/hw2/deltas_by_fold.csv - выигрыш по фолдам и 95% парный бутстрап (seed 42);
- outputs/hw2/seed_noise_by_fold.csv - тот же бутстрап для baseline с другим seed (шум);
- outputs/hw2/plots/*.png

Запуск: python src/compare.py
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from utils import bootstrap_diff_ci, load_config, load_data, rmsle

BASELINE = "E0_baseline"
FULL = "E3_full"
MAIN_SEED = 42
SCHOOL = "SCHOOL AND OFFICE SUPPLIES"
# дни из анализа ошибок ДЗ №1: длинные выходные после переноса праздника и дни после зарплаты
LONG_WEEKEND_DAYS = pd.to_datetime(["2017-08-12", "2017-08-13"])
PAYDAY_DAYS = pd.to_datetime(["2017-07-31", "2017-08-01"])
# абляции: какой компонент убран из полной модели
ABLATIONS = {
    "E2_calendar": "yoy (годовая сезонность)",
    "E1_yoy": "весь календарь",
    "A1_full_minus_long_weekend": "long_weekend",
    "A2_full_minus_local_holiday": "local_holiday",
    "A3_full_minus_payday": "payday",
}


def to_markdown(frame):
    """Markdown-таблица без зависимости от tabulate."""
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def network_bias(pred):
    """Смещение суммарного прогноза по сети в процентах: (сумма прогноза / сумма факта - 1)."""
    return 100 * (pred["lightgbm"].sum() / pred["sales"].sum() - 1)


def load_run(path):
    config = load_config(path)
    out = config["output_dir"]
    folds = pd.read_csv(f"{out}/results_by_fold.csv")
    folds = folds[folds["model"] == "lightgbm"].set_index("fold")
    series = pd.read_csv(f"{out}/series_errors.csv").sort_values(["fold", "store_nbr", "family"])
    pred = pd.read_csv(f"{out}/val_predictions_last_fold.csv", parse_dates=["date"])
    school = pred[pred["family"] == SCHOOL]
    row = {
        "experiment": config["experiment_id"],
        "extra_features": ", ".join(config.get("extra_features", [])) or "-",
        "hypothesis": config.get("hypothesis", ""),
        "seed": config["seed"],
        "rmsle_mean": folds["rmsle"].mean(),
        "rmsle_std": folds["rmsle"].std(),
        "mae_mean": folds["mae"].mean(),
        **{f"rmsle_f{fold}": value for fold, value in folds["rmsle"].items()},
        # срезы последнего фолда, где ошибался baseline ДЗ №1
        "school_rmsle_f3": rmsle(school["sales"], school["lightgbm"]),
        "bias_12_13_aug_pct": network_bias(pred[pred["date"].isin(LONG_WEEKEND_DAYS)]),
        "bias_31jul_1aug_pct": network_bias(pred[pred["date"].isin(PAYDAY_DAYS)]),
    }
    return row, series, pred


def summarize(runs, reference):
    """Сводка по экспериментам: среднее по seed и разница с reference на тех же seed.

    delta = RMSLE(reference) - RMSLE(эксперимент) на одном seed, > 0 - эксперимент лучше.
    """
    ref = runs[runs["experiment"] == reference].set_index("seed")["rmsle_mean"]
    runs = runs.assign(delta=ref.reindex(runs["seed"]).values - runs["rmsle_mean"].values)
    summary = runs.groupby("experiment", sort=False).agg(
        extra_features=("extra_features", "first"),
        hypothesis=("hypothesis", "first"),
        n_seeds=("seed", "count"),
        rmsle_mean=("rmsle_mean", "mean"),
        rmsle_seed_range=("rmsle_mean", lambda s: s.max() - s.min()),
        delta_mean=("delta", "mean"),
        delta_min=("delta", "min"),
        delta_max=("delta", "max"),
        seeds_better=("delta", lambda s: int((s > 0).sum())),
        seeds_worse=("delta", lambda s: int((s < 0).sum())),
        mae_mean=("mae_mean", "mean"),
        school_rmsle_f3=("school_rmsle_f3", "mean"),
        bias_12_13_aug_pct=("bias_12_13_aug_pct", "mean"),
        bias_31jul_1aug_pct=("bias_31jul_1aug_pct", "mean"),
    )
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="configs/hw2")
    parser.add_argument("--output", default="outputs/hw2")
    args = parser.parse_args()
    os.makedirs(f"{args.output}/plots", exist_ok=True)

    names = sorted(n for n in os.listdir(args.configs) if n.endswith(".yaml"))
    rows, series, preds = [], {}, {}
    for name in names:
        row, ser, pred = load_run(os.path.join(args.configs, name))
        rows.append(row)
        series[(row["experiment"], row["seed"])] = ser
        preds[(row["experiment"], row["seed"])] = pred
    runs = pd.DataFrame(rows)
    # порядок: эксперименты E, затем абляции A
    order = sorted(runs["experiment"].unique(), key=lambda e: ("EA".find(e[0]), e))
    runs["order"] = runs["experiment"].map(order.index)
    runs = runs.sort_values(["order", "seed"]).drop(columns="order").reset_index(drop=True)
    runs.round(6).to_csv(f"{args.output}/runs.csv", index=False)

    # сводка против baseline на одинаковых seed; все расчеты на полной точности, округление только при выводе
    summary = summarize(runs, BASELINE)
    # критерий улучшения (эвристика, раздел 3 HW2.md): delta > 0 на всех seed
    # и средняя delta больше разброса baseline по seed (max - min среднего RMSLE)
    threshold = summary.loc[summary["experiment"] == BASELINE, "rmsle_seed_range"].iloc[0]
    summary["threshold"] = threshold
    summary["passes_criterion"] = (
        (summary["experiment"] != BASELINE)
        & (summary["seeds_better"] == summary["n_seeds"])
        & (summary["delta_mean"] > threshold)
    )
    print(f"порог улучшения (разброс baseline по seed): {threshold:.6f}")
    summary.round(6).to_csv(f"{args.output}/experiments.csv", index=False)
    cols = [
        "experiment", "extra_features", "rmsle_mean", "rmsle_seed_range", "delta_mean",
        "delta_min", "delta_max", "seeds_better", "passes_criterion", "mae_mean", "school_rmsle_f3",
        "bias_12_13_aug_pct", "bias_31jul_1aug_pct",
    ]  # fmt: skip
    with open(f"{args.output}/experiments.md", "w") as f:
        f.write("Среднее по 3 seed; delta = RMSLE(baseline) - RMSLE(эксп.) на том же seed\n\n")
        f.write(to_markdown(summary[cols].round(4)))
        f.write("\n")
    print(summary[cols].round(4).to_string(index=False))

    # абляции против полной модели на одинаковых seed:
    # increase = RMSLE(без компонента) - RMSLE(полная), > 0 - без компонента хуже, компонент полезен
    full_summary = summarize(runs, FULL).set_index("experiment")
    ablations = full_summary.loc[list(ABLATIONS)].reset_index()
    ablations.insert(1, "removed", ablations["experiment"].map(ABLATIONS))
    ablations["increase_mean"] = -ablations["delta_mean"]
    ablations["increase_min"] = -ablations["delta_max"]
    ablations["increase_max"] = -ablations["delta_min"]
    for col in ["mae_mean", "school_rmsle_f3", "bias_12_13_aug_pct", "bias_31jul_1aug_pct"]:
        ablations[f"{col}_change"] = ablations[col] - full_summary.loc[FULL, col]
    ab_cols = [
        "experiment", "removed", "rmsle_mean", "increase_mean", "increase_min", "increase_max",
        "seeds_worse", "mae_mean_change", "school_rmsle_f3_change",
        "bias_12_13_aug_pct_change", "bias_31jul_1aug_pct_change",
    ]  # fmt: skip
    ablations[ab_cols].round(6).to_csv(f"{args.output}/ablations.csv", index=False)
    print(ablations[ab_cols].round(4).to_string(index=False))

    # по фолдам на seed 42: выигрыш против baseline и 95% парный бутстрап по рядам
    base_config = load_config(os.path.join(args.configs, f"{BASELINE}__seed{MAIN_SEED}.yaml"))
    base_series = series[(BASELINE, MAIN_SEED)]
    n_folds, n_boot = base_config["n_folds"], base_config["n_bootstrap"]
    deltas = []
    for exp_id in order:
        exp_series = series[(exp_id, MAIN_SEED)]
        for row in fold_diffs(base_series, exp_series, n_folds, n_boot, base_config["seed"]):
            deltas.append({"experiment": exp_id, **row})
    deltas = pd.DataFrame(deltas)
    deltas.round(6).to_csv(f"{args.output}/deltas_by_fold.csv", index=False)

    # проверка критерия: тот же baseline, другой seed. Признаки не меняются, меняется
    # только случайный выбор признаков в деревьях (feature_fraction 0.8)
    noise = []
    for seed in sorted(runs.loc[runs["experiment"] == BASELINE, "seed"]):
        if seed == MAIN_SEED:
            continue
        other = series[(BASELINE, seed)]
        for row in fold_diffs(base_series, other, n_folds, n_boot, base_config["seed"]):
            noise.append({"comparison": f"seed {MAIN_SEED} vs seed {seed}", **row})
    noise = pd.DataFrame(noise)
    noise.round(6).to_csv(f"{args.output}/seed_noise_by_fold.csv", index=False)
    print(noise.round(4).to_string(index=False))

    plot_rmsle(runs, order, f"{args.output}/plots/rmsle_by_experiment.png")
    plot_fold_deltas(deltas, order, f"{args.output}/plots/delta_by_fold.png")
    plot_bias(runs, order, f"{args.output}/plots/bias_long_weekend.png")
    plot_school_store(preds, f"{args.output}/plots/school_store47.png")
    plot_school_by_year(base_config["data_dir"], f"{args.output}/plots/school_by_year.png")


def fold_diffs(series_a, series_b, n_folds, n_bootstrap, seed):
    """По каждому фолду: RMSLE(a) - RMSLE(b) и 95% парный бутстрап по рядам."""
    rows = []
    for fold in range(n_folds):
        a = series_a[series_a["fold"] == fold]
        b = series_b[series_b["fold"] == fold]
        assert (a[["store_nbr", "family"]].values == b[["store_nbr", "family"]].values).all()
        ea, eb = a["sq_log_err"].values, b["sq_log_err"].values
        low, high = bootstrap_diff_ci(ea, eb, n_bootstrap, seed)
        diff = ea.mean() ** 0.5 - eb.mean() ** 0.5
        rows.append({"fold": fold, "diff": diff, "ci_low": low, "ci_high": high})
    return rows


def plot_rmsle(runs, order, path):
    """Средний RMSLE по 4 фолдам: точки - seed, черта - среднее по seed."""
    fig, ax = plt.subplots(figsize=(9, 4))
    base = runs[runs["experiment"] == BASELINE]["rmsle_mean"]
    ax.axhspan(
        base.min(), base.max(), color="orange", alpha=0.25, label="baseline: min-max по seed"
    )
    for i, exp_id in enumerate(order):
        values = runs[runs["experiment"] == exp_id]["rmsle_mean"]
        color = "tab:grey" if exp_id == BASELINE else "tab:blue"
        ax.scatter([i] * len(values), values, color=color, s=18, alpha=0.6)
        ax.hlines(values.mean(), i - 0.25, i + 0.25, color=color, lw=2)
        ax.annotate(
            f"{values.mean():.4f}", (i, values.max()), xytext=(0, 6),
            textcoords="offset points", ha="center", fontsize=8,
        )  # fmt: skip
    ax.set_xticks(range(len(order)), order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSLE, среднее по 4 фолдам")
    ax.set_title("RMSLE по экспериментам: точки - 3 seed, черта - среднее (меньше - лучше)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_fold_deltas(deltas, order, path):
    """Выигрыш против baseline по фолдам 0-3 и 95% парный бутстрап, seed 42."""
    others = [e for e in order if e != BASELINE]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, exp_id in enumerate(others):
        d = deltas[deltas["experiment"] == exp_id]
        x = i + (d["fold"] - 1.5) * 0.15
        yerr = [d["diff"] - d["ci_low"], d["ci_high"] - d["diff"]]
        ax.errorbar(x, d["diff"], yerr=yerr, fmt="o", ms=4, capsize=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(others)), others, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSLE(baseline) - RMSLE(эксп.)")
    ax.set_title(
        "Выигрыш против baseline по фолдам 0-3, 95% парный бутстрап, seed 42 (> 0 - лучше)"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_bias(runs, order, path):
    """Смещение суммы прогноза по сети в длинные выходные 12-13.08, среднее и min-max по seed."""
    stats = runs.groupby("experiment")["bias_12_13_aug_pct"].agg(["mean", "min", "max"]).loc[order]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = ["tab:grey" if e == BASELINE else "tab:blue" for e in order]
    yerr = [stats["mean"] - stats["min"], stats["max"] - stats["mean"]]
    ax.bar(range(len(order)), stats["mean"], yerr=yerr, color=colors, alpha=0.8, capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(order)), order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("прогноз / факт - 1, %")
    ax.set_title("Перепрогноз суммы продаж по сети 12-13.08.2017 (среднее и min-max по 3 seed)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_school_store(preds, path, store=47):
    """Худший ряд ДЗ №1 (канцтовары, магазин 47): факт и прогнозы, seed 42."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for i, exp_id in enumerate([BASELINE, "E1_yoy", FULL]):
        pred = preds[(exp_id, MAIN_SEED)]
        sel = pred[(pred["family"] == SCHOOL) & (pred["store_nbr"] == store)]
        if i == 0:
            ax.plot(sel["date"], sel["sales"], color="black", label="факт")
        ax.plot(sel["date"], sel["lightgbm"], label=exp_id)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.set_title(f"Магазин {store}, {SCHOOL}, фолд 3, seed {MAIN_SEED}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_school_by_year(data_dir, path):
    """Канцтовары по всей сети по годам: когда начинается школьный сезон."""
    df = load_data(data_dir)
    school = df[df["family"] == SCHOOL].groupby("date")["sales"].sum()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for year in [2015, 2016, 2017]:
        part = school[f"{year}-07-01" : f"{year}-10-15"]
        # даты всех лет переносим на 2017 год: dayofyear в високосном 2016 сдвинут на день
        same_day = pd.to_datetime({"year": 2017, "month": part.index.month, "day": part.index.day})
        ax.plot(same_day.dt.dayofyear, part.values / 1000, label=str(year))
    start, end = pd.Timestamp("2017-07-31").dayofyear, pd.Timestamp("2017-08-15").dayofyear
    ax.axvspan(start, end, color="orange", alpha=0.2, label="фолд 3")
    # данные 2017 года заканчиваются 15.08: пик сезона 2017 после этой даты не наблюдается
    ax.axvline(end, color="black", ls="--", lw=1, label="конец данных 2017 (15.08)")
    ticks = pd.date_range("2017-07-01", "2017-10-01", freq="MS")
    ax.set_xticks(ticks.dayofyear, ticks.strftime("%d.%m"))
    ax.set_ylabel("продажи за день, тыс.")
    ax.set_title(f"{SCHOOL}: сумма по всем магазинам, июль - октябрь")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
