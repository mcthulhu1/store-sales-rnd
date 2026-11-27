"""ДЗ №3: анализ ошибок, срезы и сводка проверок устойчивости E2_calendar.

Читает результат hw3_experiments.py (outputs/hw3/predictions, models, robustness_*.csv)
и пишет в outputs/hw3/ таблицы errors_*.csv, slices_*.csv, robustness_*_summary.csv
и графики в outputs/hw3/plots/. Обучения здесь нет; модели seed 42 нужны только
для контрфактической проверки пропусков промо (ошибка 2).

Запуск: python src/hw3_analysis.py --config configs/hw3/hw3.yaml
"""

import argparse
import os

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from hw3_common import base_tags, fold_bounds, read_config, setup_mlflow
from utils import add_features, feature_list, load_data

SCHOOL = "SCHOOL AND OFFICE SUPPLIES"
SCHOOL_STATES = ["Pichincha", "Tungurahua"]
HIST_DAYS = 28  # окно истории перед началом фолда для срезов по объему и промо
LATE_JULY = ("2017-07-27", "2017-08-02")  # серия недопрогноза по сети (график err5)
SPIKE_RATIO, SPIKE_MIN = 5, 20  # всплеск: факт >= 5 x уровня ряда и не меньше 20
VOLUME_LABELS = ["0 (нет продаж)", "(0, 1]", "(1, 10]", "(10, 100]", "(100, 1000]", "> 1000"]
ORDER = {
    "type": ["A", "B", "C", "D", "E"],
    "volume": VOLUME_LABELS,
    "promo_class": [
        "нет промо",
        "промо обычного уровня",
        "промо >= 3x обычного",
        "промо, а 28 дней до фолда не было",
    ],
    "day_type": ["будни", "обычные сб-вс", "сб-вс длинных выходных", "национальный выходной"],
}


def sq(pred, fact):
    return (np.log1p(np.clip(pred, 0, None)) - np.log1p(fact)) ** 2


def check_inputs(config, out, preds, n_series):
    """Анализ только по полному прогону: все seed, фолды, доли train и sigma, уникальные ключи.

    Смоук-прогон пишет в другой каталог (--output-dir), но если входы неполные - падаем сразу.
    """
    folds = set(range(config["n_folds"]))
    expected_rows = config["n_folds"] * config["horizon"] * n_series
    problems = []
    for seed, p in preds.items():
        if set(p["fold"]) != folds:
            problems.append(f"seed {seed}: фолды {sorted(set(p['fold']))}, нужны {sorted(folds)}")
        if len(p) != expected_rows:
            problems.append(f"seed {seed}: строк {len(p)}, нужно {expected_rows}")
        if p.duplicated(["fold", "date", "store_nbr", "family"]).any():
            problems.append(f"seed {seed}: повторяются ключи fold, date, store_nbr, family")
    for name, levels in [
        ("robustness_train_fraction_by_fold.csv", config["train_fractions"]),
        ("robustness_noise_by_fold.csv", config["noise_sigmas"]),
    ]:
        path = f"{out}/{name}"
        if not os.path.exists(path):
            problems.append(f"нет {path}")
            continue
        r = pd.read_csv(path)
        got = set(zip(r["level"], r["seed"], r["fold"], strict=True))
        need = {(lvl, s, f) for lvl in levels for s in config["seeds"] for f in folds}
        if got != need or len(r) != len(need):
            problems.append(f"{name}: {len(r)} строк, нужно {len(need)} (уровень x seed x фолд)")
    for fold in folds:
        model = f"{out}/models/E2_calendar_seed{config['main_seed']}_fold{fold}.txt"
        if not os.path.exists(model):
            problems.append(f"нет {model}")
    if problems:
        raise SystemExit(
            "Входы анализа неполные, сначала полный прогон scripts/run_hw3.sh:\n- "
            + "\n- ".join(problems)
        )


def load_predictions(out, seed):
    path = f"{out}/predictions/E2_calendar_seed{seed}.csv.gz"
    if not os.path.exists(path):
        raise SystemExit(f"нет {path}: сначала полный прогон scripts/run_hw3.sh")
    p = pd.read_csv(f"{out}/predictions/E2_calendar_seed{seed}.csv.gz", parse_dates=["date"])
    p["log_err"] = np.log1p(p["pred"]) - np.log1p(p["sales"])
    p["se"] = p["log_err"] ** 2
    return p


def history_stats(df, bounds):
    """Статистика ряда за 28 дней до начала фолда: известна на момент прогноза, утечки нет."""
    rows = []
    for fold, cutoff, _ in bounds:
        h = df[(df["date"] >= cutoff - pd.Timedelta(days=HIST_DAYS)) & (df["date"] < cutoff)]
        g = h.groupby(["store_nbr", "family"], observed=True).agg(
            sales_hist=("sales", "mean"),
            zero_hist=("sales", lambda s: (s == 0).mean()),
            promo_hist=("onpromotion", "mean"),
        )
        rows.append(g.reset_index().assign(fold=fold))
    return pd.concat(rows)


def add_slice_columns(p, stores):
    p = p.merge(stores[["store_nbr", "city", "state"]], on="store_nbr", how="left")
    p["volume"] = pd.cut(
        p["sales_hist"],
        [-1, 0, 1, 10, 100, 1000, np.inf],
        labels=VOLUME_LABELS,
    )
    p["promo_class"] = np.select(
        [
            p["onpromotion"] == 0,
            p["promo_hist"] == 0,
            p["onpromotion"] >= 3 * p["promo_hist"],
        ],
        ["нет промо", "промо, а 28 дней до фолда не было", "промо >= 3x обычного"],
        "промо обычного уровня",
    )
    p["day_type"] = np.select(
        [p["holiday"] == 1, p["long_weekend"] == 1, p["dayofweek"] >= 5],
        ["национальный выходной", "сб-вс длинных выходных", "обычные сб-вс"],
        "будни",
    )
    p["local_holiday_slice"] = np.where(p["local_holiday"] == 1, "местный праздник", "нет")
    level = np.maximum(np.expm1(p["roll_mean_28"]), 1)
    p["spike"] = (p["sales"] >= SPIKE_RATIO * level) & (p["sales"] >= SPIKE_MIN)
    return p


def bootstrap_ci_rows(se_sum, n_rows, n_bootstrap, seed):
    """95% интервал RMSLE группы: выбираем ряды с повторами, RMSLE по всем строкам выбранных рядов.

    То же, что utils.bootstrap_ci, но ряды внутри среза могут иметь разное число строк
    (например, промо было только в часть дней), поэтому суммируем ошибки и строки, а не средние.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(se_sum), size=len(se_sum))
        scores.append(np.sqrt(se_sum[idx].sum() / n_rows[idx].sum()))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def group_metrics(p, by, n_boot, seed):
    """Размер группы, RMSLE, смещение в логах, 95% бутстрап по рядам, доля общей ошибки."""
    total = p["se"].sum()
    rows = []
    for key, g in p.groupby(by, observed=True, sort=False):
        per_series = g.groupby(["store_nbr", "family"], observed=True)["se"].agg(["sum", "size"])
        low, high = bootstrap_ci_rows(
            per_series["sum"].values, per_series["size"].values, n_boot, seed
        )
        rows.append(
            {
                by: key,
                "n_rows": len(g),
                "n_series": len(per_series),
                "n_days": g["date"].nunique(),
                "rmsle": np.sqrt(g["se"].mean()),
                "ci_low": low,
                "ci_high": high,
                "bias_log": g["log_err"].mean(),
                "fact_mean": g["sales"].mean(),
                "pred_mean": g["pred"].mean(),
                "share_of_sq_error": g["se"].sum() / total,
            }
        )
    return pd.DataFrame(rows)


def slice_table(preds, by, main_seed, n_boot):
    """Метрики среза на основном seed + min-max RMSLE по всем seed."""
    main = group_metrics(preds[main_seed], by, n_boot, main_seed)
    per_seed = pd.DataFrame(
        {seed: p.groupby(by, observed=True)["se"].mean() ** 0.5 for seed, p in preds.items()}
    )
    main["rmsle_seed_min"] = main[by].map(per_seed.min(axis=1))
    main["rmsle_seed_max"] = main[by].map(per_seed.max(axis=1))
    # малая группа: меньше 30 рядов или меньше 5 дней - метрика может сильно зависеть от случая
    main["small_group"] = (main["n_series"] < 30) | (main["n_days"] < 5)
    return main


# ---------- ошибки ----------


def error_school(p, df, out):
    """Ошибка 1: канцтовары по провинциям, факт и прогноз по фолдам, тот же период год назад."""
    s = p[p["family"] == SCHOOL]
    by_state = s.groupby(["state", "fold"])[["sales", "pred"]].sum().unstack("fold")
    by_state.columns = [f"{c}_f{f}" for c, f in by_state.columns]
    # тот же период годом раньше (сдвиг 364 дня, те же дни недели)
    last_year = df[df["family"] == SCHOOL].copy()
    last_year["date"] = last_year["date"] + pd.Timedelta(days=364)
    last_year = last_year.merge(
        s[["date", "store_nbr", "fold", "state"]], on=["date", "store_nbr"], how="inner"
    )
    ly = last_year.groupby(["state", "fold"])["sales"].sum().unstack("fold")
    ly.columns = [f"sales_year_ago_f{f}" for f in ly.columns]
    table = by_state.join(ly).sort_values("sales_f3", ascending=False)
    table.round(1).to_csv(f"{out}/errors_1_school_by_state.csv")

    # проверка гипотезы о региональном сезоне: продажи канцтоваров по месяцам за всю историю train
    store_state = p.drop_duplicates("store_nbr").set_index("store_nbr")["state"]
    hist = df[(df["family"] == SCHOOL) & (df["date"] >= "2015-01-01")].copy()
    hist["region"] = np.where(
        hist["store_nbr"].map(store_state).isin(SCHOOL_STATES), "Pichincha+Tungurahua", "остальные"
    )
    monthly = hist.groupby([hist["date"].dt.to_period("M"), "region"])["sales"].sum().unstack()
    monthly.round(0).to_csv(f"{out}/errors_1_school_by_month.csv")

    share = {}
    for fold in [2, 3]:
        f = s[s["fold"] == fold]
        share[fold] = f.loc[f["state"].isin(SCHOOL_STATES), "se"].sum() / f["se"].sum()
    # RMSLE канцтоваров в двух провинциях и вне их: все фолды и фолд 3
    outside = ~s["state"].isin(SCHOOL_STATES)
    for name, mask in [("pichincha_tungurahua", ~outside), ("other_states", outside)]:
        share[f"rmsle_{name}"] = np.sqrt(s.loc[mask, "se"].mean())
        share[f"rmsle_{name}_f3"] = np.sqrt(s.loc[mask & (s["fold"] == 3), "se"].mean())

    in_states = s["state"].isin(SCHOOL_STATES)
    n_in, n_out = s[in_states]["store_nbr"].nunique(), s[~in_states]["store_nbr"].nunique()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    for ax, mask, title in [
        (axes[0], in_states, f"Pichincha + Tungurahua ({n_in} магазинов)"),
        (axes[1], ~in_states, f"остальные провинции ({n_out} магазина)"),
    ]:
        daily = s[mask].groupby("date")[["sales", "pred"]].sum()
        ax.plot(daily.index, daily["sales"], color="black", label="факт")
        ax.plot(daily.index, daily["pred"], color="tab:blue", label="прогноз E2_calendar")
        ax.set_title(title, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("продажи за день, сумма")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{SCHOOL}: факт и прогноз по дням, фолды 0-3, seed 42", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out}/plots/err1_school_by_region.png", dpi=110)
    plt.close()
    n_stores = s[s["state"].isin(SCHOOL_STATES)]["store_nbr"].nunique()
    return table, share, n_stores


def find_promo_gaps(df, start):
    """Кандидаты на аномалию промо: магазино-день, где промо 0 во всех категориях,
    а накануне или на следующий день > 100.

    Не считаем закрытые дни: дни, добавленные load_data (25 декабря, use_for_train = False),
    и дни, когда у магазина нет продаж совсем (1 января и другие закрытия).
    """
    sd = (
        df.groupby(["store_nbr", "date"], observed=True)
        .agg(
            promo=("onpromotion", "sum"),
            sales=("sales", "sum"),
            in_train_csv=("use_for_train", "any"),
        )
        .reset_index()
    )
    sd = sd.sort_values(["store_nbr", "date"])
    sd["promo_prev"] = sd.groupby("store_nbr", observed=True)["promo"].shift(1)
    sd["promo_next"] = sd.groupby("store_nbr", observed=True)["promo"].shift(-1)
    detected = sd[
        (sd["promo"] == 0)
        & ((sd["promo_prev"] > 100) | (sd["promo_next"] > 100))
        & (sd["date"] >= start)
    ]
    closed = ~detected["in_train_csv"] | (detected["sales"] == 0)
    counts = {
        "detected_raw": len(detected),
        "excluded_added_by_load_data": int((~detected["in_train_csv"]).sum()),
        "excluded_zero_sales": int((detected["in_train_csv"] & (detected["sales"] == 0)).sum()),
    }
    return detected[~closed], counts


def error_promo_gaps(p, df, config, bounds, out):
    """Ошибка 2: кандидаты на аномалию промо. Контрфактическая проверка чувствительности:
    промо = среднее соседних дней, та же модель. Истинное промо неизвестно.
    """
    features = feature_list(config)
    gaps, gap_counts = find_promo_gaps(df, pd.Timestamp(config["train_start"]))
    promo = df.set_index(["store_nbr", "family", "date"])["onpromotion"]
    rows, cf_rows = [], []
    for fold, cutoff, val_end in bounds:
        g = gaps[(gaps["date"] >= cutoff) & (gaps["date"] <= val_end)]
        if g.empty:
            continue
        model = lgb.Booster(
            model_file=f"{config['output_dir']}/models/E2_calendar_seed{config['main_seed']}_fold{fold}.txt"
        )
        for _, gap in g.iterrows():
            v = df[(df["store_nbr"] == gap["store_nbr"]) & (df["date"] == gap["date"])].copy()
            keys = list(zip(v["store_nbr"], v["family"], strict=True))
            prev = [promo[(s, f, gap["date"] - pd.Timedelta(days=1))] for s, f in keys]
            nxt = [promo[(s, f, gap["date"] + pd.Timedelta(days=1))] for s, f in keys]
            pred = np.clip(np.expm1(model.predict(v[features])), 0, None)
            v_cf = v.assign(onpromotion=(np.array(prev) + np.array(nxt)) / 2)
            pred_cf = np.clip(np.expm1(model.predict(v_cf[features])), 0, None)
            rows.append(
                {
                    "fold": fold,
                    "date": gap["date"].date(),
                    "store_nbr": gap["store_nbr"],
                    "promo_prev_day": gap["promo_prev"],
                    "promo_next_day": gap["promo_next"],
                    "fact_sum": v["sales"].sum(),
                    "pred_sum": pred.sum(),
                    "pred_sum_promo_filled": pred_cf.sum(),
                    "rmsle": np.sqrt(sq(pred, v["sales"].values).mean()),
                    "rmsle_promo_filled": np.sqrt(sq(pred_cf, v["sales"].values).mean()),
                }
            )
            cf_rows.append(
                v[["date", "store_nbr", "family"]].assign(pred_cf=pred_cf, pred_check=pred)
            )
    table = pd.DataFrame(rows)
    table.round(3).to_csv(f"{out}/errors_2_promo_gaps.csv", index=False)
    cf = pd.concat(cf_rows) if cf_rows else pd.DataFrame()

    # проверка: прогноз сохраненной модели совпадает с прогнозом из predictions
    check = cf.merge(p[["date", "store_nbr", "family", "pred"]], on=["date", "store_nbr", "family"])
    max_diff = float(np.abs(check["pred"] - check["pred_check"]).max())

    # пример: магазин 36, фолд 3 - промо и продажи по дням
    ex_store, ex_fold = 36, 3
    ex = p[(p["store_nbr"] == ex_store) & (p["fold"] == ex_fold)]
    daily = ex.groupby("date")[["sales", "pred", "onpromotion"]].sum()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(daily.index, daily["sales"], color="black", label="факт, сумма по 33 категориям")
    ax.plot(daily.index, daily["pred"], color="tab:blue", label="прогноз E2_calendar")
    gap_days = table[(table["store_nbr"] == ex_store) & (table["fold"] == ex_fold)]
    for _, r in gap_days.iterrows():
        d = pd.Timestamp(r["date"])
        ax.scatter(
            [d],
            [r["pred_sum_promo_filled"]],
            color="tab:green",
            zorder=5,
            label="прогноз, если промо = среднее соседних дней",
        )
    ax2 = ax.twinx()
    ax2.bar(
        daily.index, daily["onpromotion"], color="tab:orange", alpha=0.25, label="товаров в промо"
    )
    ax2.set_ylabel("товаров в промо")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylabel("продажи")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    lines = ax.get_legend_handles_labels()
    bars = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + bars[0], lines[1] + bars[1], fontsize=7, loc="lower right")
    ax.set_title(
        f"Магазин {ex_store}, фолд {ex_fold}: 01.08 промо = 0 во всех категориях", fontsize=10
    )
    plt.tight_layout()
    plt.savefig(f"{out}/plots/err2_promo_gap_store36.png", dpi=110)
    plt.close()
    return table, gaps, cf, max_diff, gap_counts


def error_low_volume(p, out):
    """Ошибка 3: маленькие и прерывистые ряды. Разложение ошибки ряда на смещение и разброс.

    Для ряда на фолде: MSE в логах = (среднее log_err)^2 + дисперсия log_err.
    «Оракул» - лучший постоянный прогноз ряда на фолде (среднее log1p(факт)): его ошибка -
    это разброс факта внутри фолда, ее не убрать никаким постоянным уровнем.
    """
    g = p.groupby(["fold", "store_nbr", "family"], observed=True)
    per = g.agg(
        volume=("volume", "first"),
        mse=("se", "mean"),
        bias=("log_err", "mean"),
        fact_log_var=("sales", lambda s: np.log1p(s).var(ddof=0)),
        zero_share=("sales", lambda s: (s == 0).mean()),
    ).reset_index()
    per["bias_sq"] = per["bias"] ** 2
    table = per.groupby("volume", observed=True).agg(
        n_series_folds=("mse", "size"),
        rmsle=("mse", lambda s: np.sqrt(s.mean())),
        bias_sq_share=("bias_sq", "sum"),
        mse_sum=("mse", "sum"),
        oracle_const_rmsle=("fact_log_var", lambda s: np.sqrt(s.mean())),
        zero_days_share=("zero_share", "mean"),
    )
    table["bias_sq_share"] = table["bias_sq_share"] / table["mse_sum"]
    table = table.drop(columns="mse_sum").reset_index()
    table.round(4).to_csv(f"{out}/errors_3_low_volume.csv", index=False)

    # пример типичного маленького ряда: во всех 4 фолдах в группе (1, 10],
    # RMSLE ряда ближе всего к RMSLE всей группы
    group = "(1, 10]"
    group_rmsle = table.set_index("volume").loc[group, "rmsle"]
    per_series = p.groupby(["store_nbr", "family"], observed=True).agg(
        n_group=("volume", lambda v: (v == group).sum()),
        rmsle=("se", lambda e: np.sqrt(e.mean())),
    )
    candidates = per_series[per_series["n_group"] == len(p["fold"].unique()) * 16]
    typical = (candidates["rmsle"] - group_rmsle).abs().idxmin()
    examples = [
        (
            typical,
            f"типичный ряд группы {group}: RMSLE ряда {candidates.loc[typical, 'rmsle']:.2f}",
        ),
        ((19, "GROCERY II"), "ряд №2 по вкладу в ошибку: рост в фолде 2, прогноз растет в фолде 3"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6))
    for ax, ((store, family), title) in zip(axes, examples, strict=True):
        ex = p[(p["store_nbr"] == store) & (p["family"] == family)]
        for fold in range(4):
            f = ex[ex["fold"] == fold]
            ax.plot(f["date"], f["sales"], color="black", label="факт" if fold == 0 else None)
            ax.plot(f["date"], f["pred"], color="tab:blue", label="прогноз" if fold == 0 else None)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        ax.set_title(f"Магазин {store}, {family} - {title}", fontsize=9)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out}/plots/err3_series_examples.png", dpi=110)
    plt.close()
    pd.Series({"typical_store": typical[0], "typical_family": typical[1]}).to_csv(
        f"{out}/errors_3_typical_series.csv", header=["value"]
    )
    return table


def error_spikes(p, df, config, bounds, out):
    """Ошибка 4: резкие всплески (факт >= 5x уровня ряда). Частота всплесков в истории train."""
    total = p["se"].sum()
    sp = p[p["spike"]]
    summary = {
        "n_rows": len(sp),
        "share_rows": len(sp) / len(p),
        "share_of_sq_error": sp["se"].sum() / total,
        "rmsle_spike": np.sqrt(sp["se"].mean()),
        "rmsle_other": np.sqrt(p.loc[~p["spike"], "se"].mean()),
        "promo_share_spike": (sp["onpromotion"] > 0).mean(),
        "promo_share_other": (p.loc[~p["spike"], "onpromotion"] > 0).mean(),
        "n_spike_school": int((sp["family"] == SCHOOL).sum()),
        "median_fact_to_pred": float((sp["sales"] / np.maximum(sp["pred"], 1)).median()),
    }
    by_family = sp.groupby("family").agg(n=("se", "size"), share_of_sq_error=("se", "sum"))
    by_family["share_of_sq_error"] /= total
    by_family = by_family.sort_values("n", ascending=False)
    by_family.round(4).to_csv(f"{out}/errors_4_spikes_by_family.csv")

    # база: как часто такие всплески бывают в train (весь период обучения последнего фолда)
    _, cutoff, _ = bounds[-1]
    hist = df[(df["date"] >= config["train_start"]) & (df["date"] < cutoff) & df["use_for_train"]]
    level = np.maximum(np.expm1(hist["roll_mean_28"]), 1)
    is_spike = (hist["sales"] >= SPIKE_RATIO * level) & (hist["sales"] >= SPIKE_MIN)
    base = (
        pd.DataFrame(
            {"promo": np.where(hist["onpromotion"] > 0, "промо", "нет промо"), "spike": is_spike}
        )
        .groupby("promo")["spike"]
        .agg(["size", "mean"])
        .rename(columns={"size": "n_rows", "mean": "spike_rate"})
    )
    base.loc["все", ["n_rows", "spike_rate"]] = [len(hist), is_spike.mean()]
    base.round(5).to_csv(f"{out}/errors_4_spike_rate_train.csv")
    summary["spike_rate_train"] = float(is_spike.mean())
    summary["spike_rate_train_no_school"] = float(is_spike[hist["family"] != SCHOOL].mean())
    pd.Series(summary).round(5).to_csv(f"{out}/errors_4_spikes_summary.csv", header=["value"])
    sp.sort_values("se", ascending=False).head(20)[
        ["fold", "date", "store_nbr", "family", "onpromotion", "sales", "pred", "roll_mean_28"]
    ].round(2).to_csv(f"{out}/errors_4_spikes_top.csv", index=False)
    return summary, by_family, base


def network_bias_windows(daily):
    """Смещение суммы по сети за окно = сумма прогноза / сумма факта - 1 (как в compare.py)."""
    result = {}
    for name, (start, end) in {
        "12_13_aug": ("2017-08-12", "2017-08-13"),
        "31jul_1aug": ("2017-07-31", "2017-08-01"),
        "1aug": ("2017-08-01", "2017-08-01"),
    }.items():
        d = daily.loc[start:end]
        result[name] = 100 * (d["pred"].sum() / d["fact"].sum() - 1)
        result[f"{name}_promo_filled"] = 100 * (d["pred_promo_filled"].sum() / d["fact"].sum() - 1)
    return result


def error_calendar(p, cf, out):
    """Ошибка 5: смещение суммы по сети по дням всех 4 фолдов; 01.08 с заполненным промо."""
    daily = p.groupby("date").agg(
        fold=("fold", "first"),
        fact=("sales", "sum"),
        pred=("pred", "sum"),
        rmsle=("se", lambda s: np.sqrt(s.mean())),
        holiday=("holiday", "max"),
        long_weekend=("long_weekend", "max"),
        dayofweek=("dayofweek", "first"),
    )
    daily["bias_pct"] = 100 * (daily["pred"] / daily["fact"] - 1)
    # то же, если в магазино-днях с пропуском промо подставить контрфактический прогноз
    adj = p[["date", "store_nbr", "family", "pred"]].merge(
        cf[["date", "store_nbr", "family", "pred_cf"]],
        on=["date", "store_nbr", "family"],
        how="left",
    )
    adj["pred_adj"] = adj["pred_cf"].fillna(adj["pred"])
    daily["pred_promo_filled"] = adj.groupby("date")["pred_adj"].sum()
    daily["bias_pct_promo_filled"] = 100 * (daily["pred_promo_filled"] / daily["fact"] - 1)
    daily.round(3).to_csv(f"{out}/errors_5_network_bias_by_day.csv")

    # окно недопрогноза в конце июля: по регионам и категориям, отдельно CLEANING по дням
    window = p[p["date"].between(*LATE_JULY)]
    region = np.where(window["state"].isin(SCHOOL_STATES), "Pichincha+Tungurahua", "остальные")
    parts = []
    for name, keys in [("регион", region), ("категория", window["family"])]:
        g = window.groupby(keys)[["sales", "pred"]].sum()
        g["bias_pct"] = 100 * (g["pred"] / g["sales"] - 1)
        g["gap"] = g["pred"] - g["sales"]
        parts.append(g.sort_values("gap").assign(slice=name))
    total = window[["sales", "pred"]].sum()
    parts.append(
        pd.DataFrame(
            {
                "sales": [total["sales"]],
                "pred": [total["pred"]],
                "bias_pct": [100 * (total["pred"] / total["sales"] - 1)],
                "gap": [total["pred"] - total["sales"]],
                "slice": ["вся сеть"],
            },
            index=["вся сеть"],
        )
    )
    pd.concat(parts).round(1).to_csv(f"{out}/errors_5_late_july_window.csv")
    cleaning = p[p["family"] == "CLEANING"].groupby("date")[["sales", "pred", "onpromotion"]].sum()
    cleaning["bias_pct"] = 100 * (cleaning["pred"] / cleaning["sales"] - 1)
    cleaning.loc["2017-07-15":"2017-08-10"].round(1).to_csv(f"{out}/errors_5_cleaning_by_day.csv")

    fig, ax = plt.subplots(figsize=(11, 3.6))
    colors = np.where(
        daily["long_weekend"] == 1,
        "tab:red",
        np.where(daily["dayofweek"] >= 5, "tab:orange", "tab:blue"),
    )
    ax.bar(daily.index, daily["bias_pct"], color=colors, width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    for d in [p["date"].min()] + [p[p["fold"] == f]["date"].min() for f in range(1, 4)]:
        ax.axvline(d - pd.Timedelta(hours=12), color="grey", lw=0.6, ls="--")
    ax.set_ylabel("прогноз / факт - 1, %")
    ax.set_title(
        "Смещение суммы прогноза по сети по дням (синий - будни, оранжевый - сб-вс, "
        "красный - длинные выходные), пунктир - границы фолдов",
        fontsize=9,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    plt.tight_layout()
    plt.savefig(f"{out}/plots/err5_network_bias_by_day.png", dpi=110)
    plt.close()
    return daily


# ---------- устойчивость ----------


def robustness(out, seeds_range):
    frac = pd.read_csv(f"{out}/robustness_train_fraction_by_fold.csv")
    noise = pd.read_csv(f"{out}/robustness_noise_by_fold.csv")
    tables = {}
    for name, d in [("train_fraction", frac), ("noise", noise)]:
        per_seed = d.groupby(["level", "seed"])["rmsle"].mean().unstack("seed")
        ref_level = 1.0 if name == "train_fraction" else 0.0
        delta = per_seed - per_seed.loc[ref_level]
        t = pd.DataFrame(
            {
                "rmsle_mean": per_seed.mean(axis=1),
                "rmsle_seed_min": per_seed.min(axis=1),
                "rmsle_seed_max": per_seed.max(axis=1),
                "delta_vs_ref_mean": delta.mean(axis=1),
                "delta_vs_ref_min": delta.min(axis=1),
                "delta_vs_ref_max": delta.max(axis=1),
                "seeds_worse": (delta > 0).sum(axis=1),
            }
        )
        by_fold = d.groupby(["level", "fold"])["rmsle"].mean().unstack("fold")
        by_fold.columns = [f"rmsle_f{c}" for c in by_fold.columns]
        t = t.join(by_fold)
        if name == "train_fraction":
            days = (
                d[d["seed"] == d["seed"].iloc[0]]
                .groupby("level")
                .agg(
                    train_days_min=("train_days", "min"),
                    train_days_max=("train_days", "max"),
                    train_rows_fold3=("train_rows", "last"),
                    train_start_fold3=("train_start", "last"),
                )
            )
            t = days.join(t)
        t = t.sort_index(ascending=(name == "noise"))
        t.round(5).to_csv(f"{out}/robustness_{name}_summary.csv")
        tables[name] = t

    # какие категории дают разницу на фолде 3 между долей 0.35 и 1.0 (среднее по seed)
    fam = pd.read_csv(f"{out}/robustness_train_fraction_by_family.csv")
    f3 = (
        fam[fam["fold"] == 3].groupby(["level", "family"])["sq_log_err_sum"].mean().unstack("level")
    )
    n_fold3 = fam[
        (fam["fold"] == 3) & (fam["level"] == 1.0) & (fam["seed"] == fam["seed"].iloc[0])
    ]["n"].sum()
    contrib = pd.DataFrame(
        {
            # вклад категории в разницу MSE фолда 3 (сумма по категориям = MSE(доля) - MSE(1.0))
            f"mse_contrib_{lvl}": (f3[lvl] - f3[1.0]) / n_fold3
            for lvl in sorted(f3.columns)
            if lvl != 1.0
        }
    )
    contrib["rmsle_family_1.0"] = np.sqrt(f3[1.0] / (n_fold3 / 33))
    contrib.sort_values("mse_contrib_0.35").round(5).to_csv(
        f"{out}/robustness_train_fraction_fold3_by_family.csv"
    )
    tables["fold3_family"] = contrib

    # фолд 3: RMSLE всех категорий, без канцтоваров и только канцтоваров (среднее по seed)
    def fold3_rmsle(part):
        per_seed = part.groupby(["level", "seed"])[["sq_log_err_sum", "n"]].sum()
        return np.sqrt(per_seed["sq_log_err_sum"] / per_seed["n"]).groupby("level").mean()

    fam3 = fam[fam["fold"] == 3]
    school = fam3["family"] == SCHOOL
    split = pd.DataFrame(
        {
            "rmsle_f3_all": fold3_rmsle(fam3),
            "rmsle_f3_without_school": fold3_rmsle(fam3[~school]),
            "rmsle_f3_school_only": fold3_rmsle(fam3[school]),
        }
    ).sort_index(ascending=False)
    split.round(4).to_csv(f"{out}/robustness_train_fraction_fold3_school.csv")
    tables["fold3_school"] = split

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, name, xlabel in [
        (axes[0], "train_fraction", "доля train по времени (последние дни перед фолдом)"),
        (axes[1], "noise", "sigma шума в лагах и скользящих средних (log1p)"),
    ]:
        t = tables[name].sort_index()
        ax.fill_between(
            t.index,
            t["rmsle_seed_min"],
            t["rmsle_seed_max"],
            color="tab:blue",
            alpha=0.25,
            label="среднее по фолдам, min-max по 3 seed",
        )
        ax.plot(
            t.index, t["rmsle_mean"], color="tab:blue", marker="o", label="среднее по фолдам и seed"
        )
        for fold in range(4):
            ax.plot(t.index, t[f"rmsle_f{fold}"], lw=0.8, ls="--", alpha=0.7, label=f"фолд {fold}")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("RMSLE")
        ax.grid(alpha=0.3)
    axes[0].invert_xaxis()
    axes[0].set_title("Проверка 1: сокращение train", fontsize=10)
    axes[1].set_title("Проверка 2: шум во входных признаках (без переобучения)", fontsize=10)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{out}/plots/robustness.png", dpi=110)
    plt.close()
    return tables


def plot_slices(slices, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (name, t) in zip(axes.ravel(), slices.items(), strict=True):
        t = t.copy()
        y = np.arange(len(t))
        err = [t["rmsle"] - t["ci_low"], t["ci_high"] - t["rmsle"]]
        colors = ["tab:grey" if s else "tab:blue" for s in t["small_group"]]
        ax.barh(y, t["rmsle"], xerr=err, color=colors, alpha=0.85, capsize=2)
        labels = [f"{k}  (рядов {n})" for k, n in zip(t[name], t["n_series"], strict=True)]
        ax.set_yticks(y, labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("RMSLE, 95% бутстрап по рядам", fontsize=8)
        ax.set_title(name, fontsize=9)
    fig.suptitle(
        "Срезы, E2_calendar, seed 42, фолды 0-3 (серым - меньше 30 рядов или 5 дней)", fontsize=10
    )
    plt.tight_layout()
    plt.savefig(f"{out}/plots/slices.png", dpi=110)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hw3/hw3.yaml")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    out, main_seed = config["output_dir"], config["main_seed"]
    n_boot = config["n_bootstrap"]
    os.makedirs(f"{out}/plots", exist_ok=True)

    df = add_features(load_data(config["data_dir"]), config)
    stores = pd.read_csv(f"{config['data_dir']}/stores.csv")
    bounds = fold_bounds(df["date"].max(), config["horizon"], config["n_folds"])
    hist = history_stats(df, bounds)
    n_series = df.groupby(["store_nbr", "family"], observed=True).ngroups
    raw = {seed: load_predictions(out, seed) for seed in config["seeds"]}
    check_inputs(config, out, raw, n_series)
    preds = {}
    for seed, p in raw.items():
        p = p.merge(hist, on=["fold", "store_nbr", "family"], how="left")
        preds[seed] = add_slice_columns(p, stores)
    p = preds[main_seed]
    print(f"строк validation: {len(p)}, RMSLE seed {main_seed}: {np.sqrt(p['se'].mean()):.4f}")

    # худшие строки и ряды - отправная точка для поиска типовых ошибок
    p.sort_values("se", ascending=False).head(30)[
        [
            "fold",
            "date",
            "store_nbr",
            "family",
            "state",
            "onpromotion",
            "sales",
            "pred",
            "roll_mean_28",
        ]
    ].round(3).to_csv(f"{out}/errors_top_rows.csv", index=False)
    top_series = (
        p.groupby(["store_nbr", "family"], observed=True)
        .agg(
            state=("state", "first"),
            rmsle=("se", lambda s: np.sqrt(s.mean())),
            fact_mean=("sales", "mean"),
            pred_mean=("pred", "mean"),
            bias_log=("log_err", "mean"),
            share_of_sq_error=("se", lambda s: s.sum() / p["se"].sum()),
        )
        .sort_values("share_of_sq_error", ascending=False)
        .head(30)
    )
    top_series.round(4).to_csv(f"{out}/errors_top_series.csv")

    school, school_share, n_school_stores = error_school(p, df, out)
    gaps, all_gaps, cf, gap_check, gap_counts = error_promo_gaps(p, df, config, bounds, out)
    low = error_low_volume(p, out)
    spikes, _, spike_base = error_spikes(p, df, config, bounds, out)
    daily = error_calendar(p, cf, out)
    pd.Series(
        {
            "school_stores_in_pichincha_tungurahua": n_school_stores,
            "school_sq_error_share_pichincha_tungurahua_f2": school_share[2],
            "school_sq_error_share_pichincha_tungurahua_f3": school_share[3],
            **{f"school_{k}": v for k, v in school_share.items() if isinstance(k, str)},
            "promo_zero_detected_raw_since_train_start": gap_counts["detected_raw"],
            "promo_zero_excluded_added_by_load_data": gap_counts["excluded_added_by_load_data"],
            "promo_zero_excluded_zero_sales": gap_counts["excluded_zero_sales"],
            "promo_anomaly_candidates_since_train_start": len(all_gaps),
            "promo_anomaly_candidates_in_validation": len(gaps),
            "saved_model_vs_predictions_max_abs_diff_on_candidates": gap_check,
            **{f"network_bias_pct_{k}": v for k, v in network_bias_windows(daily).items()},
        }
    ).to_csv(f"{out}/errors_summary.csv", header=["value"])
    print("канцтовары:\n", school.round(0).head(6).to_string())
    print("доля ошибки канцтоваров в Pichincha+Tungurahua:", school_share, n_school_stores)
    print(
        f"кандидаты на аномалию промо с {config['train_start']}: {len(all_gaps)} {gap_counts}, "
        f"сверка модели {gap_check:.2e}"
    )
    print(gaps.round(3).to_string(index=False))
    print(low.round(4).to_string(index=False))
    print(pd.Series(spikes).round(4).to_string())
    print(spike_base.round(5).to_string())
    sel = daily.loc[["2017-07-31", "2017-08-01", "2017-08-11", "2017-08-12", "2017-08-13"]]
    print(sel[["fact", "pred", "bias_pct", "bias_pct_promo_filled"]].round(2).to_string())
    for name, days in [
        ("31.07-01.08", ["2017-07-31", "2017-08-01"]),
        ("12-13.08", ["2017-08-12", "2017-08-13"]),
    ]:
        d = daily.loc[days]
        print(
            name,
            "смещение:",
            round(100 * (d["pred"].sum() / d["fact"].sum() - 1), 2),
            "с промо:",
            round(100 * (d["pred_promo_filled"].sum() / d["fact"].sum() - 1), 2),
        )
    print(pd.read_csv(f"{out}/errors_5_late_july_window.csv").head(12).to_string())
    print(
        "дни |смещение| > 10%:\n",
        daily[daily["bias_pct"].abs() > 10][["fold", "bias_pct"]].round(1),
    )

    # срезы
    slices = {}
    for name, col in [
        ("тип магазина", "type"),
        ("объем ряда: средние продажи за 28 дней до фолда", "volume"),
        ("промо в день прогноза", "promo_class"),
        ("тип дня", "day_type"),
    ]:
        t = slice_table(preds, col, main_seed, n_boot).rename(columns={col: name})
        if col in ORDER:
            t = t.set_index(name).loc[[k for k in ORDER[col] if k in set(t[name])]].reset_index()
        slices[name] = t
    file_names = {
        "тип магазина": "type",
        "объем ряда: средние продажи за 28 дней до фолда": "volume",
        "промо в день прогноза": "promo",
        "тип дня": "day_type",
    }
    for name, t in slices.items():
        t.round(4).to_csv(f"{out}/slices_{file_names[name]}.csv", index=False)
        print(t.round(4).to_string(index=False))
    extra = {
        "family": slice_table(preds, "family", main_seed, n_boot).sort_values(
            "rmsle", ascending=False
        ),
        "cluster": slice_table(preds, "cluster", main_seed, n_boot).sort_values("cluster"),
        "local_holiday": slice_table(preds, "local_holiday_slice", main_seed, n_boot),
        "fold": slice_table(preds, "fold", main_seed, n_boot).sort_values("fold"),
    }
    for name, t in extra.items():
        t.round(4).to_csv(f"{out}/slices_{name}.csv", index=False)
    print(extra["family"].round(4).head(6).to_string(index=False))
    print(extra["family"].round(4).tail(5).to_string(index=False))
    print(extra["local_holiday"].round(4).to_string(index=False))
    print(extra["fold"].round(4).to_string(index=False))
    plot_slices(slices, out)

    rob = robustness(out, config["seeds"])
    for name, t in rob.items():
        print(name, "\n", t.round(4).to_string())

    if not args.no_mlflow:
        setup_mlflow(config)
        # повторный анализ без переобучения: прежний запуск анализа помечаем удаленным
        run_name = f"analysis/seed{main_seed}"
        old = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'")
        for run_id in old.get("run_id", []):
            mlflow.delete_run(run_id)
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(base_tags(config, main_seed, "error_analysis_and_slices"))
            mlflow.log_params(
                {
                    "seed": main_seed,
                    "hist_days": HIST_DAYS,
                    "spike_ratio": SPIKE_RATIO,
                    "spike_min": SPIKE_MIN,
                    "n_bootstrap": n_boot,
                }
            )
            mlflow.log_metrics(
                {
                    "rmsle_all_folds": float(np.sqrt(p["se"].mean())),
                    "spike_share_of_sq_error": float(spikes["share_of_sq_error"]),
                    "school_share_pichincha_tungurahua_f3": float(school_share[3]),
                    "promo_gaps_in_val": float(len(gaps)),
                }
            )
            for name in sorted(os.listdir(out)):
                if name.endswith(".csv"):
                    mlflow.log_artifact(f"{out}/{name}", artifact_path="tables")
            mlflow.log_artifacts(f"{out}/plots", artifact_path="plots")
            mlflow.log_artifact(args.config)


if __name__ == "__main__":
    main()
