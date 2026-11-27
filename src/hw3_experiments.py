"""ДЗ №3: переобучение E2_calendar на 4 фолдах и две проверки устойчивости.

1. Доля train по времени (train_fractions): validation и фолды те же, меняется только начало train.
2. Гауссов шум в лагах и скользящих средних validation (noise_sigmas): модель обучена на чистых
   данных (доля train 1.0) и не переобучается.

При доле 1.0 модель совпадает с E2_calendar из ДЗ №2 (тот же конфиг, deterministic: true);
скрипт сверяет RMSLE по фолдам с outputs/hw2/E2_calendar/seed<N>/results_by_fold.csv.
Прогнозы всех 4 фолдов при доле 1.0 сохраняются для анализа ошибок (hw3_analysis.py).

Пишет outputs/hw3/robustness_*.csv, outputs/hw3/predictions/, outputs/hw3/models/ (seed 42),
запуски в MLflow.

Запуск: python src/hw3_experiments.py --config configs/hw3/hw3.yaml
Смоук (в отдельный каталог, полные результаты не трогает):
    python src/hw3_experiments.py --seeds 42 --fractions 1.0 0.2 --sigmas 0 0.1 --folds 3 \
        --output-dir outputs/hw3_smoke --no-mlflow
"""

import argparse
import os
import time

import mlflow
import numpy as np
import pandas as pd

from hw3_common import (
    PRED_COLUMNS,
    base_tags,
    fold_bounds,
    read_config,
    setup_mlflow,
    train_start_for_fraction,
)
from train import train_lgbm
from utils import add_features, feature_list, load_data, rmsle

NOISE_PREFIXES = ("lag_", "roll_mean_")


def add_noise(val, features, sigma, rng):
    """Шум N(0, sigma) в лаги и скользящие средние (они в log1p), снизу обрезаем нулем."""
    noisy = val.copy()
    for col in features:
        if col.startswith(NOISE_PREFIXES):
            noisy[col] = np.clip(noisy[col] + rng.normal(0, sigma, len(noisy)), 0, None)
    return noisy


def check_against_hw2(config, seed, rows):
    """Доля 1.0 должна повторить ДЗ №2: та же модель и те же фолды."""
    path = f"outputs/hw2/{config['experiment_id']}/seed{seed}/results_by_fold.csv"
    if not os.path.exists(path):
        print(f"нет {path}, сверку с ДЗ №2 пропускаю")
        return None
    hw2 = pd.read_csv(path)
    hw2 = hw2[hw2["model"] == "lightgbm"].set_index("fold")["rmsle"]
    # results_by_fold.csv ДЗ №2 хранит RMSLE без округления - сравниваем на полной точности
    max_diff = max(abs(r["rmsle"] - hw2.loc[r["fold"]]) for r in rows)
    print(f"seed {seed}: сверка с ДЗ №2, max |diff RMSLE| по фолдам = {max_diff:.2e}")
    return max_diff


def log_run(config, seed, check, run_name, params, fold_rows, extra_metrics=None):
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(base_tags(config, seed, check))
        mlflow.log_params(
            {
                "seed": seed,
                "extra_features": ",".join(config.get("extra_features", [])),
                "lags": config["lags"],
                "roll_windows": config["roll_windows"],
                "num_boost_round": config["num_boost_round"],
                "base_train_start": config["train_start"],
                **config["lgbm_params"],
                **params,
            }
        )
        for r in fold_rows:
            mlflow.log_metric("rmsle", r["rmsle"], step=int(r["fold"]))
            mlflow.log_metric("mae", r["mae"], step=int(r["fold"]))
        mlflow.log_metrics(
            {
                "rmsle_mean": float(np.mean([r["rmsle"] for r in fold_rows])),
                "mae_mean": float(np.mean([r["mae"] for r in fold_rows])),
                **(extra_metrics or {}),
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hw3/hw3.yaml")
    parser.add_argument("--seeds", type=int, nargs="*", help="переопределить seeds (смоук)")
    parser.add_argument("--fractions", type=float, nargs="*", help="переопределить доли train")
    parser.add_argument("--sigmas", type=float, nargs="*", help="переопределить уровни шума")
    parser.add_argument("--folds", type=int, nargs="*", help="считать только эти фолды (смоук)")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--output-dir", help="другой каталог результатов (смоук), MLflow тоже пишется туда"
    )
    args = parser.parse_args()
    config = read_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
        config["mlflow"] = dict(
            config["mlflow"],
            tracking_uri=f"sqlite:///{args.output_dir}/mlflow.db",
            artifact_dir=f"{args.output_dir}/mlartifacts",
        )
    seeds = args.seeds or config["seeds"]
    fractions = args.fractions or config["train_fractions"]
    sigmas = args.sigmas if args.sigmas is not None else config["noise_sigmas"]
    out = config["output_dir"]
    os.makedirs(f"{out}/predictions", exist_ok=True)
    os.makedirs(f"{out}/models", exist_ok=True)
    if not args.no_mlflow:
        setup_mlflow(config)

    t0 = time.time()
    df = add_features(load_data(config["data_dir"]), config)
    features = feature_list(config)
    bounds = fold_bounds(df["date"].max(), config["horizon"], config["n_folds"])
    if args.folds:
        bounds = [b for b in bounds if b[0] in args.folds]
    print(f"данные и признаки: {time.time() - t0:.1f} с, признаков {len(features)}")

    frac_rows, noise_rows, hw2_check, family_rows = [], [], [], []
    for seed in seeds:
        run_config = dict(config, seed=seed)
        for fraction in fractions:
            t1 = time.time()
            rows, preds = [], []
            for fold, cutoff, val_end in bounds:
                start = train_start_for_fraction(config, cutoff, fraction)
                train = df[(df["date"] >= start) & (df["date"] < cutoff) & df["use_for_train"]]
                val = df[(df["date"] >= cutoff) & (df["date"] <= val_end)].reset_index(drop=True)
                pred, model = train_lgbm(train, val, features, run_config)
                sq_err = (np.log1p(pred) - np.log1p(val["sales"].values)) ** 2
                by_family = pd.Series(sq_err).groupby(val["family"].values).agg(["size", "sum"])
                family_rows += [
                    {
                        "level": fraction,
                        "seed": seed,
                        "fold": fold,
                        "family": fam,
                        "n": int(r["size"]),
                        "sq_log_err_sum": r["sum"],
                    }
                    for fam, r in by_family.iterrows()
                ]
                rows.append(
                    {
                        "check": "train_fraction",
                        "level": fraction,
                        "seed": seed,
                        "fold": fold,
                        "train_start": start.date(),
                        "train_days": (cutoff - start).days,
                        "train_rows": len(train),
                        "val_rows": len(val),
                        "rmsle": rmsle(val["sales"], pred),
                        "mae": float(np.mean(np.abs(val["sales"] - pred))),
                    }
                )
                if fraction == 1.0 and seed == config["main_seed"]:
                    # модели основного seed - для контрфактических проверок в hw3_analysis.py
                    model.save_model(f"{out}/models/E2_calendar_seed{seed}_fold{fold}.txt")
                if fraction == 1.0:
                    # проверка 2: та же модель, шум только во входных признаках validation
                    for sigma in sigmas:
                        rng = np.random.default_rng([config["noise_seed"], seed, fold])
                        noisy = add_noise(val, features, sigma, rng) if sigma > 0 else val
                        noisy_pred = np.clip(np.expm1(model.predict(noisy[features])), 0, None)
                        noise_rows.append(
                            {
                                "check": "input_noise",
                                "level": sigma,
                                "seed": seed,
                                "fold": fold,
                                "rmsle": rmsle(val["sales"], noisy_pred),
                                "mae": float(np.mean(np.abs(val["sales"] - noisy_pred))),
                            }
                        )
                    keep = val.assign(fold=fold)[PRED_COLUMNS].copy()
                    keep["pred"] = pred
                    preds.append(keep)
            frac_rows += rows
            mean = np.mean([r["rmsle"] for r in rows])
            print(f"seed {seed}, доля {fraction}: RMSLE {mean:.4f} ({time.time() - t1:.1f} с)")
            if fraction == 1.0:
                pd.concat(preds).to_csv(
                    f"{out}/predictions/E2_calendar_seed{seed}.csv.gz", index=False
                )
                if len(rows) == config["n_folds"]:
                    hw2_check.append(
                        {"seed": seed, "max_abs_diff": check_against_hw2(config, seed, rows)}
                    )
            if not args.no_mlflow:
                log_run(
                    config,
                    seed,
                    "train_fraction",
                    f"train_fraction/{fraction:.2f}/seed{seed}",
                    {
                        "train_fraction": fraction,
                        "train_days_by_fold": [r["train_days"] for r in rows],
                    },
                    rows,
                )
        if not args.no_mlflow:
            for sigma in sigmas:
                rows = [r for r in noise_rows if r["seed"] == seed and r["level"] == sigma]
                log_run(
                    config,
                    seed,
                    "input_noise",
                    f"input_noise/{sigma:.2f}/seed{seed}",
                    {
                        "noise_sigma": sigma,
                        "noise_seed": config["noise_seed"],
                        "train_fraction": 1.0,
                    },
                    rows,
                )

    pd.DataFrame(frac_rows).to_csv(f"{out}/robustness_train_fraction_by_fold.csv", index=False)
    pd.DataFrame(family_rows).to_csv(f"{out}/robustness_train_fraction_by_family.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(f"{out}/robustness_noise_by_fold.csv", index=False)
    if hw2_check:
        pd.DataFrame(hw2_check).to_csv(f"{out}/check_vs_hw2.csv", index=False)
    print(f"всего: {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
