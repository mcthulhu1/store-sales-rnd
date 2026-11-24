#!/usr/bin/env bash
# ДЗ №2: все эксперименты из configs/hw2 (обучение + анализ ошибок), затем сводная таблица
set -e
cd "$(dirname "$0")/.."
for config in configs/hw2/*.yaml; do
    echo "=== $config"
    python src/train.py --config "$config"
    python src/evaluate.py --config "$config"
done
python src/compare.py
