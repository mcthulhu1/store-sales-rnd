#!/usr/bin/env bash
# Полный прогон: EDA, обучение и валидация (+ MLflow), анализ ошибок
set -e
cd "$(dirname "$0")/.."
CONFIG=${1:-configs/baseline.yaml}
python src/eda.py --config "$CONFIG"
python src/train.py --config "$CONFIG"
python src/evaluate.py --config "$CONFIG"
