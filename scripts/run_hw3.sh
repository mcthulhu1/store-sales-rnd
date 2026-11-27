#!/usr/bin/env bash
# ДЗ №3: переобучение E2_calendar (3 seed, 4 фолда), проверки устойчивости, анализ ошибок и срезы.
# База MLflow ДЗ №3 (outputs/hw3/mlflow.db) пересоздается, чтобы повторный запуск не дублировал запуски.
set -e
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
CONFIG=${1:-configs/hw3/hw3.yaml}
# пути базы и артефактов MLflow берутся из выбранного конфига
PATHS=$(python - "$CONFIG" <<'PY'
import sys
sys.path.insert(0, "src")
from utils import load_config
m = load_config(sys.argv[1])["mlflow"]
print(m["tracking_uri"].removeprefix("sqlite:///"), m["artifact_dir"])
PY
)
read -r MLFLOW_DB MLFLOW_ART <<< "$PATHS"
rm -rf "$MLFLOW_DB" "$MLFLOW_ART"
python src/hw3_experiments.py --config "$CONFIG"
python src/hw3_analysis.py --config "$CONFIG"
