# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Threshold Tuning
# MAGIC
# MAGIC Sample-mode threshold tuning over baseline prediction scores. Shared
# MAGIC threshold selection, fixed-0.5 comparison, metrics, and MLflow payload
# MAGIC construction live in `src/levers/` and `scripts/tune_threshold.py`.

# COMMAND ----------

import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _add_repo_paths() -> None:
    candidates = [Path.cwd(), Path.cwd().parent]
    for candidate in candidates:
        src_dir = candidate / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        if (candidate / "scripts").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _widget(name: str, default: str) -> str:
    try:
        dbutils.widgets.text(name, default)
        return dbutils.widgets.get(name)
    except NameError:
        return os.environ.get(name.upper(), default)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


_add_repo_paths()

from levers.config import ProjectConfig
from levers.threshold_tuning import run_sample_threshold_tuning
from scripts.tune_threshold import (
    build_metric_payload,
    build_runtime_params,
    maybe_log_mlflow,
)

# COMMAND ----------

sample_mode = _as_bool(_widget("sample_mode", os.environ.get("SAMPLE_MODE", "true")))
runtime = _widget("runtime", os.environ.get("CV_RUNTIME", "databricks_serverless_cpu"))
log_mlflow = _as_bool(_widget("log_mlflow", "true"))
min_recall = float(_widget("min_recall", "0.75"))
baseline_threshold = float(_widget("baseline_threshold", "0.5"))
split_seed = int(_widget("split_seed", "1"))
feature_seed = int(_widget("feature_seed", "17"))
experiment_id = _widget("experiment_id", os.environ.get("MLFLOW_EXPERIMENT_ID", ""))
experiment_name = _widget("experiment_name", os.environ.get("MLFLOW_EXPERIMENT_NAME", ""))

config = ProjectConfig.from_env()
result = run_sample_threshold_tuning(
    sample_mode=sample_mode,
    split_seed=split_seed,
    feature_seed=feature_seed,
    min_recall=min_recall,
    baseline_threshold=baseline_threshold,
)
metrics = build_metric_payload(result)
params = build_runtime_params(
    config=config,
    result=result,
    runtime=runtime,
    sample_mode=sample_mode,
    min_recall=min_recall,
    split_seed=split_seed,
    feature_seed=feature_seed,
)

maybe_log_mlflow(
    args=SimpleNamespace(
        log_mlflow=log_mlflow,
        tracking_uri=config.mlflow_tracking_uri,
        experiment_id=experiment_id or None,
        experiment_name=experiment_name or None,
        run_name="threshold_tuning_notebook",
    ),
    config=config,
    result=result,
    metrics=metrics,
    params=params,
)

print("threshold_tuning_metrics")
for key, value in metrics.items():
    print(f"{key}={value}")

# COMMAND ----------

display(
    [
        {
            "image_path": prediction.image_path,
            "group_id": prediction.group_id,
            "split": prediction.split,
            "label": prediction.label,
            "score": prediction.score,
        }
        for prediction in result.predictions
    ]
)

# COMMAND ----------

display(result.validation_sweep_payload())
