# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Baseline Whole-Image Classifier
# MAGIC
# MAGIC Sample-mode baseline orchestration. Shared training, threshold selection,
# MAGIC metrics, and MLflow payload construction live in `src/levers/` and
# MAGIC `scripts/train_baseline.py`.

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

from levers.baseline import run_sample_baseline
from levers.config import ProjectConfig
from levers.mlflow_utils import flatten_metrics
from scripts.train_baseline import build_runtime_params, maybe_log_mlflow

# COMMAND ----------

sample_mode = _as_bool(_widget("sample_mode", os.environ.get("SAMPLE_MODE", "true")))
runtime = _widget("runtime", os.environ.get("CV_RUNTIME", "databricks_serverless_cpu"))
log_mlflow = _as_bool(_widget("log_mlflow", "true"))
min_recall = float(_widget("min_recall", "0.75"))
experiment_id = _widget("experiment_id", os.environ.get("MLFLOW_EXPERIMENT_ID", ""))
experiment_name = _widget("experiment_name", os.environ.get("MLFLOW_EXPERIMENT_NAME", ""))

config = ProjectConfig.from_env()
result = run_sample_baseline(sample_mode=sample_mode, min_recall=min_recall)
metrics = flatten_metrics(result.metric_payload())
metrics.update(flatten_metrics(result.validation_metric_payload(), prefix="val"))
params = build_runtime_params(
    config=config,
    result=result,
    runtime=runtime,
    sample_mode=sample_mode,
    min_recall=min_recall,
    split_seed=1,
    feature_seed=17,
)

maybe_log_mlflow(
    args=SimpleNamespace(
        log_mlflow=log_mlflow,
        tracking_uri=config.mlflow_tracking_uri,
        experiment_id=experiment_id or None,
        experiment_name=experiment_name or None,
        run_name="baseline_whole_image_notebook",
    ),
    config=config,
    result=result,
    metrics=metrics,
    params=params,
)

print("baseline_metrics")
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
