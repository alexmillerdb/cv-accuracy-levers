# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Label Quality Embeddings
# MAGIC
# MAGIC Sample-mode label-quality review over deterministic baseline synthetic
# MAGIC feature embeddings. Shared neighbor ranking, review candidates, metrics,
# MAGIC and MLflow payload construction live in `src/levers/` and
# MAGIC `scripts/review_label_quality_embeddings.py`.

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
from levers.label_quality import run_sample_label_quality_embeddings
from scripts.review_label_quality_embeddings import (
    build_metric_payload,
    build_runtime_params,
    maybe_log_mlflow,
)

# COMMAND ----------

sample_mode = _as_bool(_widget("sample_mode", os.environ.get("SAMPLE_MODE", "true")))
runtime = _widget("runtime", os.environ.get("CV_RUNTIME", "databricks_serverless_cpu"))
log_mlflow = _as_bool(_widget("log_mlflow", "true"))
min_recall = float(_widget("min_recall", "0.75"))
split_seed = int(_widget("split_seed", "1"))
feature_seed = int(_widget("feature_seed", "17"))
label_conflict_similarity_threshold = float(
    _widget("label_conflict_similarity_threshold", "0.98")
)
cross_split_similarity_threshold = float(
    _widget("cross_split_similarity_threshold", "0.999")
)
neighbor_count = int(_widget("neighbor_count", "5"))
top_k = int(_widget("top_k", "20"))
inject_synthetic_label_issue = _as_bool(
    _widget("inject_synthetic_label_issue", "false")
)
experiment_id = _widget("experiment_id", os.environ.get("MLFLOW_EXPERIMENT_ID", ""))
experiment_name = _widget("experiment_name", os.environ.get("MLFLOW_EXPERIMENT_NAME", ""))

config = ProjectConfig.from_env()
result = run_sample_label_quality_embeddings(
    sample_mode=sample_mode,
    split_seed=split_seed,
    feature_seed=feature_seed,
    min_recall=min_recall,
    label_conflict_similarity_threshold=label_conflict_similarity_threshold,
    cross_split_similarity_threshold=cross_split_similarity_threshold,
    neighbor_count=neighbor_count,
    top_k=top_k,
    inject_synthetic_label_issue=inject_synthetic_label_issue,
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
        run_name="label_quality_embeddings_notebook",
    ),
    config=config,
    result=result,
    metrics=metrics,
    params=params,
)

print("label_quality_metrics")
for key, value in metrics.items():
    print(f"{key}={value}")

# COMMAND ----------

display(result.review_rows_payload())

# COMMAND ----------

display(result.summary_payload())

# COMMAND ----------

display(result.neighbor_rows_payload())
