# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Setup and Ingest
# MAGIC
# MAGIC Public-data ingest orchestration for the CV accuracy levers demo.
# MAGIC Shared ingest logic lives in `src/levers/ingest.py` and
# MAGIC `scripts/prepare_dataset.py`.

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

from scripts.prepare_dataset import run_prepare_dataset

# COMMAND ----------

source = _widget("source", os.environ.get("CV_DATA_SOURCE", "manifest"))
manifest_path = _widget("manifest_path", os.environ.get("CV_DATA_MANIFEST", ""))
data_dir = _widget("data_dir", os.environ.get("CV_DATA_DIR", ""))
output_path = _widget("output_path", "artifacts/ingest/normalized_manifest.jsonl")
sample_mode = _as_bool(_widget("sample_mode", os.environ.get("SAMPLE_MODE", "true")))
sample_size = int(_widget("sample_size", "100"))
split_seed = int(_widget("split_seed", "42"))
runtime = _widget("runtime", os.environ.get("CV_RUNTIME", "databricks_serverless_cpu"))
catalog = _widget("catalog", os.environ.get("CV_CATALOG", "main"))
schema = _widget("schema", os.environ.get("CV_SCHEMA", "cv_accuracy_levers"))
volume = _widget("volume", os.environ.get("CV_VOLUME", "cv_accuracy_levers"))
volume_subpath = _widget("volume_subpath", os.environ.get("CV_VOLUME_SUBPATH", "artifacts"))
positive_labels = _widget("positive_labels", "")
negative_labels = _widget("negative_labels", "")
allow_missing_images = _as_bool(_widget("allow_missing_images", "false"))
copy_images_to_uc_volume = _as_bool(_widget("copy_images_to_uc_volume", "false"))
uc_image_dir = _widget("uc_image_dir", "")
uc_image_upload_mode = _widget("uc_image_upload_mode", "local")
databricks_profile = _widget("databricks_profile", os.environ.get("DATABRICKS_CONFIG_PROFILE", ""))
write_uc = _as_bool(_widget("write_uc", "false"))
uc_table = _widget("uc_table", "image_manifest")
uc_manifest_uri = _widget("uc_manifest_uri", "")
uc_write_mode = _widget("uc_write_mode", "overwrite")
create_uc_objects = _as_bool(_widget("create_uc_objects", "false"))

args = SimpleNamespace(
    source=source,
    manifest_path=manifest_path or None,
    data_dir=data_dir or None,
    output_path=output_path,
    sample_mode=sample_mode,
    sample_size=sample_size,
    split_seed=split_seed,
    runtime=runtime,
    catalog=catalog,
    schema=schema,
    volume=volume,
    volume_subpath=volume_subpath,
    positive_labels=positive_labels or None,
    negative_labels=negative_labels or None,
    allow_missing_images=allow_missing_images,
    copy_images_to_uc_volume=copy_images_to_uc_volume,
    uc_image_dir=uc_image_dir or None,
    uc_image_upload_mode=uc_image_upload_mode,
    databricks_profile=databricks_profile or None,
    write_uc=write_uc,
    uc_table=uc_table,
    uc_manifest_uri=uc_manifest_uri or None,
    uc_write_mode=uc_write_mode,
    create_uc_objects=create_uc_objects,
)

result = run_prepare_dataset(args)
summary = result.summary()
summary["runtime"] = runtime

print("dataset_ingest_summary")
for key, value in summary.items():
    print(f"{key}={value}")

# COMMAND ----------

display([row.as_manifest_row() for row in result.records])
