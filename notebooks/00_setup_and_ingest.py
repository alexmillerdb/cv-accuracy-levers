# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Setup and Ingest
# MAGIC
# MAGIC Public-data ingest notebook for the CV accuracy levers demo.
# MAGIC Start with sample mode on serverless CPU before full-dataset work.

# COMMAND ----------

dbutils.widgets.text("sample_mode", "true")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "cv_accuracy_levers")
dbutils.widgets.text("volume", "cv_accuracy_levers")
dbutils.widgets.text("volume_subpath", "artifacts")
dbutils.widgets.text("experiment_name", "/Shared/cv-accuracy-levers")

sample_mode = dbutils.widgets.get("sample_mode").lower() == "true"
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
volume_subpath = dbutils.widgets.get("volume_subpath")
experiment_name = dbutils.widgets.get("experiment_name")

print(
    {
        "sample_mode": sample_mode,
        "catalog": catalog,
        "schema": schema,
        "volume": volume,
        "volume_subpath": volume_subpath,
        "experiment_name": experiment_name,
    }
)

# COMMAND ----------

# TODO: Download and register the open wood-defect dataset.
# Keep this cell serverless-CPU compatible in sample mode.
# Full image download and GPU-specific preprocessing belong behind explicit
# parameters so smoke tests remain cheap.
