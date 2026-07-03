# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Threshold Tuning
# MAGIC
# MAGIC Recall is an operating point. This notebook should consume baseline
# MAGIC validation predictions and produce threshold/precision/recall tradeoffs.

# COMMAND ----------

dbutils.widgets.text("sample_mode", "true")
dbutils.widgets.text("experiment_name", "/Shared/cv-accuracy-levers")

sample_mode = dbutils.widgets.get("sample_mode").lower() == "true"
experiment_name = dbutils.widgets.get("experiment_name")

print({"sample_mode": sample_mode, "experiment_name": experiment_name})

# COMMAND ----------

# TODO: Use levers.thresholds.threshold_sweep on validation predictions and log
# the selected operating point to MLflow.

