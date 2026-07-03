# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Crop-First A/B Test
# MAGIC
# MAGIC Compare whole-image classification with crop-first classification on the
# MAGIC same split and metrics.

# COMMAND ----------

dbutils.widgets.text("sample_mode", "true")

sample_mode = dbutils.widgets.get("sample_mode").lower() == "true"
print({"sample_mode": sample_mode})

# COMMAND ----------

# TODO: Create crop cache, run visual QA, then rerun the baseline classifier on
# crops. GPU-only pieces must be clearly parameter gated.

