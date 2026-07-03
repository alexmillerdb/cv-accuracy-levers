# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Error Review Gut Check
# MAGIC
# MAGIC Review high-confidence false negatives and bucket likely failure modes
# MAGIC before adding heavier model changes.

# COMMAND ----------

dbutils.widgets.text("sample_mode", "true")

sample_mode = dbutils.widgets.get("sample_mode").lower() == "true"
print({"sample_mode": sample_mode})

# COMMAND ----------

# TODO: Render a review table/grid for high-confidence Good predictions that
# are labeled defective, then bucket as crop/background issue, invisible defect,
# or possible label issue.

