# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Recommendation Summary
# MAGIC
# MAGIC Aggregate MLflow runs into a levers leaderboard with recall, precision,
# MAGIC false negatives, effort, and recommended next action.

# COMMAND ----------

dbutils.widgets.text("experiment_name", "/Shared/cv-accuracy-levers")

experiment_name = dbutils.widgets.get("experiment_name")
print({"experiment_name": experiment_name})

# COMMAND ----------

# TODO: Query MLflow runs and render the comparison table.

