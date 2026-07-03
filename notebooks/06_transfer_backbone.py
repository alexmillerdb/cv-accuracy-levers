# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Transfer Backbone
# MAGIC
# MAGIC Compare baseline ResNet with stronger frozen features and a light
# MAGIC classifier, using the same split and operating-point reporting.

# COMMAND ----------

dbutils.widgets.text("sample_mode", "true")

sample_mode = dbutils.widgets.get("sample_mode").lower() == "true"
print({"sample_mode": sample_mode})

# COMMAND ----------

# TODO: Cache frozen features and train/evaluate a lightweight classifier.

