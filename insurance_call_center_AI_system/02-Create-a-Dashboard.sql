-- Databricks notebook source
-- MAGIC %md
-- MAGIC # This Notebook will Perform
-- MAGIC
-- MAGIC 1. Create a Table Called `call_center_transcripts_analysis` from the DLT materialized view `call_center_transcripts_analysis_gold`
-- MAGIC 2. Show how you can build a databricks dashboard on top of the table

-- COMMAND ----------

CREATE OR REPLACE TABLE fins_genai.call_center.call_center_transcripts_analysis
AS SELECT distinct * FROM fins_genai.call_center.call_center_transcripts_analysis_gold;

-- COMMAND ----------

select
  *
from fins_genai.call_center.call_center_transcripts_analysis

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Now you can build a dashboard based on `call_center_transcripts_analysis`
-- MAGIC
-- MAGIC To build a dashboard, please review [Dashboard Document](https://docs.databricks.com/aws/en/dashboards/)
-- MAGIC
-- MAGIC Check out the example dashboard (./call_center_transcripts_dashboard.lvdash.json)
