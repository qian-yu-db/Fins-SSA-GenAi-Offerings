-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC # Persist DLT Materialized View
-- MAGIC
-- MAGIC To easily support DLT / UC / ML during the preview, we temporary recopy the final DLT view to another UC table
-- MAGIC
-- MAGIC * Why: Currently Delta Live Table generated Materialized View can only be read on shared clusters
-- MAGIC * How: Please run this notebook with a shared cluster

-- COMMAND ----------

-- MAGIC %run ./config

-- COMMAND ----------

CREATE OR REPLACE TABLE transcript_enriched_ml 
AS SELECT * FROM transcript_enriched;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Next Step, we go to [Notebook 02-Prompt-Engineering]($./02-Prompt-Engineering)

-- COMMAND ----------


