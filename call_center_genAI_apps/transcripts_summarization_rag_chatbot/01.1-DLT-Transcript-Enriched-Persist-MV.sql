-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC # Persist DLT Materialized View
-- MAGIC
-- MAGIC To easily support DLT / UC / ML during the preview, we temporary recopy the final DLT view to another UC table
-- MAGIC * Currently DLT Materialized View can only be read on shared clusters

-- COMMAND ----------

CREATE OR REPLACE TABLE fins_genai.speech.transcript_enriched_ml 
AS SELECT * FROM fins_genai.speech.transcript_enriched;
