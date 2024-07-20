-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC # Apply SQL AI Query to Perform Sentiment Analysis, Summarization, and topic classification

-- COMMAND ----------

CREATE OR REPLACE TABLE fins_genai.speech.transcript_enriched_ml_v3_sql 
AS SELECT distinct * FROM fins_genai.speech.transcript_enriched_v3_sql;

-- COMMAND ----------

CREATE OR REPLACE TABLE fins_genai.customer_information.customer_information_interaction
AS
select
*
from
(
  select
    policy_number,
    first_name,
    last_name,
    DRV_DOB,
    address,
    email,
    pol_expiry_date,
    MAKE,
    MODEL,
    ai_summarize
    (
      transcript
      , 150
    ) as summary,
    ai_analyze_sentiment
    (
      transcript
    ) as sentiment,
    ai_classify
    (
      transcript,
      ARRAY("car accident", "policy change", "motocycle related", "home accident", "general question", "theft")
    ) as topics
    from fins_genai.speech.transcript_enriched_ml_v3_sql
)
where topics is not null and sentiment is not null and summary is not null

-- COMMAND ----------


