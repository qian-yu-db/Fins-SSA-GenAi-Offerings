-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC # This delta live table will perform
-- MAGIC
-- MAGIC * Perform cleaning and transformation of raw transcript table 
-- MAGIC * Perform cleaning and transformation of raw policy table
-- MAGIC * Join the transcripts and policy tables with policy number
-- MAGIC * Perform NLP Analysis with Databricks AI Functions
-- MAGIC
-- MAGIC ###AI Functions
-- MAGIC
-- MAGIC [Databricks AI Functions](https://docs.databricks.com/aws/en/large-language-models/ai-functions) are built-in SQL functions that allow you to apply AI on your data directly from SQL. They are automatically provisioned scaled to offer 10x faster job completion

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Bronze: 
-- MAGIC
-- MAGIC * `raw_polcy`: Use Auto Loader to ingest the raw policy data csv files
-- MAGIC * `raw_transcript`: Use Auto Loader to ingest the raw transcript data json files

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE raw_policies
COMMENT "Policy data loaded from csv files."
AS 
SELECT * FROM cloud_files(
    "/Volumes/fins_genai/call_center/volume_policies/policies/",
    'json',
    map("mode", 'PERMISSIVE')
)

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE raw_transcripts
COMMENT "Transcript data loaded from json files."
AS 
SELECT * FROM cloud_files(
    "/Volumes/fins_genai/call_center/volume_transcripts/transcripts_json_data/",
    'json',
    map("mode", 'PERMISSIVE')
)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Silver: 
-- MAGIC
-- MAGIC * `policy`: Perform data cleaning and data type changes
-- MAGIC * `transcript`: Perform data cleaning, filtering

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE policies_cleaned(
  CONSTRAINT valid_policy_number EXPECT (policy_number IS NOT NULL)
)
COMMENT "clean raw policies data table."
AS SELECT
  POLICYTYPE as policy_type,
  MAKE as make,
  MODEL as model,
  MODEL_YEAR as model_year,
  CHASSIS_NO as chassis_no,
  USE_OF_VEHICLE as use_of_vehicle,
  DRV_DOB as driver_dob,
  SUM_INSURED as sum_insured,
  premium,
  DEDUCTABLE as deductable,
  first_name,
  last_name,
  email,
  to_date(pol_eff_date, 'yyyy-MM-dd') as effective_date,
  to_date(pol_expiry_date, 'yyyy-MM-dd') as expiration_date,
  to_date(pol_issue_date, 'yyyy-MM-dd') as issue_date,
  policy_number,
  phone_number,
  address
FROM stream(live.raw_policies);

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE transcripts_cleaned(
  CONSTRAINT valid_phone_number EXPECT (phone_number IS NOT NULL)
)
COMMENT "clean raw transcript data table."
AS SELECT
  phone_number,
  call_timestamp,
  transcript
from stream(live.raw_transcripts);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Gold: 
-- MAGIC
-- MAGIC * `transcripts_gold`
-- MAGIC   * Aggregate transcripts by policy_no so it is in one conversation
-- MAGIC   * Join policy using `policy_number` to enrich meta-data
-- MAGIC   * Perform NLP analysis with **Databricks AI Functions**
-- MAGIC     * `ai_analyze_sentiment()`
-- MAGIC     * `ai_summarize()`
-- MAGIC     * `ai_classify()` 

-- COMMAND ----------

CREATE OR REFRESH MATERIALIZED VIEW call_center_transcripts_analysis_gold
COMMENT "createn enriched transcript table with meta data from policy"
AS 
SELECT
  p.policy_number,
  p.policy_type,
  p.issue_date,
  p.effective_date,
  p.expiration_date,
  p.make,
  p.model,
  p.model_year,
  p.driver_dob,
  p.chassis_no,
  p.use_of_vehicle,
  p.sum_insured,
  p.premium,
  p.deductable,
  p.address,
  p.first_name,
  p.last_name,
  p.email,
  p.phone_number,
  to_timestamp(t.call_timestamp, 'yyyy-MM-dd\'T\'HH:mm:ss.SSS\'Z\'') as call_timestamp,  
  ai_summarize(
    t.transcript,
    300
  ) as summary,
  ai_analyze_sentiment(
    t.transcript
  ) as sentiment,
  ai_classify(
    t.transcript,
    ARRAY("car accident", "home accident", "motocycle related", "theft", "policy related")
  ) as intent,
  t.transcript
FROM live.policies_cleaned p
INNER JOIN live.transcripts_cleaned t ON p.phone_number = t.phone_number;
