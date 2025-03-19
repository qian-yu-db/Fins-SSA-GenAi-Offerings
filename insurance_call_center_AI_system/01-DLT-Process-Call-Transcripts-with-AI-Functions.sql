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
-- MAGIC * Perform data cleaning and data type changes for `raw_policies` and `raw_transcripts` table
-- MAGIC * Join these 2 tables to create `call_center_transcripts_cleaned` table with all key columns

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE call_center_transcripts_cleaned(
  CONSTRAINT valid_records EXPECT (p.policy_number IS NOT NULL and t.phone_number IS NOT NULL)
)
COMMENT "clean and join raw policies data table and raw transcript table"
AS 
SELECT
  p.POLICYTYPE as policy_type,
  p.MAKE as make,
  p.MODEL as model,
  p.MODEL_YEAR as model_year,
  p.CHASSIS_NO as chassis_no,
  p.USE_OF_VEHICLE as use_of_vehicle,
  p.DRV_DOB as driver_dob,
  p.SUM_INSURED as sum_insured,
  p.premium,
  p.DEDUCTABLE as deductable,
  p.first_name,
  p.last_name,
  p.email,
  to_date(p.pol_eff_date, 'yyyy-MM-dd') as effective_date,
  to_date(p.pol_expiry_date, 'yyyy-MM-dd') as expiration_date,
  to_date(p.pol_issue_date, 'yyyy-MM-dd') as issue_date,
  p.policy_number,
  p.address,
  t.operator_id,
  t.call_timestamp,
  t.phone_number,
  t.transcript
FROM stream(live.raw_policies) p
inner join stream(live.raw_transcripts) t on p.phone_number = t.phone_number

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
  operator_id,
  policy_number,
  policy_type,
  issue_date,
  effective_date,
  expiration_date,
  make,
  model,
  model_year,
  driver_dob,
  chassis_no,
  use_of_vehicle,
  sum_insured,
  premium,
  deductable,
  address,
  first_name,
  last_name,
  email,
  phone_number,
  to_timestamp(call_timestamp, 'yyyy-MM-dd\'T\'HH:mm:ss.SSS\'Z\'') as call_timestamp,  
  ai_summarize(
    transcript,
    300
  ) as summary,
  ai_analyze_sentiment(
    transcript
  ) as sentiment,
  ai_query(
      "databricks-meta-llama-3-3-70b-instruct",
      "extract the customer intent (of either 'auto accident', 'home accident', 'motocycle', or 'policy related') and
      key context of the intent from the transcript:" || transcript,
      responseFormat => 'STRUCT<intent_analysis:STRUCT<intent:STRING, context:STRING>>',
      failOnError => false
  ) as intent_analysis,
  ai_query(
      "databricks-meta-llama-3-3-70b-instruct",
      concat("You are an expert in insurance operation, flag any mistconduct: (in the catagory of 'rudeness', 'discrimination', 'data overreach')",
      " for the following transcript: <transcript>", transcript, "<transcript>\n",
      "Answer with the catagory of the mistconduct, do not explain. ", 
      "If there is no mistconduct, reply with 'none', and answer a flag as a boolean"),
      responseFormat => 'STRUCT<misconduct_analysis:STRUCT<catagory:ARRAY<STRING>, flag:BOOLEAN>>',
      failOnError => false
    ) as misconduct_analysis,
  ai_query(
      "databricks-meta-llama-3-3-70b-instruct",
      concat("You are an expert in customer relationship management, analysis the following transcript: ", "<transcript>", transcript, "<transcript>\n",
      "give a recommendation of the next best action from the following list of choices: ", 
      "'follow-up call', 'promotional email', 'automated email', 'apology email'",
      "Answer with one recommended action, do not explain. If no action is required, return 'none'."
      "Answer with a succinct justificaction of the recommendation in less than 5 words "),
      responseFormat => 'STRUCT<recommendation:STRUCT<catagory:STRING, justification:STRING>>',
      failOnError => false
    ) as next_best_action,
  transcript
FROM live.call_center_transcripts_cleaned;
