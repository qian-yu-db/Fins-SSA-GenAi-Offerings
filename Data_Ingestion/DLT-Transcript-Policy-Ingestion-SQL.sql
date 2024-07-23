-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC # This delta live table will perform
-- MAGIC
-- MAGIC * Perform cleaning and transformation of raw transcript table 
-- MAGIC * Perform cleaning and transformation of raw policy table
-- MAGIC * Join the transcripts and policy tables with policy number

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Bronze: 
-- MAGIC
-- MAGIC * `raw_polcy`: Use Auto Loader to ingest the raw policy data csv files
-- MAGIC * `raw_transcript`: Use Auto Loader to ingest the raw transcript data json files

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE raw_policy_sql
COMMENT "Policy data loaded from csv files."
AS 
SELECT * FROM cloud_files(
    "/Volumes/fins_genai/speech/volume_policies/policies_name_email/",
    'csv',
    map("mode", 'PERMISSIVE')
)

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE raw_transcript_sql
COMMENT "Transcript data loaded from json files."
AS 
SELECT * FROM cloud_files(
    "/Volumes/fins_genai/speech/volume_speech/transcripts_json_data/",
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

CREATE OR REFRESH STREAMING TABLE policy_v3_sql(
  CONSTRAINT valid_policy_number EXPECT (policy_number IS NOT NULL)
)
COMMENT "clean raw policy data table."
AS SELECT
  CUST_ID,
  POLICYTYPE,
  BODY,
  MAKE,
  MODEL,
  MODEL_YEAR,
  CHASSIS_NO,
  USE_OF_VEHICLE,
  DRV_DOB,
  NEIGHBORHOOD,
  PRODUCT,
  SUM_INSURED,
  DEDUCTABLE,
  id,
  first_name,
  last_name,
  email,
  abs(premium) as premium,
  to_date(POL_EFF_DATE, 'dd-MM-yyyy') as pol_eff_date,
  to_date(POL_EXPIRY_DATE, 'dd-MM-yyyy') as pol_expiry_date,
  to_date(POL_ISSUE_DATE, 'dd-MM-yyyy') as pol_issue_date,
  POLICY_NO as policy_number,
  concat(BOROUGH, ', ', cast(ZIP_CODE as string)) as address
FROM stream(live.raw_policy_sql);

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE trascript_v3_sql(
  CONSTRAINT valid_policy_number EXPECT (policy_number IS NOT NULL)
)
COMMENT "clean raw transcript data table."
AS SELECT
  POLICY_NO as policy_number,
  conversation,
  datetime_record
from stream(live.raw_transcript_sql);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Gold: 
-- MAGIC
-- MAGIC * `transcript_enriched`
-- MAGIC   * Aggregate transcripts by policy_no so it is in one conversation
-- MAGIC   * Join policy using `policy_number` to enrich meta-data

-- COMMAND ----------

CREATE OR REFRESH MATERIALIZED VIEW transcript_enriched_v3_sql
COMMENT "createn enriched transcript table with meta data from policy"
AS SELECT
  p.policy_number,
  p.CUST_ID,
  p.POLICYTYPE,
  p.pol_issue_date,
  p.pol_eff_date,
  p.pol_expiry_date,
  p.MAKE,
  p.MODEL,
  p.MODEL_YEAR,
  p.DRV_DOB,
  p.address,
  p.first_name,
  p.last_name,
  p.email,
  t.conversation as transcript,
  t.datetime_record
FROM live.policy_v3_sql p
INNER JOIN live.trascript_v3_sql t ON p.policy_number = t.policy_number;
