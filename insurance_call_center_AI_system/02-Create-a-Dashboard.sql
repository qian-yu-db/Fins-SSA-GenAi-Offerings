-- Databricks notebook source
-- MAGIC %md
-- MAGIC # This Notebook will Perform
-- MAGIC
-- MAGIC 1. Create a Table Called `call_center_transcripts_analysis` from the DLT materialized view `call_center_transcripts_analysis_gold`
-- MAGIC 2. Show how you can build a databricks dashboard on top of the table

-- COMMAND ----------

USE CATALOG fins_genai;
USE SCHEMA call_center;

-- COMMAND ----------

CREATE OR REPLACE TABLE fins_genai.call_center.call_center_transcripts_analysis
AS 
with processed_analysis as (
  select
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
      call_timestamp,
      cast(call_duration_in_sec as INT) as call_duration_in_sec,
      call_time_limit,
      summary,
      sentiment,
      from_json(compliance_score, 'STRUCT<points:INT, justification:STRING>') as compliance_score,
      from_json(customer_intent, 'STRUCT<intent:STRING, context:STRING>') AS customer_intent,
      from_json(next_best_action, 'STRUCT<catagory:STRING, justification:STRING>') as next_best_action,
      customer_asks,
      transcript
  from call_center_transcripts_analysis_gold
),

processed_analysis_enrich as (
  select
    pa.*,
    vs.section as related_policy_doc_section
  from processed_analysis pa
  join
    lateral (
      select
        *
      from
        vector_search(
          index => "fins_genai.call_center.policy_docs_chunked_files_vs_index",
          query_text => pa.customer_asks,
          num_results => 1
        )
    ) vs
)

select
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
  call_timestamp,
  call_duration_in_sec,
  call_time_limit,
  summary,
  sentiment,
  compliance_score.points as compliance_score,
  compliance_score.justification as compliance_violations_justification,
  case when lower(compliance_violations_justification) like '%guideline #1%' then 1 else 0 end as no_inform_compliance,
  case when lower(compliance_violations_justification) like '%guideline #2%' then 1 else 0 end as unprofessionalism,
  case when lower(compliance_violations_justification) like '%guideline #3%' then 1 else 0 end as no_identity_verification,
  case when lower(compliance_violations_justification) like '%guideline #4%' then 1 else 0 end as no_detail_explaination,
  case when lower(compliance_violations_justification) like '%guideline #5%' then 1 else 0 end as no_inform_company_contact,
  customer_intent.intent as customer_intent,
  customer_intent.context as customer_intent_context,
  next_best_action.catagory as action_recommendation,
  next_best_action.justification as action_recommendation_justification,
  customer_asks,
  related_policy_doc_section,
  transcript
from processed_analysis_enrich

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
