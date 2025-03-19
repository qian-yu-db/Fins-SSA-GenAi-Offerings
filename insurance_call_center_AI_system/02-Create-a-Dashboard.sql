-- Databricks notebook source
-- MAGIC %md
-- MAGIC # This Notebook will Perform
-- MAGIC
-- MAGIC 1. Create a Table Called `call_center_transcripts_analysis` from the DLT materialized view `call_center_transcripts_analysis_gold`
-- MAGIC 2. Show how you can build a databricks dashboard on top of the table

-- COMMAND ----------

CREATE OR REPLACE TABLE fins_genai.call_center.call_center_transcripts_analysis
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
    call_timestamp,
    summary,
    sentiment,
    intent_result:intent as intent,
    intent_result:context as intent_context,
    misconduct_result:catagory as misconduct_catagory,
    misconduct_result:flag as misconduct_flag,
    next_best_action_result:catagory as action_recommendation,
    next_best_action_result:justification as action_recommendation_justification,
    transcript
from
(
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
        call_timestamp,
        summary,
        sentiment,
        intent_analysis.result AS intent_result,
        intent_analysis.errorMessage AS intent_error_message,
        misconduct_analysis.result AS misconduct_result,
        misconduct_analysis.errorMessage AS misconduct_error_message,
        next_best_action.result AS next_best_action_result,
        next_best_action.errorMessage AS next_best_action_error_message,
        transcript
    FROM fins_genai.call_center.call_center_transcripts_analysis_gold
) where intent_error_message is null and misconduct_error_message is null and next_best_action_error_message is null

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
