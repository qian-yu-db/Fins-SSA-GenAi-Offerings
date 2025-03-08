# Databricks notebook source
# MAGIC %md 
# MAGIC # Turn the Review App logs into an Evaluation Set
# MAGIC
# MAGIC The Review application captures your user feedbacks.
# MAGIC
# MAGIC This feedback is saved under 2 tables within your schema.
# MAGIC
# MAGIC In this notebook, we will show you how to extract the logs from the Review App into an Evaluation Set.  It is important to review each row and ensure the data quality is high e.g., the question is logical and the response makes sense.
# MAGIC
# MAGIC 1. Requests with a 👍 :
# MAGIC     - `request`: As entered by the user
# MAGIC     - `expected_response`: If the user edited the response, that is used, otherwise, the model's generated response.
# MAGIC 2. Requests with a 👎 :
# MAGIC     - `request`: As entered by the user
# MAGIC     - `expected_response`: If the user edited the response, that is used, otherwise, null.
# MAGIC 3. Requests without any feedback
# MAGIC     - `request`: As entered by the user
# MAGIC
# MAGIC Across all types of requests, if the user 👍 a chunk from the `retrieved_context`, the `doc_uri` of that chunk is included in `expected_retrieved_context` for the question.

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-agents mlflow mlflow-skinny databricks-sdk==0.23.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Extracting the logs 
# MAGIC
# MAGIC
# MAGIC *Note: for now, this part requires a few SQL queries that we provide in this notebook to properly format the review app into training dataset.*
# MAGIC
# MAGIC *We'll update this notebook soon with an simpler version - stay tuned!*

# COMMAND ----------

from databricks import agents
import mlflow

MODEL_NAME = "rag_chatbot_customer_service"
MODEL_NAME_FQN = f"{catalog}.{db}.{MODEL_NAME}"
browser_url = mlflow.utils.databricks_utils.get_browser_hostname()

# # Get the name of the Inference Tables where logs are stored
active_deployments = agents.list_deployments()
active_deployment = next((item for item in active_deployments if item.model_name == MODEL_NAME_FQN), None)

# COMMAND ----------

active_deployment

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
print(active_deployment)
endpoint = w.serving_endpoints.get(active_deployment.endpoint_name)

try:
    endpoint_config = endpoint.config.auto_capture_config
except AttributeError as e:
    endpoint_config = endpoint.pending_config.auto_capture_config

inference_table_name = endpoint_config.state.payload_table.name
inference_table_catalog = endpoint_config.catalog_name
inference_table_schema = endpoint_config.schema_name

# Cleanly formatted tables
assessment_table = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}_assessment_logs`"
request_table = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}_request_logs`"

# Note: you might have to wait a bit for the tables to be ready
print(f"Request logs: {request_table}")
requests_df = spark.table(request_table)
print(f"Assessment logs: {assessment_table}")
#Temporary helper to extract the table - see _resources/00-init-advanced 
assessment_df = deduplicate_assessments_table(assessment_table)

# COMMAND ----------

requests_with_feedback_df = requests_df.join(assessment_df, requests_df.databricks_request_id == assessment_df.request_id, "left")
display(requests_with_feedback_df.select("request_raw", "trace", "source", "text_assessment", "retrieval_assessments"))

# COMMAND ----------

requests_with_feedback_df.createOrReplaceTempView('latest_assessments')
eval_dataset = spark.sql(f"""
-- Thumbs up.  Use the model's generated response as the expected_response
select
  a.request_id,
  r.request,
  r.response as expected_response,
  'thumbs_up' as type,
  a.source.id as user_id
from
  latest_assessments as a
  join {request_table} as r on a.request_id = r.databricks_request_id
where
  a.text_assessment.ratings ["answer_correct"].value == "positive"
union all
  --Thumbs down.  If edited, use that as the expected_response.
select
  a.request_id,
  r.request,
  IF(
    a.text_assessment.suggested_output != "",
    a.text_assessment.suggested_output,
    NULL
  ) as expected_response,
  'thumbs_down' as type,
  a.source.id as user_id
from
  latest_assessments as a
  join {request_table} as r on a.request_id = r.databricks_request_id
where
  a.text_assessment.ratings ["answer_correct"].value = "negative"
union all
  -- No feedback.  Include the request, but no expected_response
select
  a.request_id,
  r.request,
  IF(
    a.text_assessment.suggested_output != "",
    a.text_assessment.suggested_output,
    NULL
  ) as expected_response,
  'no_feedback_provided' as type,
  a.source.id as user_id
from
  latest_assessments as a
  join {request_table} as r on a.request_id = r.databricks_request_id
where
  a.text_assessment.ratings ["answer_correct"].value != "negative"
  and a.text_assessment.ratings ["answer_correct"].value != "positive"
  """)
display(eval_dataset)

# COMMAND ----------


