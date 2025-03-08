# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Configuration file
# MAGIC
# MAGIC **Prerequisite**: unity catalog needs to be enabled at your workspace
# MAGIC
# MAGIC Please change your catalog and schema inside the `config` here to run the notebook on a different catalog

# COMMAND ----------

catalog = "qyu"
dbName = db = "test"
volume_name_policies = "volume_policies"
volume_name_audio = "volume_speech"
volume_name_rag = "volume_rag"
VECTOR_SEARCH_ENDPOINT_NAME = "rag_endpoint_qyu"

# COMMAND ----------

def use_and_create_db(catalog, dbName, cloud_storage_path = None):
  print(f"USE CATALOG `{catalog}`")
  spark.sql(f"USE CATALOG `{catalog}`")
  spark.sql(f"""create database if not exists `{dbName}` """)

assert catalog not in ['hive_metastore', 'spark_catalog']

#If the catalog is defined, we force it to the given value and throw exception if not.
if len(catalog) > 0:
  current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
  if current_catalog != catalog:
    catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
    if catalog not in catalogs:
      spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
  use_and_create_db(catalog, dbName)

# COMMAND ----------

spark.sql(f'USE CATALOG {catalog};')
spark.sql(f'USE SCHEMA {db};')

print("---------------")
print("Current Setup")
print("---------------")
print(f"Use Catalog: {catalog}")
print(f"Use Schema: {db}")
print(f"Use Volumes for policy data: {volume_name_audio}")
print(f"Use Volumes for speech data: {volume_name_policies}")
print(f"Use Volumes for policy data: {volume_name_rag}")
print(f"Use Vector Search Endpoint name: {VECTOR_SEARCH_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Helper Functions Shared Across Notebooks
# MAGIC
# MAGIC * `get_latest_model_version()`: Return the latest model version
# MAGIC * `index_exists()`: Check whether a vector index already exists
# MAGIC * `wait_for_vs_endpoint_to_be_ready()`: wait until the vector index endpoint is ready to be queried
# MAGIC * `wait_for_index_to_be_ready()`: wait for vector index to be ready
# MAGIC * `get_endpoint_status()`: collect endpoint status
# MAGIC * `wait_for_model_serving_endpoint_to_be_ready()`: wait for model serving endpoint to be ready

# COMMAND ----------

import time

def get_latest_model_version(model_name):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version
  

def index_exists(vsc, endpoint_name, index_full_name):
    try:
        dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
        return dict_vsindex.get('status').get('ready', False)
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False


def endpoint_exists(vsc, vs_endpoint_name):
  try:
    return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
  except Exception as e:
    #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
    if "REQUEST_LIMIT_EXCEEDED" in str(e):
      print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. The demo will consider it exists")
      return True
    else:
      raise e
  

def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    try:
      endpoint = vsc.get_endpoint(vs_endpoint_name)
    except Exception as e:
      #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
      if "REQUEST_LIMIT_EXCEEDED" in str(e):
        print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your endpoint status")
        return
      else:
        raise e
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")


def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")


def get_endpoint_status(endpoint_name):
    # Fetch the PAT token to send in the API request
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}", json={"name": endpoint_name}, headers=headers).json()

    # Verify that Inference Tables is enabled.
    if "auto_capture_config" not in response.get("config", {}) or not response["config"]["auto_capture_config"]["enabled"]:
        raise Exception(f"Inference Tables is not enabled for endpoint {endpoint_name}. \n"
                        f"Received response: {response} from endpoint.\n"
                        "Please create an endpoint with Inference Tables enabled before running this notebook.")

    return response
  
def wait_for_model_serving_endpoint_to_be_ready(ep_name):
  from databricks.sdk import WorkspaceClient
  from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
  import time

  # TODO make the endpoint name as a param
  # Wait for it to be ready
  w = WorkspaceClient()
  state = ""
  for i in range(200):
      state = w.serving_endpoints.get(ep_name).state
      if state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
          if i % 40 == 0:
              print(f"Waiting for endpoint to deploy {ep_name}. Current state: {state}")
          time.sleep(10)
      elif state.ready == EndpointStateReady.READY:
        print('endpoint ready.')
        return
      else:
        break
  raise Exception(f"Couldn't start the endpoint, timeout, please check your endpoint for more details: {state}")

# COMMAND ----------

def deduplicate_assessments_table(assessment_table):
    # De-dup response assessments
    assessments_request_deduplicated_df = spark.sql(f"""select * except(row_number)
                                        from ( select *, row_number() over (
                                                partition by request_id
                                                order by
                                                timestamp desc
                                            ) as row_number from {assessment_table} where text_assessment is not NULL
                                        ) where row_number = 1""")
    # De-dup the retrieval assessments
    assessments_retrieval_deduplicated_df = spark.sql(f"""select * except( retrieval_assessment, source, timestamp, text_assessment, schema_version),
        any_value(timestamp) as timestamp,
        any_value(source) as source,
        collect_list(retrieval_assessment) as retrieval_assessments
      from {assessment_table} where retrieval_assessment is not NULL group by request_id, source.id, step_id"""    )

    # Merge together
    assessments_request_deduplicated_df = assessments_request_deduplicated_df.drop("retrieval_assessment", "step_id")
    assessments_retrieval_deduplicated_df = assessments_retrieval_deduplicated_df.withColumnRenamed("request_id", "request_id2").withColumnRenamed("source", "source2").drop("step_id", "timestamp")

    merged_deduplicated_assessments_df = assessments_request_deduplicated_df.join(
        assessments_retrieval_deduplicated_df,
        (assessments_request_deduplicated_df.request_id == assessments_retrieval_deduplicated_df.request_id2) &
        (assessments_request_deduplicated_df.source.id == assessments_retrieval_deduplicated_df.source2.id),
        "full"
    ).select(
        [str(col) for col in assessments_request_deduplicated_df.columns] +
        [assessments_retrieval_deduplicated_df.retrieval_assessments]
    )

    return merged_deduplicated_assessments_df
