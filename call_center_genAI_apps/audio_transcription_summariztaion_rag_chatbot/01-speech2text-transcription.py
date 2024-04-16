# Databricks notebook source
# MAGIC %pip install --upgrade databricks-sdk
# MAGIC %pip install tqdm
# MAGIC %pip install mlflow==2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook Description
# MAGIC
# MAGIC In this notebook, we will show how user can enable an open source AI model from Databricks Marketing Place and deploy it to a Databricks Model Serving Endpoint to Perform Speech to Text Task and save the model predicted transcript with the model inference table from Databricks model serving endpoint
# MAGIC
# MAGIC Steps:
# MAGIC
# MAGIC 1. Ingest raw audio datafiles using `Autoloader` functionaltiy to a Delta lake table
# MAGIC 2. Register a open source speech to text model (`whisper-large-v3`) to unity catalog and deploy to model serving
# MAGIC 3. Perform model inference on the sample data

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Load sample audio data from a table
# MAGIC
# MAGIC We created samples of synthetic audio data and stored in a delta table.

# COMMAND ----------

audio_sub = "audio_clips"
volume_folder_audio_top = f"/Volumes/{catalog}/{db}/{volume_name_audio}"
volume_folder_audio_samples = f"{volume_folder_audio_top}/audio_clips"
display(dbutils.fs.ls(f'{volume_folder_audio_samples}/policy_no_102147884/'))

# COMMAND ----------

# Optionally clean up checkpoint directory
import os

clean_checkpoint_remove = True 
checkpoint_path = f'{volume_folder_audio_top}/checkpoints/'
raw_table_name = 'audio_raw'
print(f"checkpoint path: {checkpoint_path}")
print(f"audio table name: {raw_table_name}")

if os.path.exists(f'{checkpoint_path}') and clean_checkpoint_remove:
    dbutils.fs.rm(f"{checkpoint_path}", recurse=True)

spark.sql(f"DROP TABLE IF EXISTS {raw_table_name}")

# COMMAND ----------

from pyspark.sql.functions import regexp_extract

df = (spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("recursiveFileLookup", "true")
    .load(volume_folder_audio_samples))

df = df.withColumn('POLICY_NO', regexp_extract("path", r'.*\/policy_no_(\d+)\/.*', 1)) \
       .withColumn('speech_idx', regexp_extract("path", r'.*\/speech_(\d+)\.wav$', 1))

(df.writeStream
 .trigger(availableNow=True)
 .option("checkpointLocation", f'dbfs:{volume_folder_audio_top}/checkpoints/raw_audio')
 .table('audio_raw').awaitTermination()
 )

# COMMAND ----------

audio_df = spark.table("audio_raw")
display(audio_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## play a couple of clips of audio bytes

# COMMAND ----------

from IPython.display import Audio
import base64
import io

def bytes2audio(audio_bytes):
    """_summary_

    Args:
        audio_bytes (bytearray): bytearray of an audio
        transcripts (string): text transcript of the audio

    Returns:
        audio: audio clip
    """
    audio_io = io.BytesIO(audio_bytes)
    audio = Audio(audio_io.read())
    return audio

# COMMAND ----------

df_audio = audio_df.toPandas()
df_audio = df_audio.sort_values(by=['POLICY_NO', 'speech_idx']).reset_index(drop=True)
df_audio.head()

# COMMAND ----------


audio_byte = df_audio['content'][0]
display(bytes2audio(audio_byte))

# COMMAND ----------

audio_byte = df_audio['content'][1]
display(bytes2audio(audio_byte))

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the Open Source Whisper Model at a Databricks Serverless Model Serving Endpoint
# MAGIC
# MAGIC Access the open source AI model from Databricks Marketplace
# MAGIC
# MAGIC 1. Navigate to Databricks `Marketplace` in the right menu
# MAGIC 2. Search for `whisper` in the search box
# MAGIC 3. Click `whisper V3 Model` box, click `Get instant acess` botton at the top right corner to regiser the model to the unity catalog
# MAGIC     * You can define the catalog name to save and registered the model, this notebook uses `databricks_whisper_v3_model_{catalog}` as the name. If you defined a different name, please be sure to change the `model_catalog_name` variable in the below cell 
# MAGIC     * If the model already be registered, click `Open` to access the registered model
# MAGIC
# MAGIC Deploy the model to the Databricks model serving endpoint either with UI or programmetically using Databricks API
# MAGIC
# MAGIC Reference:
# MAGIC
# MAGIC * [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html#model-serving-with-databricks)
# MAGIC * [Model Serving Endpoint for Custom Models](https://docs.databricks.com/en/machine-learning/model-serving/custom-models.html#deploy-custom-models)

# COMMAND ----------

# Default catalog name when installing the model from Databricks Marketplace.
# Replace with the name of the catalog containing this model
# You can also specify a different model version to load for inference

model_name = 'whisper_large_v3'
version = "1"
model_catalog_name = f'databricks_whisper_v3_model_{catalog}' # please change to your defined model catalog name
model_uc_path = f"{model_catalog_name}.models.{model_name}"
endpoint_name = f'{model_name}_{catalog}'
print(f"Model name: {model_name}")
print(f"Unity Catalog Path: {model_uc_path}")
print(f"Endpoint Name: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Check Model is registered in the Unity Catalog after getting it from Market Place

# COMMAND ----------

from mlflow import MlflowClient
import mlflow 

mlflow.set_registry_uri("databricks-uc")

client = MlflowClient()
client.search_model_versions(f"name='{model_uc_path}'")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Deploy the model to model serving endpoint
# MAGIC
# MAGIC **Note**: Whisper is a very large and needs to be deployed on a GPU instances and it typically takes about 20 - 30 mins to be deployed. Please be patient and it is a good time to take a coffee break!

# COMMAND ----------

import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
from databricks.sdk.service.serving import AutoCaptureConfigInput

w = WorkspaceClient()
workload_type = "GPU_SMALL"

# COMMAND ----------

# DBTITLE 1,Define an endpint configuration
config = EndpointCoreConfigInput.from_dict({
    "served_models": [
        {
            "name": endpoint_name,
            "model_name": model_uc_path,
            "model_version": version,
            "workload_type": workload_type,
            "workload_size": "Small",
            "scale_to_zero_enabled": "True",
        }
    ],
    "auto_capture_config":{
        "catalog_name": catalog,
        "schema_name": db,
        "table_name_prefix": "speech2text_model_inference"
    }
})

# COMMAND ----------

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == endpoint_name), None
)

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
serving_endpoint_url = f"{host}/ml/endpoints/{endpoint_name}"
latest_model_version = get_latest_model_version(model_uc_path)

if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=endpoint_name, config=config, timeout=datetime.timedelta(minutes=30))
else:
    print(f"The endpoint {serving_endpoint_url} already exist...")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Perform Model Inference
# MAGIC
# MAGIC * There are multiple ways to construct the model request, we are using the `dataframe split` format for the request, for more details please see [document](https://docs.databricks.com/en/machine-learning/model-serving/score-custom-model-endpoints.html#supported-scoring-formats)
# MAGIC * We will use mlflow client api for model request

# COMMAND ----------

def mlflow_model_query(request, endpoint_name):
    import mlflow.deployments

    client = mlflow.deployments.get_deploy_client("databricks")
    response = client.predict(
        endpoint = endpoint_name,
        inputs=request
    )
    print(response['predictions'])

# COMMAND ----------

df_audio['audio_base64'] = df_audio['content'].apply(lambda x: base64.b64encode(x).decode('ascii'))
df_audio.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Perform inference for a single request to the endpoint

# COMMAND ----------

rows = df_audio.iterrows()
idx, row = next(rows)

request_example = {
    "client_request_id": row['POLICY_NO'],
    "dataframe_split": {
        "data": [row['audio_base64']]
    }

}

print(f"Inference at endpoint: {endpoint_name}")
mlflow_model_query(request_example, endpoint_name)

# COMMAND ----------

idx, row = next(rows)

request_example = {
    "client_request_id": row['POLICY_NO'],
    "dataframe_split": {
        "data": [row['audio_base64']]
    }
}
print(f"Inference at endpoint: {endpoint_name}")
mlflow_model_query(request_example, endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Perform Batch Inference
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Batch Inference with model serving endpoint

# COMMAND ----------

print(f"endpoint_name: {endpoint_name}")

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf

@pandas_udf("string")
def speech2text(audio_bytes: pd.Series, policy_id: pd.Series) -> pd.Series:
    import mlflow.deployments
    import base64
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    endpoint_name = "whisper_large_v3_qyu" # change to the right endpoint name if needed

    def transcribe(audio, policy_id):
        request = {
            "client_request_id": policy_id,
            "dataframe_split": {
                "data": [base64.b64encode(audio).decode('ascii')]
            }
        }

        response = deploy_client.predict(
            endpoint = endpoint_name,
            inputs=request
        )
        return response['predictions'][0]
    
    return pd.Series([transcribe(a, p) for a, p in zip(audio_bytes, policy_id)])

# COMMAND ----------

audio_df = audio_df.withColumn("transcript", speech2text("content", "POLICY_NO"))
display(audio_df)

# COMMAND ----------

from pyspark.sql.functions import col, concat_ws, collect_list, current_timestamp

transcript_df = audio_df.select("POLICY_NO", "speech_idx", "transcript") \
  .orderBy("POLICY_NO", "speech_idx") \
  .groupBy("POLICY_NO") \
  .agg(concat_ws(', ', collect_list("transcript")).alias("transcript")) 
display(transcript_df)

# COMMAND ----------

volume_folder_audio_top

# COMMAND ----------

volume_folder_to_save = f"{volume_folder_audio_top}/transcript_saved"

transcript_df = transcript_df.withColumn("datetime_record", current_timestamp())
transcript_df.repartition(10) \
    .write \
    .format("json") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{volume_folder_to_save}")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Batch Inference without model serving endpoint (**Optional**)
# MAGIC
# MAGIC You can also directly load the model as a Spark UDF and run batch inference on Databricks compute using Spark. We recommend using a GPU cluster with Databricks Runtime for Machine Learning version 14.1 or greater.
# MAGIC
# MAGIC Requirements:
# MAGIC
# MAGIC * This approach would require GPU cluster
# MAGIC * Download model from Unity Catalog
# MAGIC * Package the model using MLFlow Pyfunc Template to create a spark UDF for inference
# MAGIC * Model related packages needs to be enabled at worker nodes
# MAGIC     * Install `ffmpeg` to both driver and worker nodes
# MAGIC         * Install to driver with `%sh apt-get update -y && apt install ffmpeg -y`
# MAGIC         * Install to worker node with:
# MAGIC         ```scala
# MAGIC         import scala.concurrent.duration._
# MAGIC         import sys.process._
# MAGIC         var res = sc.runOnEachExecutor({ () =>
# MAGIC         var cmd_Result=Seq("bash", "-c", "apt-get update -y && apt install ffmpeg -y").!!
# MAGIC         cmd_Result }, 1000.seconds)
# MAGIC         ```
# MAGIC     * Install Huggingface transformers and torch modules if user is not using ML Runtime `%pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]`

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### batch inference code example
# MAGIC
# MAGIC
# MAGIC ```python
# MAGIC from pyspark.sql.types import StringType
# MAGIC import mlflow
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC
# MAGIC version = get_latest_model_version(model_uc_path)
# MAGIC predict = mlflow.pyfunc.spark_udf(spark, f"models:/{model_uc_path}/{version}", StringType())
# MAGIC
# MAGIC import pandas as pd
# MAGIC
# MAGIC wav_file = '/dbfs/user/<user_name>/speech/test.wav'
# MAGIC with open(wav_file, 'rb') as audio_file:
# MAGIC     audio_bytes = audio_file.read()
# MAGIC     dataset = pd.DataFrame(pd.Series([audio_bytes]))
# MAGIC
# MAGIC df_test = spark.createDataFrame(dataset)
# MAGIC
# MAGIC transcript = df_test.select(transcribe(df_test["0"]).alias('transcription'))
# MAGIC display(trascript)
# MAGIC ```
