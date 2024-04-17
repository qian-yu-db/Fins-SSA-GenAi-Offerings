# Databricks notebook source
# MAGIC %pip install mlflow==2.9.0 lxml==4.9.3 transformers==4.30.2 langchain==0.0.344 databricks-vectorsearch==0.22
# MAGIC %pip install --upgrade databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Notebook Description
# MAGIC
# MAGIC In this notebook, We will build a RAG Knowledge Chatbot
# MAGIC
# MAGIC * We first setup the vector search index using databricks BGE embedding model and databricks vector search db
# MAGIC * We then build a RAG (Retrival Augmented Generation) chatbot app using the vector search index as the knowedge context

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Read Transcripts and Policy Data

# COMMAND ----------

table_name = "customer_service_nlp"

spark.sql(f"USE CATALOG {catalog};")
spark.sql(f"USE schema {db};")
df_conversation = spark.table(table_name)
display(df_conversation)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Vector Search Index
# MAGIC
# MAGIC ## Enable Databricks Embedding Foundation Model Endpoint (`databricks-bg-large-en`)

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

#Embeddings endpoints convert text into a vector (array of float). Here is an example using BGE:
response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create a Vector Search Index endpoint and sync with the transcript summary table
# MAGIC
# MAGIC **Note**: It will take a few minutes to get a new vector search index to be created and be ready to use

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

# COMMAND ----------

# Check if the endpoint already exist otherwise create an endpoint
if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# DBTITLE 1,If the vector store endpoint already exist, we do not need to recreate
index_name = "customer_service_vs_index"
vs_index_fullname = f"{catalog}.{db}.{index_name}"
source_table_fullname = f"{catalog}.{db}.{table_name}"

if not index_exists(vsc=vsc, endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_full_name=vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="POLICY_NO",
    embedding_source_column='summary', #The column containing our text
    embedding_model_endpoint_name='databricks-bge-large-en' #The embedding endpoint used to create the embeddings
  )
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else: 
  #Trigger a sync to update our vs content with the new data saved in the table
  print(f"The vector search index endpoint {VECTOR_SEARCH_ENDPOINT_NAME} is already exists, we will perform a sync")
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### We can perform similarity search on the vector search index
# MAGIC
# MAGIC Here we retreive top 3 records related to `car accidents`

# COMMAND ----------

question = "car accident"

# We can calculate the embedding of the question directly using the embedding endpoint
# response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
# embeddings = [e['embedding'] for e in response.data]

# We can perform similarity search using vecctor search index
results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text=question,
  columns=["summary", "POLICY_NO"],
  num_results=3)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Create an RAG Application to get AI recommendations

# COMMAND ----------

import os

# url used to send the request to your model from the serverless endpoint
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("databricks_token_qyu", "qyu_rag_sp_token")

# COMMAND ----------

from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings

def get_retriever(persist_dir: str = None):
    embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=vs_index_fullname
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="summary", embedding=embedding_model
    )
    return vectorstore.as_retriever()

# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("Question about policy changes?")
print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Setup LLM Chat Endpoint with the Databricks `DBRX`

# COMMAND ----------

from langchain.chat_models import ChatDatabricks

chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 300)

# COMMAND ----------

from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks

TEMPLATE = """You are an Insurance company Support Assistant. You help representatives assist customers with their home, auto or life policies and claims.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

question = {"query": "How can we improve our response to theft related incidents"}

answer = chain.run(question)
print(answer)

# COMMAND ----------

displayHTML(answer.replace("\n", "<br>"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Saving the RAG model to Unity Catalog Registry

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.rag_customer_service_chatbot"

with mlflow.start_run(run_name="rag_customer_service_chatbot") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC # Deploying our Chat Model as a Serverless Model Endpoint 
# MAGIC
# MAGIC * Our model is saved in Unity Catalog. The last step is to deploy it as a Model Serving.
# MAGIC * We'll then be able to sending requests from an frontend application.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient

w = WorkspaceClient()

# COMMAND ----------

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

serving_endpoint_name = f"rag_cs_endpoint_{catalog}_{db}"[:63]
latest_model_version = get_latest_model_version(model_name)

config = EndpointCoreConfigInput.from_dict({
    "served_models": [
        {
            "name": serving_endpoint_name,
            "model_name": model_name,
            "model_version": latest_model_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": "True",
            "environment_vars": {"DATABRICKS_TOKEN": "{{secrets/databricks_token_qyu/qyu_rag_sp_token}}"}
        }
    ],
    "auto_capture_config":{
        "catalog_name": catalog,
        "schema_name": db,
        "table_name_prefix": "rag_chat_customer_service"
    }
})

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=config.served_models, 
                                               name=serving_endpoint_name, 
                                               auto_capture_config=config.auto_capture_config)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the endpoint we created

# COMMAND ----------

question = {"query": "What are some of main incidents related to home insurance?"}

answer = w.serving_endpoints.query(serving_endpoint_name, inputs=[question])
print(answer.predictions[0])

# COMMAND ----------

question = {"query": "What are some of the customer's asks related to motocycle?"}

answer = w.serving_endpoints.query(serving_endpoint_name, inputs=[question])
print(answer.predictions[0])
