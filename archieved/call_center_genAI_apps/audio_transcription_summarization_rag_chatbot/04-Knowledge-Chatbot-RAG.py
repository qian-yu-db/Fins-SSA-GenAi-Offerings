# Databricks notebook source
# MAGIC %pip install --quiet -U databricks-agents mlflow-skinny mlflow mlflow[gateway] langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 databricks-vectorsearch databricks-sdk==0.23.0
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

table_name = "customer_service"

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
# MAGIC
# MAGIC * Implemented with Agent Framework
# MAGIC   * Define RAG Chain config in a yaml file
# MAGIC   * Define RAG Chain in a python script
# MAGIC   * Log the RAG model from location
# MAGIC   * Enable human feedback app 

# COMMAND ----------

import os
import yaml
import mlflow

mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

rag_chain_config = {
    "databricks_resources": {
        "llm_endpoint_name": "databricks-dbrx-instruct",
        "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,
    },
    "input_example": {
        "messages": [{"content": "car accident", "role": "user"}]
    },
    "llm_config": {
        "llm_parameters": {"max_tokens": 500, "temperature": 0.01},
        "llm_prompt_template": "You are a trusted insurance company Support Assistant that help representatives assist customers with their home, auto or life policies and claims. You helps answer questions based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know.  Here is some context which might or might not help you answer: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this context, answer this question: {question}",
        "llm_prompt_template_variables": ["context", "question"],
    },
    "retriever_config": {
        "chunk_template": "Passage: {chunk_text}\n",
        "data_pipeline_tag": "poc",
        "parameters": {"k": 5, "query_type": "ann"},
        "schema": {"chunk_text": "summary", "document_uri": "topic", "primary_key": "POLICY_NO"},
        "vector_search_index": f"{catalog}.{db}.{index_name}",
    },
}
try:
    with open('rag_chain_config.yaml', 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')

# COMMAND ----------

# MAGIC %%writefile chain.py
# MAGIC import os
# MAGIC import mlflow
# MAGIC from operator import itemgetter
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from langchain_community.chat_models import ChatDatabricks
# MAGIC from langchain_community.vectorstores import DatabricksVectorSearch
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.prompts import PromptTemplate
# MAGIC from langchain_core.runnables import RunnablePassthrough
# MAGIC
# MAGIC ## Enable MLflow Tracing
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC # Return the string contents of the most recent message from the user
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]["content"]
# MAGIC
# MAGIC #Get the conf from the local conf file
# MAGIC model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')
# MAGIC
# MAGIC databricks_resources = model_config.get("databricks_resources")
# MAGIC retriever_config = model_config.get("retriever_config")
# MAGIC llm_config = model_config.get("llm_config")
# MAGIC
# MAGIC # Connect to the Vector Search Index
# MAGIC vs_client = VectorSearchClient(disable_notice=True)
# MAGIC vs_index = vs_client.get_index(
# MAGIC     endpoint_name=databricks_resources.get("vector_search_endpoint_name"),
# MAGIC     index_name=retriever_config.get("vector_search_index"),
# MAGIC )
# MAGIC vector_search_schema = retriever_config.get("schema")
# MAGIC
# MAGIC # Turn the Vector Search index into a LangChain retriever
# MAGIC vector_search_as_retriever = DatabricksVectorSearch(
# MAGIC     vs_index,
# MAGIC     text_column=vector_search_schema.get("chunk_text"),
# MAGIC     columns=[
# MAGIC         vector_search_schema.get("primary_key"),
# MAGIC         vector_search_schema.get("chunk_text"),
# MAGIC         vector_search_schema.get("document_uri"),
# MAGIC     ],
# MAGIC ).as_retriever(search_kwargs=retriever_config.get("parameters"))
# MAGIC
# MAGIC # Required to:
# MAGIC # 1. Enable the RAG Studio Review App to properly display retrieved chunks
# MAGIC # 2. Enable evaluation suite to measure the retriever
# MAGIC mlflow.models.set_retriever_schema(
# MAGIC     primary_key=vector_search_schema.get("primary_key"),
# MAGIC     text_column=vector_search_schema.get("chunk_text"),
# MAGIC     doc_uri=vector_search_schema.get("document_uri")
# MAGIC )
# MAGIC
# MAGIC # Method to format the docs returned by the retriever into the prompt
# MAGIC def format_context(docs):
# MAGIC     chunk_template = retriever_config.get("chunk_template")
# MAGIC     chunk_contents = [
# MAGIC         chunk_template.format(
# MAGIC             chunk_text=d.page_content,
# MAGIC         )
# MAGIC         for d in docs
# MAGIC     ]
# MAGIC     return "".join(chunk_contents)
# MAGIC
# MAGIC # Prompt Template for generation
# MAGIC prompt = PromptTemplate(
# MAGIC     template=llm_config.get("llm_prompt_template"),
# MAGIC     input_variables=llm_config.get("llm_prompt_template_variables"),
# MAGIC )
# MAGIC
# MAGIC # FM for generation
# MAGIC model = ChatDatabricks(
# MAGIC     endpoint=databricks_resources.get("llm_endpoint_name"),
# MAGIC     extra_params=llm_config.get("llm_parameters"),
# MAGIC )
# MAGIC
# MAGIC # RAG Chain
# MAGIC chain = (
# MAGIC     {
# MAGIC         "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
# MAGIC         "context": itemgetter("messages")
# MAGIC         | RunnableLambda(extract_user_query_string)
# MAGIC         | vector_search_as_retriever
# MAGIC         | RunnableLambda(format_context),
# MAGIC     }
# MAGIC     | prompt
# MAGIC     | model
# MAGIC     | StrOutputParser()
# MAGIC )
# MAGIC
# MAGIC # Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=chain)

# COMMAND ----------

# Log the model to MLflow
with mlflow.start_run(run_name=f"rag_chatbot_customer_service"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(os.getcwd(), 'chain.py'),  # Chain code file e.g., /path/to/the/chain.py 
        model_config='rag_chain_config.yaml',  # Chain configuration 
        artifact_path="chain",  # Required by MLflow
        input_example=model_config.get("input_example"),  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
    )

# Test the chain locally
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(model_config.get("input_example"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy our RAG application and open it for external expert users

# COMMAND ----------

from databricks import agents
MODEL_NAME = "rag_chatbot_customer_service"
MODEL_NAME_FQN = f"{catalog}.{db}.{MODEL_NAME}"

# COMMAND ----------

MODEL_NAME_FQN

# COMMAND ----------

instructions_to_reviewer = f"""### Instructions for Testing the our Customer Service Chatbot assistant

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement.

1. **Variety of Questions**:
   - Please try a wide range of questions that you anticipate the end users of the application will ask. This helps us ensure the application can handle the expected queries effectively.

2. **Feedback on Answers**:
   - After asking each question, use the feedback widgets provided to review the answer given by the application.
   - If you think the answer is incorrect or could be improved, please use "Edit Answer" to correct it. Your corrections will enable our team to refine the application's accuracy.

3. **Review of Returned Customer Service Summary**:
   - Carefully review each customer service summary that the system returns in response to your question.
   - Use the thumbs up/down feature to indicate whether the document was relevant to the question asked. A thumbs up signifies relevance, while a thumbs down indicates the document was not useful.

Thank you for your time and effort in testing our assistant. Your contributions are essential to delivering a high-quality product to our end users."""

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=MODEL_NAME_FQN)

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=MODEL_NAME_FQN, model_version=uc_registered_model_info.version, scale_to_zero=True)

# Add the user-facing instructions to the Review App
agents.set_review_instructions(MODEL_NAME_FQN, instructions_to_reviewer)

wait_for_model_serving_endpoint_to_be_ready(deployment_info.endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grant stakeholders access to the Mosaic AI Agent Evaluation App
# MAGIC
# MAGIC Now, grant your stakeholders permissions to use the Review App. To simplify access, stakeholders do not require to have Databricks accounts.

# COMMAND ----------

user_list = ["q.yu@databricks.com"]
# Set the permissions.
agents.set_permissions(model_name=MODEL_NAME_FQN, users=user_list, permission_level=agents.PermissionLevel.CAN_QUERY)

print(f"Share this URL with your stakeholders: {deployment_info.review_app_url}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Find review app name
# MAGIC
# MAGIC If you lose this notebook's state and need to find the URL to your Review App, you can list the chatbot deployed:

# COMMAND ----------

active_deployments = agents.list_deployments()
active_deployment = next((item for item in active_deployments if item.model_name == MODEL_NAME_FQN), None)
if active_deployment:
  print(f"Review App URL: {active_deployment.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC Next Step, we create a RAG Chatbot with Notebook [05-Inference-Tables-Monitoring-LLM-Metrics]($./05-Inference-Tables-Monitoring-LLM-Metrics)

# COMMAND ----------


