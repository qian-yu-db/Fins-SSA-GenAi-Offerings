# Databricks notebook source
# MAGIC %pip install --upgrade --quiet langchain langchain-openai langchainhub langchain-community 
# MAGIC %pip install --upgrade langchain databricks-sql-connector databricks_vectorsearch gradio
# MAGIC %pip install mlflow mlflow-skinny databricks-agents -U -qqq
# MAGIC %pip install SQLAlchemy -U
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC # Databrick SQL agent

# COMMAND ----------

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_community.chat_models import ChatDatabricks
import mlflow

## Enable MLflow Tracing
mlflow.langchain.autolog()

# COMMAND ----------

agent_config = {
    "llm_endpoint_name": "databricks-dbrx-instruct",
    "llm_parameters": {"temperature": 0.0, "max_tokens": 1500}
}

# COMMAND ----------

db = SQLDatabase.from_databricks(catalog="fins_genai", schema="customer_information")
llm = ChatDatabricks(
    endpoint=agent_config.get("llm_endpoint_name"),
    extra_params=agent_config.get("llm_parameters"),
)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True, handle_parsing_errors=True)

# COMMAND ----------

response = agent.invoke("find the customer with policy number 101618572")
response['output']

# COMMAND ----------

response = agent.invoke("Is William Dyer's interaction positive?")
response['output']

# COMMAND ----------

response = agent.invoke("write a email response based on William Dyer (policy number 101618572)'s interactions")
response['output']

# COMMAND ----------


