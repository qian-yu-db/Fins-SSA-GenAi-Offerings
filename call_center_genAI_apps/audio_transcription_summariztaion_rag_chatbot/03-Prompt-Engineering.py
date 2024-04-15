# Databricks notebook source
# MAGIC %pip install mlflow==2.9.0
# MAGIC %pip install tiktoken==0.5.1
# MAGIC %pip install databricks-genai-inference langchain
# MAGIC %pip install --upgrade databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook Description
# MAGIC
# MAGIC In this notebook, we will perform summarization and sentiment analysis on the customer service transcripts in batch using prompt engineering with databricks foundation model endpoint and databricks LLM **`DBRX`** 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Read Transcript Data from a Delta Lake Table

# COMMAND ----------

table_name = "transcript_enriched_ml"

spark.sql(f"USE CATALOG {catalog};")
spark.sql(f"USE SCHEMA {db};")
transcript_df = spark.table(table_name)
display(transcript_df)

# COMMAND ----------

# Showing an example of a conversation
transcripts = transcript_df.select("transcript").collect()
transcripts[0].transcript

# COMMAND ----------

# MAGIC %md
# MAGIC # Perform summerization and sentiment analysis Using Databricks DBRX LLM

# COMMAND ----------

from databricks_genai_inference import ChatCompletion, ChatSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql import functions as F
import pandas as pd
import json
import tiktoken

# COMMAND ----------

import pandas as pd

@pandas_udf("string")
def summarizer(conversations: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    def get_summary(conv):
        business='insurance'
        system_message = f"You are an expert in {business} and a helpful assistant."

        # ensure the token size does not exceed limit
        encoding_name = "cl100k_base"
        max_number_of_tokens = 3000

        Prompt = \
        f"""
        Please summarize the below conversation in 3 to 5 sentences and highlight 3 key words

        <conversation>
        {conv}
        <conversation>
        """
        messages = [{"role":"system", "content":system_message},
                    {"role":"user", "content":Prompt}]
        response = deploy_client.predict(endpoint="databricks-dbrx-instruct", inputs={"messages": messages})
        return response.choices[0]['message']['content']
    
    return pd.Series([get_summary(c) for c in conversations])



@pandas_udf("string")
def sentiment(conversations: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    def get_sentiment(conv):
        system_message = f"You are an expert in human emotion and a helpful assistant."
        Prompt = \
        f"""
        Analyze the sentiment of this customer agent conversation and evalute a sentiment score: 
        
        - if the customer is 'happy', give score of 1 
        - if customer is 'neutral', give score of 0
        - if 'unhappy', give score of -1.

        <conversation>
        {conv}
        <conversation>

        Please return the sentiment score in the format of: sentiment score: `score`
        """
        messages = [{"role":"system", "content":system_message},
                    {"role":"user", "content":Prompt}]
        response = deploy_client.predict(endpoint="databricks-dbrx-instruct", inputs={"messages": messages})
        return response.choices[0]['message']['content']
    
    return pd.Series([get_sentiment(c) for c in conversations])

# COMMAND ----------

transcript_df_with_summary = (transcript_df
                                .withColumn("summary", summarizer("transcript"))
                                .withColumn("sentiment", sentiment("transcript")))
display(transcript_df_with_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Save the summarization and sentiment results with the transcript to a table 
# MAGIC
# MAGIC * we can build a LLM RAG chatbot with the summary as context using the databricks vector database next

# COMMAND ----------

(transcript_df_with_summary.write
    .mode('overwrite')
    .option("overwriteSchema", "true")
    .saveAsTable("customer_service_nlp"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Lastly, we change the table properties of the result table so it can be used as the source[vector search index](https://docs.databricks.com/en/generative-ai/vector-search.html)
# MAGIC
# MAGIC **Note**: The following requirements are needed for enable databricks vector search index:
# MAGIC
# MAGIC * Unity Catalog enabled workspace.
# MAGIC * Serverless compute enabled.
# MAGIC * Source delta lake table have Change Data Feed enabled.
# MAGIC * CREATE TABLE privileges on catalog schema(s) to create indexes.
# MAGIC * Personal access tokens enabled.
# MAGIC

# COMMAND ----------

# DBTITLE 1,To enable for vector db, we want to enable chage data feed table property
# MAGIC %sql
# MAGIC ALTER TABLE customer_service_nlp SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC DESCRIBE EXTENDED customer_service_nlp

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Next Step, we create a RAG Chatbot with [Notebook 03-Knowledge-Chatbot-RAG]($./03-Knowledge-Chatbot-RAG)

# COMMAND ----------


