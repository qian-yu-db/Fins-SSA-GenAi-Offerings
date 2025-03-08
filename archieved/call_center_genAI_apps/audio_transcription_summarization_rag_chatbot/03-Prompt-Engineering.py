# Databricks notebook source
# MAGIC %pip install mlflow
# MAGIC %pip install tiktoken==0.5.1
# MAGIC %pip install databricks-genai-inference langchain
# MAGIC %pip install --upgrade databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

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
from pyspark.sql.types import IntegerType, StringType

@pandas_udf("string")
def summarizer(conversations: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    def get_summary(conv):
        business='insurance'
        system_message = f"You are an expert in {business} and a helpful assistant."

        Prompt = \
        f"""
        Summarize the below conversation in 3 to 5 sentences and highlight 3 key words

        ---------------------
        conversation: {conv}
        ---------------------
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
        system_message = f"You are a helpful assistant."
        Prompt = \
        f"""
        Is the predominant sentiment in the following conversation positive, negative, or neutral?

        --------------------
        conversation: {conv}
        --------------------

        Respond in one word: positive, negative, or neutral. DO NOT EXPLAIN!
        """

        messages = [{"role":"system", "content":system_message},
                    {"role":"user", "content":Prompt}]
        response = deploy_client.predict(endpoint="databricks-dbrx-instruct", inputs={"messages": messages})
        return response.choices[0]['message']['content']
    
    return pd.Series([get_sentiment(c) for c in conversations])


@pandas_udf("string")
def topic_classification(conversations: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    topics_descr_list = \
    """
    car accident: customer involved in a car accident
    policy change: customer would like change, update, or add on their policy or information
    motorcrycle related: customer has a motorcycle related query
    home accident: customer has a damage about their home such as kitchen, roof, bathroom, etc
    general question: customer asked a general question on their insurance
    theft incident: customer had things stolen from their cars and homes
    """

    def get_topic(conv):
        system_message = f"You are a helpful assistant."
        Prompt = \
        f"""
        what is the predominant topic in the below conversation? please include only one main topic from the provided list.

        List of topics with descriptions (delimited with ":"):
        {topics_descr_list}

        --------------
        Conversation: {conv}
        --------------

        Respond with one topic. DO NOT EXPLAIN!
        """
        messages = [{"role":"system", "content":system_message},
                    {"role":"user", "content":Prompt}]
        response = deploy_client.predict(endpoint="databricks-dbrx-instruct", inputs={"messages": messages})
        return response.choices[0]['message']['content']
    
    return pd.Series([get_topic(c) for c in conversations])


@udf(returnType=StringType())
def get_short_topic(topic):
    topics_descr_list = ["car accident", "policy change", "motorcycle related", "home accident", 
                         "general question", "theft incident"]
    for t in topics_descr_list:
        if t in topic.lower():
            return t
    return "no topic"

# COMMAND ----------

transcript_df_with_summary = (transcript_df
                                .withColumn("summary", summarizer("transcript"))
                                .withColumn("sentiment", sentiment("transcript"))
                                .withColumn("topic", topic_classification("transcript")))

display(transcript_df_with_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Post-processing to create short class names

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col, udf, max, regexp_extract, regexp, cast, lower

transcript_df_classification_final = transcript_df_with_summary \
        .withColumn('sentiment_clean', lower(regexp_extract("sentiment", r'(?i)(positive|negative|neutral)', 0))) \
        .withColumn('topic_clean', get_short_topic(col('topic')))
        
display(transcript_df_classification_final)

# COMMAND ----------

transcript_df_classification_final = transcript_df_classification_final \
    .select("POLICY_NO", "transcript", "datetime_record", "MAKE", "MODEL", "DRV_DOB", "address", "summary", "sentiment_clean", "topic_clean")
transcript_df_classification_final = transcript_df_classification_final \
    .withColumnRenamed("sentiment_clean", "sentiment") \
    .withColumnRenamed("topic_clean", "topic")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Save the summarization and sentiment results with the transcript to a table 
# MAGIC
# MAGIC * we can build a LLM RAG chatbot with the summary as context using the databricks vector database next
# MAGIC * We write a table with < 20 columns due to vector search index current column limits for workspace

# COMMAND ----------

(transcript_df_classification_final
    .write
    .mode('overwrite')
    .option("overwriteSchema", "true")
    .saveAsTable("customer_service"))

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
# MAGIC ALTER TABLE customer_service SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC DESCRIBE EXTENDED customer_service

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Next Step, we create a RAG Chatbot with [Notebook 04-Knowledge-Chatbot-RAG]($./04-Knowledge-Chatbot-RAG)
