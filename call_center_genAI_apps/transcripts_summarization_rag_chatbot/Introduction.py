# Databricks notebook source
# MAGIC %md
# MAGIC # Call Center Transcript Summarization LLM Chatbots with Databricks DBRX Instruct
# MAGIC
# MAGIC In this set of notebooks, we will take raw call center tanscripts and perform summarization and sentiment analysis using Prompt engineering with DBRX instruct, we will also create a RAG Chatbot with Databricks Vector Search and DBRX instruct.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Datasets Provided
# MAGIC
# MAGIC The dataset provided for this PoC Template are synthetic datasets of a customer service operation of an insurance business
# MAGIC
# MAGIC * Paw call center transcript in JSON
# MAGIC * Raw policy information in CSV

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC <img src="https://github.com/qian-yu-db/Fins-SSA-GenAi-Offerings/blob/main/imgs/transcripts_summarization_rag.png?raw=true"  width="1000px;">
# MAGIC
# MAGIC
# MAGIC Steps:
# MAGIC
# MAGIC 1. Using Delta Live Table the ingest raw transcript and enrich with raw policy data
# MAGIC 2. Perform summarization and sentiment analsysi using DBRX Instruct in batch
# MAGIC 3. Create a RAG Chatbot using the summariztion as context using vector search index and DBRX Instruct

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Now get started with the [00-setup notebook]($./00-setup).
