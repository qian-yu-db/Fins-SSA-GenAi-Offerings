# Introduction

This series of notebooks, we shows how Generative AI (GenAI) can be used to unlock the value of call center data, e.g. `.json` transcript data, `.wav` audio data leveraging Databricks' advanced GenAI capabilities.

The goal is to give Databricks’ customers a blueprint to start a POC for more responsive, efficient, and customer-centric call center operation that can improve customer satisfaction, decrease turnaround times, and contribute positively to the overall business performance.

# Application 1. [Extract Intelligence from raw call center transcripts using RAG](./transcripts_summarization_rag_chatbot/)

## Architecture

![image](../imgs/transcripts_summarization_rag.png)


## Notebooks

### 0. Download Dataset and setup unity catalog schema & volume

* 00-setup

### 1. Data Ingestions with Delta Live Table

* 01-DLT-Transcript-Policy
* 01.1-DLT-Transcript-Enriched-Persist-MV

### 2. Prompte Engineering with Databricks DBRX Fundation LLM

* 02-Prompt-Engineering

### 3. RAG Chatbot with Databricks DBRX Fundation LLM

* 03-Knowledge-Chatbot-RAG
