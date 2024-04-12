# Application. [Extract Intelligence from raw call center transcripts using RAG](./transcripts_summarization_rag_chatbot/)

## Architecture

![image](../../imgs/transcripts_summarization_rag.png)


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
