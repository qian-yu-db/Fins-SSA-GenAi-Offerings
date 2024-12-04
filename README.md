# Fins SSA GenAI Offering <!-- omit in toc -->

The goal of this repo is to develop and deliver GenAI solutions to enable and accelerate customer GenAI PoC Projects with Databricks

# Table of Content <!-- omit in toc -->
- [Requirements](#requirements)
- [PoC Accelerator Short Assets](#poc-accelerator-short-assets)
  - [Data Ingestion and Preprocessing Architecture Patterns](#data-ingestion-and-preprocessing-architecture-patterns)
  - [End to End GenAI Application Architecture Patterns](#end-to-end-genai-application-architecture-patterns)
- [When to Use](#when-to-use)
  - [Getting Started](#getting-started)
- [Resources](#resources)
- [Limitations](#limitations)

# Requirements

* Databricks workspace with Serverless and Unity Catalog enabled
* Python 3.9+

# PoC Accelerator Short Assets

## Data Ingestion and Preprocessing Architecture Patterns

| Input Data Types | Input Data Store  | chunking performed | OSS Technolgoy | Component Asset  |
|------------------|-------------------|--------------------|----------------|---------------|
| JSON Text Transcripts | Unity Catalog Volum |  None | None | Json Data Ingestion with [DLT (python)](./Data_Ingestion/DLT-Transcript-Policy-Ingestion-Python.py), [DLT (SQL)](./Data_Ingestion/DLT-Transcript-Policy-Ingestion-SQL.sql)||
| PDF Doc (with tables) | Unity Catalog Volum | [Unstructured chunking strategy](https://docs.unstructured.io/open-source/core-functionality/chunking) | [Unstructured](https://docs.unstructured.io/open-source/introduction/overview) | [PDF Doc Ingestion](./Data_Ingestion/PDF-Text-Table-Ingestion.py)  |
| Image Doc (text extraction) |  Unity Catalog Volum | [Unstructured chunking strategy](https://docs.unstructured.io/open-source/core-functionality/chunking) | [Unstructured](https://docs.unstructured.io/open-source/introduction/overview) | [Image Doc Ingestion](./Data_Ingestion/Image-Text-Ingestion.py) |


## End to End GenAI Application Architecture Patterns 


| Input Data            | Model                          | Tasks                                                            | GenAI Use Case           | Orchestration         | Customer Persona | PoC Template                                                                                                  |
|-----------------------|--------------------------------|------------------------------------------------------------------|--------------------------|-----------------------|----------------------|---------------------------------------------------------------------------------------------------------------|
| JSON Text Transcripts | Foundation LLM (e.g. Llama3p1) | ummarization, Sentiment, classification                          | AI Function, DBSQL Agent | DLT, LangChain        | Data Analyist, Data Scientist | [Call Center Transcript Analytics with AI](./call_center_genAI_apps/transcripts_analytics_with_AI/)           |
| JSON Text Transcripts | Foundation LLM (e.g. Llama3p1) | ummarization, Sentiment                                          | RAG                      | DLT, LangChain        | Data Scientist, MLE, Data Engineer | [Call Center Transcript RAG Apps](./call_center_genAI_apps/transcripts_summarization_rag_chatbot/)            |
| wav Audio             | Foundation LLM (e.g. Llama3p1) | Speech Transcription, Summarization, Sentiment                   | RAG                      | DLT, LangChain        | Data Scientist, MLE, Data Engineer | [Call Center Audio to Text RAG Apps](./call_center_genAI_apps/audio_transcription_summariztaion_rag_chatbot/) |
| PDF Documents         | Foundation LLM (e.g. Llama3p1) | Unstructured Data Processing, Name Entity Recognition | NER | DLT, Function calling | Data Scientist, MLE, Data Engineer | [PDF_Doc Ingestion](./Data_Ingestion/)                                                 |

 

# When to Use

You have a business use case that can potentially apply generative AI technology and fall into one of the [PoC accelerator template](#poc-accelerator-templates). You have access to a unity catalog enabled Databricks Workspace.

You may have some **existing data** available in the workspace to use as input data. If you don't have any data, the PoC accelerator templates contains synthetic sample datasets to enable the demonstration of genAI application's functionalities

## Getting Started

Clone this repo and add the repo to your Databricks Workspace. Refer to [Databricks Repo Setup](https://docs.databricks.com/en/repos/repos-setup.html) for instuctions on how to create Databricks repo on your workspace

1. Got into the folder of the selected [PoC accelerator template](#poc-accelerator-templates)
2. Review the architecture diagram in the README
3. Start with the `instruction` notebook
4. Follow the instructions in the `instruction` notebook.
5. Most of notebook can run by click `Run/Run ALL` but some may require additional steps of using databricks UI so be sure to read the instruction


# Resources

* [Databricks GenAI Cookbook](https://ai-cookbook.io/)
* [Databricks Foundation Model](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/foundation-models)
* [Model Serving](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/)
* [Vector Search](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/vector-search)
* [Inference Table](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/inference-tables)
* [Delta Live Table](https://learn.microsoft.com/en-us/azure/databricks/delta-live-tables/)
* [Databricks Python SDK](https://databricks-sdk-py.readthedocs.io/en/latest/#)
* [MLFlow](https://learn.microsoft.com/en-us/azure/databricks/mlflow/)

# Limitations

* The PoC accelerator template is designed for use Unit Catalog managed workspace only.
* The synthetic dataset provided by Databricks are generated algorithmatically based on assumptions and they are not real-life data.
* Delta Live Table technology from Databricks is used in some of PoC Accelerator Template, Currently the live table (a.k.a materialized view) from Delta Live Table cannot only be accessed by shared clusters, therefore, a copy of the materialized views are being used in some of notebooks. The limitation will be addressed in the future product releases

