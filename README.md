# Fins SSA GenAI Offering

The goal of this repo is to develop and deliver GenAI solutions to enable and accelerate customer GenAI POC Projects with Databricks


# Table of Contents:

- [Fins SSA GenAI Offering](#fins-ssa-genai-offering)
- [Table of Contents:](#table-of-contents)
- [Requirements](#requirements)
- [POC Accelerator Projects](#poc-accelerator-projects)
- [How to Use](#how-to-use)
- [Limitations](#limitations)

# Requirements

* Databricks workspace with Serverless and Unity Catalog enabled
* Python 3.9+


# POC Accelerator Projects

| Input Data | Chat Model | Tasks Performed | GenAI Use Case | Orchestrator | Business Application | Project Notebooks |
|-------------|-----------|-----------------|----------------|--------------|----------------------|-------------------|
| JSON Text Transcripts | Foundation LLM (e.g. DBRX) | Summarization, Sentiment | RAG | DLT, LangChain | Customer Call Center | [Call Center Transcript RAG Apps](./call_center_genAI_apps/transcripts_summarization_rag_chatbot/) |
| wav Audio | Foundation LLM (e.g. DBRX) | Speech Transcription, Summarization, Sentiment | RAG | DLT, LangChain | Customer Call Center | [Call Center Audio to Text RAG Apps](./call_center_genAI_apps/call_center_genAI_apps/audio_transcription_summariztaion_rag_chatbot/) |
| PDF Doc (machine generated) |           |           |                    |            |             |           |
| PDF Doc (with tables) |           |           |                    |            |             |           |


# How to Use


# Limitations

