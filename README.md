# Fins SSA GenAI Offering <!-- omit in toc -->

Financial Service Enterprise AI Applications PoC Accelerators

# Table of Content <!-- omit in toc -->
- [Requirements](#requirements)
- [PoC Accelerator Assets](#poc-accelerator-assets)
- [When to Use](#when-to-use)
- [Getting Started](#getting-started)
- [Additional Resources](#additional-resources)
- [Limitations](#limitations)

# Requirements

* Databricks workspace with Serverless and Unity Catalog enabled

# PoC Accelerator Assets

## Insurance Call Center AI Agents


| Business Problem                           | Databricks AI Capabilities                                              | Input Data  | PoC Accelerator Notebooks                                                                           |
|--------------------------------------------|-------------------------------------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------|
| Insurance Call Center Operation Automation | Foundation Model API, AI Function for Batch Inference, Agent Framework  | Transcripts (JSON Text)  | [Insurance Call Center AI System](./insurance_call_center_AI_system/) |

 

# When to Use

- You have Business Usee Case Matching
- You have access to a unity catalog enabled Databricks Workspace.
- If you do not have ready-to-use datasets. You can use the synthetic datasets provided in [datasets](./datasets) folder.

# Getting Started

Clone this repo and add the repo to your Databricks Workspace. Refer to [Databricks Repo Setup](https://docs.databricks.com/en/repos/repos-setup.html) for instuctions on how to create Databricks repo on your workspace

1. Review the architecture diagram of desired PoC Accelerator
3. Follow the instructions in README
4. Most of notebooks can run by click `Run/Run ALL` but some may require additional steps of using databricks UI so be sure to read the instruction


# Additional Resources

* [Databricks AI Functions](https://docs.databricks.com/aws/en/large-language-models/ai-functions)
* [Building genAI Apps with Databricks](https://docs.databricks.com/aws/en/generative-ai/agent-framework/build-genai-apps) 
* [Databricks GenAI Cookbook](https://ai-cookbook.io/)
* [Databricks Foundation Model](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/foundation-models)
* [Model Serving](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/)
* [Vector Search](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/vector-search)
* [Inference Table](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/inference-tables)
* [Delta Live Table](https://learn.microsoft.com/en-us/azure/databricks/delta-live-tables/)
* [Databricks Python SDK](https://databricks-sdk-py.readthedocs.io/en/latest/#)
* [MLFlow](https://learn.microsoft.com/en-us/azure/databricks/mlflow/)

# Limitations

* The PoC accelerators are designed for use Unit Catalog managed workspace only.
* The synthetic dataset provided by Databricks are generated algorithmatically based on assumptions and they are not real-life data.