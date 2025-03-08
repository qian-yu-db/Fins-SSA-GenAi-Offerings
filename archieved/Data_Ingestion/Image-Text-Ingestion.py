# Databricks notebook source
# MAGIC %md
# MAGIC # The Notebook will perform
# MAGIC
# MAGIC * Image with text Document Ingestion
# MAGIC * Parse text with Unstructured OCR
# MAGIC * Create table with text extrated
# MAGIC * Perform NER on the text

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Setup environment

# COMMAND ----------

# MAGIC %pip install -U -qqq markdownify==0.12.1 "unstructured[local-inference, all-docs]==0.14.4" unstructured-client==0.22.0 pdfminer==20191125 nltk==3.8.1 tiktoken
# MAGIC %pip install langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 pydantic==1.10.9 -qqqqq
# MAGIC %pip install mlflow-skinny mlflow mlflow[gateway] -U -qqq

# COMMAND ----------

# MAGIC %run ./helper-functions

# COMMAND ----------

install_apt_get_packages(["poppler-utils", "tesseract-ocr"])

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# Use optimizations if available
parser_debug_flag = False
dbr_majorversion = int(spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion").split(".")[0])
if dbr_majorversion >= 14:
  spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # PDF Data Ingestion

# COMMAND ----------

pipeline_config = {
    "file_format": "tiff",
    "raw_files_table_name": f"{catalog}.{db}.img_raw_files_bronze",
    "parsed_docs_table_name": f"{catalog}.{db}.img_parsed_docs_silver",
    "enriched_docs_table_name": f"{catalog}.{db}.img_enriched_docs_gold",
    "checkpoint_path": f"{CHECKPOINTS_VOLUME_PATH}/img",
}

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Bronze Layer: Ingest Images (`.tiff`) as Binary in a Delta Table

# COMMAND ----------

dbutils.widgets.text(name="clean_up_all", defaultValue="false", label="clean_up_all")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
import warnings

HOSTNAME = spark.conf.get('spark.databricks.workspaceUrl')
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
w = WorkspaceClient(host=HOSTNAME, token=TOKEN)
clean_up_all = dbutils.widgets.get("clean_up_all")

# clean up
if clean_up_all == "true":
    try:
        w.volumes.delete(f"{catalog}.{db}.{CHECKPOINTS_VOLUME_PATH.split('/')[-1]}")
        spark.sql(f"drop table if exists {pipeline_config.get('raw_files_table_name')};")
        spark.sql(f"drop table if exists {pipeline_config.get('parsed_docs_table_name')};")
        spark.sql(f"drop table if exists {pipeline_config.get('enriched_docs_table_name')};")
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during clean up")
    else:
        w.volumes.create(catalog_name=catalog, 
                         schema_name=db, 
                         name=CHECKPOINTS_VOLUME_PATH.split('/')[-1],
                         volume_type=VolumeType.MANAGED)

# COMMAND ----------

df_raw_bronze = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobfilter", f"*.{pipeline_config.get('file_format')}")
    .load(IMG_SOURCE_PATH)
)

# COMMAND ----------

df_raw_bronze.writeStream.trigger(availableNow=True).option(
    "checkpointLocation",
    f"{pipeline_config.get('checkpoint_path')}/{pipeline_config.get('raw_files_table_name').split('.')[-1]}",
).toTable(pipeline_config.get("raw_files_table_name")).awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver layer: Partition Image files using Pandas UDF function with Unstructured OSS

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col
import pandas as pd

@pandas_udf("string")
def parse_image(raw_tiffs: pd.Series) -> pd.Series:
    from unstructured.partition.image import partition_image
    import io

    def perform_partition(raw_tiff):
        tiff_bytes = io.BytesIO(raw_tiff)
        raw_tiff_elements = partition_image(
                                file=tiff_bytes,
                                languages=["eng"],
                                strategy="hi_res")

        text_content = ""
        for section in raw_tiff_elements:
            # Tables are parsed seperatly, add a \n to give the chunker a hint to split well.
            if section.category == "Table":
                if section.metadata is not None:
                    if section.metadata.text_as_html is not None:
                        # convert table to markdown
                        text_content += "\n" + section.metadata.text_as_html + "\n"
                    else:
                        text_content += " " + section.text
                else:
                    text_content += " " + section.text
            # Other content often has too-aggresive splitting, merge the content
            else:
                text_content += " " + section.text
        return text_content
    return pd.Series([perform_partition(raw_tiff) for raw_tiff in raw_tiffs])

# COMMAND ----------

df_raw_bronze = spark.readStream.table(
    pipeline_config.get("raw_files_table_name")
).select("path", "content")

# COMMAND ----------

df_parsed_silver = (
    df_raw_bronze.withColumn("parsing", parse_image("content"))
).drop("content")

# COMMAND ----------

silver_query = (
    df_parsed_silver.writeStream.trigger(availableNow=True)
    .option(
        "checkpointLocation",
        f"{pipeline_config.get('checkpoint_path')}/{pipeline_config.get('parsed_docs_table_name').split('.')[-1]}",
    )
    .toTable(pipeline_config.get("parsed_docs_table_name"))
).awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Gold Layer: Enrich the silver table
# MAGIC
# MAGIC - Add token count, timestamp metadata

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col
import pandas as pd

@pandas_udf("int")
def count_tokens(txts: pd.Series) -> pd.Series:
    import tiktoken

    def token_count(txt, encoding_name="cl100k_base"):
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(txt, disallowed_special=())
        return len(tokens)
    return pd.Series([token_count(txt) for txt in txts])

# COMMAND ----------

df_parsed_silver = spark.readStream.table(
    pipeline_config.get("parsed_docs_table_name")
)

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, date_format, col

df_enriched_gold = df_parsed_silver \
    .withColumn("token_count", count_tokens(col("parsing"))) \
    .withColumn("process_time", date_format(current_timestamp(), "yyyy-MM-dd HH:mm:ss"))

# COMMAND ----------

gold_query = (
    df_enriched_gold.writeStream.trigger(availableNow=True)
    .option(
        "checkpointLocation",
        f"{pipeline_config.get('checkpoint_path')}/{pipeline_config.get('enriched_docs_table_name').split('.')[-1]}",
    )
    .toTable(pipeline_config.get("enriched_docs_table_name"))
).awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Extract Information from Image Doc
# MAGIC
# MAGIC - The documents are engineering resumes
# MAGIC - Extract Job Title and Skills

# COMMAND ----------

df_gold_enriched_docs = spark.table(pipeline_config.get("enriched_docs_table_name"))
display(df_gold_enriched_docs)

# COMMAND ----------

from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatDatabricks
from langchain_core.prompts import PromptTemplate
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field, validator
import mlflow

## Enable MLflow Tracing
mlflow.langchain.autolog()

class Person(BaseModel):
    title: str = Field(description="The Job Title on the Resume")
    skills: str = Field(description="The list of skills the Resume contains")
    experience1: str = Field(description="The most recent experience time range and employee on the resume")
    experience2: str = Field(description="The second most recent experience time range and employee on the resume")
    experience3: str = Field(description="The third most recent experience time range and employee on the resume")

parser = JsonOutputParser(pydantic_object=Person)
format_instructions = parser.get_format_instructions()

# COMMAND ----------

prompt = PromptTemplate(
    template="""You are an assistant that extract information. Extract title, skills, the most recent work experiences time and employee from the ***query***. Answer only with those information. Do not answer if you cannot find the information. \n{format_instructions}\n{query}\n
    """,
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

# Our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint="databricks-meta-llama-3-70b-instruct",
    extra_params={"temperature": 0.0, "max_tokens": 1500}
)

chain = prompt | model | parser

# COMMAND ----------

from PIL import Image
import re

resume1 = df_gold_enriched_docs.select("path").collect()[0][0]
print(resume1)
Image.open(re.sub(r'dbfs:', '', resume1))

# COMMAND ----------

resume1_txt = df_gold_enriched_docs.select("parsing").collect()[0][0]
answer = chain.invoke(resume1_txt)
answer

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Use Pandas UDF Apply extract to all extracted text in the table

# COMMAND ----------

from pyspark.sql.types import MapType, StringType
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf(returnType=MapType(StringType(), StringType()))
def extract_user_info(txts: pd.Series) -> pd.Series:
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    
    def extract_info_LLM(txt):
        class Person(BaseModel):
            title: str = Field(description="The Job Title on the Resume")
            skills: str = Field(description="The list of skills the Resume contains")
            experience1: str = Field(description="The most recent experience time range and employee on the resume")
            experience2: str = Field(description="The second most recent experience time range and employee on the resume")
            experience3: str = Field(description="The third most recent experience time range and employee on the resume")

        parser = JsonOutputParser(pydantic_object=Person)
        format_instructions = parser.get_format_instructions()
        prompt = PromptTemplate(
            template="""You are an assistant that extract information. Extract title, skills, the most recent work experiences time and employee from the ***query***. Answer only with those information. Do not answer if you cannot find the information. \n{format_instructions}\n{query}\n
            """,
            input_variables=["query"],
            partial_variables={"format_instructions": format_instructions},
        )

        # Our foundation model answering the final prompt
        model = ChatDatabricks(
            endpoint="databricks-meta-llama-3-70b-instruct",
            extra_params={"temperature": 0.0, "max_tokens": 1500}
        )

        chain = prompt | model | parser
        return chain.invoke(txt)
    return pd.Series([extract_info_LLM(txt) for txt in txts])

# COMMAND ----------

df_gold_doc_NER = df_gold_enriched_docs.withColumn('NER_from_LLM', extract_user_info(df_gold_enriched_docs['parsing']))
display(df_gold_doc_NER)

# COMMAND ----------

df_gold_doc_NER.write \
    .mode("overwrite") \
    .option("overwriteSchema", "True") \
    .saveAsTable("img_text_NER")

# COMMAND ----------


