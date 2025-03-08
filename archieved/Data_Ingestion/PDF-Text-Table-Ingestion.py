# Databricks notebook source
# MAGIC %md
# MAGIC # The Notebook will perform
# MAGIC
# MAGIC * PDF Document Ingestion
# MAGIC * Parse text and tables
# MAGIC * Create table with text including tables in HTML format
# MAGIC * Create a table with tables only from each pdf documents

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Setup environment

# COMMAND ----------

# MAGIC %pip install -U -qqq markdownify==0.12.1 "unstructured[local-inference, all-docs]==0.14.4" unstructured-client==0.22.0 pdfminer==20191125 nltk==3.8.1 tiktoken

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
    "file_format": "pdf",
    "raw_files_table_name": f"{catalog}.{db}.pdf_raw_files_bronze",
    "parsed_docs_table_name": f"{catalog}.{db}.pdf_parsed_docs_silver",
    "enriched_docs_table_name": f"{catalog}.{db}.pdf_enriched_docs_gold",
    "extracted_tables_table_name": f"{catalog}.{db}.pdf_extracted_tables_gold",
    "checkpoint_path": f"{CHECKPOINTS_VOLUME_PATH}/pdf",
}

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Bronze Layer: Ingest PDF as Binary in a Delta Table

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
        spark.sql(f"drop table if exists {pipeline_config.get('extracted_tables_table_name')};")
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
    .load(PDF_SOURCE_PATH)
)

# COMMAND ----------

df_raw_bronze.writeStream.trigger(availableNow=True).option(
    "checkpointLocation",
    f"{pipeline_config.get('checkpoint_path')}/{pipeline_config.get('raw_files_table_name').split('.')[-1]}",
).toTable(pipeline_config.get("raw_files_table_name")).awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver layer: Partition PDF file using Pandas UDF function with Unstructured OSS

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col
import pandas as pd

@pandas_udf("string")
def process_pdfs_content(contents: pd.Series) -> pd.Series:
    from unstructured.partition.pdf import partition_pdf
    import pandas as pd
    import io
    import re

    def perform_partition(raw_doc_contents_bytes):
        pdf = io.BytesIO(raw_doc_contents_bytes)
        raw_pdf_elements = partition_pdf(
            file=pdf,
            infer_table_structure=True,
            lenguages=["eng"],
            strategy="hi_res",                                   
            extract_image_block_types=["Table"])

        text_content = ""
        for section in raw_pdf_elements:
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
    return pd.Series([perform_partition(content) for content in contents])

# COMMAND ----------

df_raw_bronze = spark.readStream.table(
    pipeline_config.get("raw_files_table_name")
).select("path", "content")

# COMMAND ----------

df_parsed_silver = (
    df_raw_bronze.withColumn("parsing", process_pdfs_content("content"))
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
# MAGIC - Extrat table to a new table

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

from pyspark.sql.functions import regexp_extract_all, lit, explode

pattern = r"(?s)(<table.*?>.*?</table>)"
df_parsed_tables_gold = df_parsed_silver.withColumn("html_tables", regexp_extract_all(col("parsing"), lit(pattern)))
df_tables_gold = df_parsed_tables_gold.select("path", explode(col("html_tables")).alias("tables"))

# COMMAND ----------

gold_tables_query = (
    df_tables_gold.writeStream.trigger(availableNow=True)
    .option(
        "checkpointLocation",
        f"{pipeline_config.get('checkpoint_path')}/{pipeline_config.get('extracted_tables_table_name').split('.')[-1]}",
    )
    .toTable(pipeline_config.get("extracted_tables_table_name"))
).awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Exam the Gold Tables

# COMMAND ----------

df_gold_enriched_docs = spark.table(pipeline_config.get("enriched_docs_table_name"))
display(df_gold_enriched_docs)

# COMMAND ----------

df_gold_extracted_tables = spark.table(pipeline_config.get("extracted_tables_table_name"))
display(df_gold_extracted_tables)

# COMMAND ----------

table = df_gold_extracted_tables.select("col").collect()[2][0]
displayHTML(table)

# COMMAND ----------

pdf = pd.read_html(table)
display(pdf[0])

# COMMAND ----------


