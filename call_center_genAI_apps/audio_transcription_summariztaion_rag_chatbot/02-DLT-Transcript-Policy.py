# Databricks notebook source
# MAGIC %md
# MAGIC # Assumptions:
# MAGIC
# MAGIC User already transcribed the audio and save the transcription to a JSON file with `POLICY_NO` as metadata
# MAGIC
# MAGIC # This delta live table will perform
# MAGIC
# MAGIC * Perform cleaning and transformation of raw transcript table 
# MAGIC * Perform cleaning and transformation of raw policy table
# MAGIC * Join the transcripts and policy tables with policy number
# MAGIC
# MAGIC

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, concat, col
from pyspark.sql import types as T

catalog = "qyu"
db = "test"
transcript_table = "customer_transcripts"
volume_name_policies = "volume_policies"
volume_name_speech = "volume_speech"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Bronze: 
# MAGIC
# MAGIC * `raw_polcy`: Use Auto Loader to ingest the raw policy data csv files
# MAGIC * `raw_transcript`: Use Auto Loader to ingest the raw transcript data json files

# COMMAND ----------

@dlt.table(
    name="raw_policy",
    comment="Policy data loaded from csv files.")
def raw_policy():
    return (
      spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "csv")
            .option("cloudFiles.schemaHints", "ZIPCODE int")
            .option("cloudFiles.inferColumnTypes", "true")
            .load(f"/Volumes/{catalog}/{db}/{volume_name_policies}/Policies"))

# COMMAND ----------

@dlt.table(
    name="raw_transcript",
    comment="Transcript data loaded from json files.")
def raw_transcript():
    return (
      spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "json")
            .option("cloudFiles.inferColumnTypes", "true")
            .load(f"/Volumes/{catalog}/{db}/{volume_name_speech}/transcript_saved"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver: 
# MAGIC
# MAGIC * `policy`: Perform data cleaning and data type changes
# MAGIC * `transcript`: Perform data cleaning, filtering

# COMMAND ----------

@dlt.table(
    name="policy",
    comment="clean raw policy data table"
)
@dlt.expect_all({"valid_policy_number": "POLICY_NO IS NOT NULL"})
def policy():
    # Read the staged policy records into memory
    return (dlt.readStream("raw_policy")
                .withColumn("premium", F.abs(col("premium")))
                # Reformat the incident date values
                .withColumn("pol_eff_date", F.to_date(col("pol_eff_date"), "dd-MM-yyyy"))
                .withColumn("pol_expiry_date", F.to_date(col("pol_expiry_date"), "dd-MM-yyyy"))
                .withColumn("pol_issue_date", F.to_date(col("pol_issue_date"), "dd-MM-yyyy"))
                .withColumn("address", concat(col("BOROUGH"), lit(", "), col("ZIP_CODE").cast("string")))
    )

# COMMAND ----------

@dlt.table(
    name="transcript",
    comment="clean raw transcript data table"
)
@dlt.expect_all({"valid_policy_number": "POLICY_NO IS NOT NULL"})
def transript():
    return (dlt.readStream("raw_transcript")
            .select("POLICY_NO", "transcript", "datetime_record"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold: 
# MAGIC
# MAGIC * `transcript_enriched`
# MAGIC   * Aggregate transcripts by `POLICY_NO` so it is in one conversation
# MAGIC   * Join policy using `POLICY_NO` to enrich meta-data

# COMMAND ----------

@dlt.table(
    name="transcript_enriched",
    comment="createn enriched transcript table with meta data from policy"
)
def transcript_enriched():
    transcript_cleaned = dlt.read("transcript")
    transcript_renamed = transcript_cleaned.select(
        col("POLICY_NO"), col("transcript"), col("datetime_record")
    )
    policy_cleaned = dlt.read("policy")
    return transcript_renamed.join(policy_cleaned, on="POLICY_NO")
