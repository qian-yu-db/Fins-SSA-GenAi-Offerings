# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Create a UC Volume to store synthetic datasets

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Set `reset_all_data` to `true` to clean up existing data

# COMMAND ----------

dbutils.widgets.text("reset_all_data", "false", "Reset Data")
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"
if reset_all_data:
    print("We will delete and recreate all data!")

# COMMAND ----------

import os
import requests

spark.sql(f'USE CATALOG {catalog};')
spark.sql(f'USE SCHEMA {schema};')
spark.sql(f'CREATE VOLUME IF NOT EXISTS {volume_name_transcripts};')
spark.sql(f'CREATE VOLUME IF NOT EXISTS {volume_name_policies};')
volume_folder_policy = f"/Volumes/{catalog}/{schema}/{volume_name_policies}"
volume_folder_speech = f"/Volumes/{catalog}/{schema}/{volume_name_transcripts}"
policy_sub = "policies"
transcript_sub = "transcripts_json_data"

# COMMAND ----------

if reset_all_data:
    try:
        dbutils.fs.rm(f'{volume_folder_policy}', True)
        dbutils.fs.rm(f'{volume_folder_speech}', True)
    except Exception as e:
      print(f'Could not clean folders, they might not exist? {e}')

# COMMAND ----------

import collections

def is_folder_empty(folder):
  try:
    return len(dbutils.fs.ls(folder)) == 0
  except:
    return True
 
def download_file_from_git(dest, owner, repo, path):
    def download_file(url, destination):
      local_filename = url.split('/')[-1]
      # NOTE the stream=True parameter below
      with requests.get(url, stream=True) as r:
          r.raise_for_status()
          print('saving '+destination+'/'+local_filename)
          with open(destination+'/'+local_filename, 'wb') as f:
              for chunk in r.iter_content(chunk_size=8192): 
                  f.write(chunk)
      return local_filename

    if not os.path.exists(dest):
      os.makedirs(dest)
    from concurrent.futures import ThreadPoolExecutor
    files = requests.get(f'https://api.github.com/repos/{owner}/{repo}/contents{path}').json()
    files = [f['download_url'] for f in files if 'NOTICE' not in f['name']]
    def download_to_dest(url):
         download_file(url, dest)
    with ThreadPoolExecutor(max_workers=10) as executor:
        collections.deque(executor.map(download_to_dest, files))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Download Data from Github to Unity Catalog Volumes

# COMMAND ----------

repo_owner = 'qian-yu-db'
repo_name = 'Fins-SSA-GenAi-Offerings'

if is_folder_empty(f"{volume_folder_policy}/{policy_sub}") or is_folder_empty(f"{volume_folder_speech}/{transcript_sub}"):
    download_file_from_git(dest=f'{volume_folder_policy}/{policy_sub}', 
                           owner=repo_owner, 
                           repo=repo_name, 
                           path="/datasets/insurance_policies")
    download_file_from_git(dest=f'{volume_folder_speech}/{transcript_sub}', 
                           owner=repo_owner, 
                           repo=repo_name, 
                           path="/datasets/call_center_audio_transcripts")
else:
    print("Data already existing. To clean up, run above cell with reset_all_date=True.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Next Step: Create a DLT Pipeline with [Notebook 01-DLT-Tanscript-Policy]($./01-DLT-Transcript-Policy)
