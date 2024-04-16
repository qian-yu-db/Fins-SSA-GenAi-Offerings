# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Create a UC Volume to store synthetic datasets

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Set `reset_all_data` to `True` to clean up existing data

# COMMAND ----------

dbutils.widgets.text("reset_all_data", "false", "Reset Data")
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"

# COMMAND ----------

import os
import requests

spark.sql(f'USE CATALOG {catalog};')
spark.sql(f'USE SCHEMA {db};')
spark.sql(f'CREATE VOLUME IF NOT EXISTS {volume_name_audio};')
spark.sql(f'CREATE VOLUME IF NOT EXISTS {volume_name_policies};')
spark.sql(f'CREATE VOLUME IF NOT EXISTS {volume_name_rag};')
volume_folder_policy = f"/Volumes/{catalog}/{db}/{volume_name_policies}"
volume_folder_speech = f"/Volumes/{catalog}/{db}/{volume_name_audio}"
volume_folder_rag = f"/Volumes/{catalog}/{db}/{volume_name_rag}"
policy_sub = "Policies"
audio_sub = "audio_clips"

# COMMAND ----------

if reset_all_data:
    try:
        print("clean volume ...")
        dbutils.fs.rm(f'{volume_folder_policy}/{policy_sub}', True)
        dbutils.fs.rm(f'{volume_folder_speech}/{audio_sub}', True)
    except Exception as e:
        print(f'Could not clean folders, they might not exist? {e}')

# COMMAND ----------

import collections

def is_folder_empty(folder):
  try:
    return len(dbutils.fs.ls(folder)) == 0
  except:
    return True
  

def get_subdir_name(owner, repo, path):
    content = requests.get(f'https://api.github.com/repos/{owner}/{repo}/contents/{path}').json()
    subdirectories = [item for item in content if item['type'] == 'dir']
    return [subdir['name'] for subdir in subdirectories]
 

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

if is_folder_empty(f"{volume_folder_policy}/{policy_sub}") or is_folder_empty(f"{volume_folder_speech}/{audio_sub}"):
    download_file_from_git(dest=f'{volume_folder_policy}/{policy_sub}', 
                           owner=repo_owner, 
                           repo=repo_name,
                           path="/datasets/insurance_policies")
    
    audio_clip_subdirs = get_subdir_name(owner=repo_owner, repo=repo_name, path="datasets/call_center_audio_clips") 
    for subdir in audio_clip_subdirs:
        download_file_from_git(dest=f'{volume_folder_speech}/{audio_sub}/{subdir}', 
                            owner=repo_owner, 
                            repo=repo_name, 
                            path=f"/datasets/call_center_audio_clips/{subdir}")
else:
    print("Data already existing. To clean up, run above cell with reset_all_date=True.")

# COMMAND ----------


