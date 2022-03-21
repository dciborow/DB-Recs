# Databricks notebook source
# MAGIC %md 
# MAGIC ### News Recommendation ALS w/ AML Example Databricks Notebook
# MAGIC ##### by Daniel Ciborowski, dciborow@microsoft.com
# MAGIC 
# MAGIC ##### Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC ##### Licensed under the MIT License.
# MAGIC 
# MAGIC ##### Setup
# MAGIC 1. Create new Cluster, DB 4.1, Spark 2.3.0, Python3
# MAGIC 1. (Optional for Ranking Metrics) From Maven add to cluster the following jar: Azure:mmlspark:0.15
# MAGIC 1. Cosmos DB Uber Jar - https://repo1.maven.org/maven2/com/microsoft/azure/azure-cosmosdb-spark_2.3.0_2.11/1.2.7/azure-cosmosdb-spark_2.3.0_2.11-1.2.7-uber.jar
# MAGIC 
# MAGIC ##### This notebook is broken down into four sections.
# MAGIC 1. Service Creation
# MAGIC 1. Training
# MAGIC 1. Scoring
# MAGIC 1. Operationalization
# MAGIC 
# MAGIC ##### The following Azure services will be deployed into a new or existing resource group.
# MAGIC 1. [ML Service](https://docs.databricks.com/user-guide/libraries.html)
# MAGIC 1. [Cosmos DB](https://azure.microsoft.com/en-us/services/cosmos-db/)
# MAGIC 1. [Container Registery](https://docs.microsoft.com/en-us/azure/container-registry/)
# MAGIC 1. [Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/)
# MAGIC 1. [Application Insights](https://azure.microsoft.com/en-us/services/monitor/)
# MAGIC 1. Storage Account
# MAGIC 1. Key Vault
# MAGIC 
# MAGIC In a news recommendation scenario, items have an active lifespan when they should be recommended. After this time has expired old stories are not recommended, and new news stories replace the expired ones. When recommending new stories, only active stories should be recommended. This example shows how to train a model using historical data, and make recommendations for the latest news stories.
# MAGIC 
# MAGIC 
# MAGIC New Recommendation Dataset can be found here. http://reclab.idi.ntnu.no/dataset/
# MAGIC 
# MAGIC ##### Citation
# MAGIC Gulla, J. A., Zhang, L., Liu, P., Özgöbek, Ö., & Su, X. (2017, August). The Adressa dataset for news recommendation. In Proceedings of the International Conference on Web Intelligence (pp. 1042-1048). ACM. 

# COMMAND ----------

import pandas as pd
import random

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.sql.functions import col, collect_list

# COMMAND ----------

# MAGIC %md
# MAGIC # I. Service Creation

# COMMAND ----------

from azure.common.client_factory import get_client_from_cli_profile

import azureml.core
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment


from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Row

import numpy as np
import os
import pandas as pd
import pprint
import shutil
import time, timeit
import urllib
import yaml

# Check core SDK version number - based on build number of preview/master.
print("SDK version:", azureml.core.VERSION)

prefix = "dcib_igor_"
subscription_id = '03909a66-bef8-4d52-8e9a-a346604e0902'
data = 'news'

workspace_region = "westus2"
resource_group = f'{prefix}_{data}'
workspace_name = f'{prefix}_{data}_aml'
experiment_name = f'{data}_als_Experiment'
aks_name = "dcibigoraks"
service_name = "dcibigoraksals"

# import the Workspace class and check the azureml SDK version
# exist_ok checks if workspace exists or not.
ws = Workspace.create(name = workspace_name,
                      subscription_id = subscription_id,
                      resource_group = resource_group, 
                      location = workspace_region,
                      exist_ok=True)

# persist the subscription id, resource group name, and workspace name in aml_config/config.json.
ws.write_config()

# start a training run by defining an experiment
myexperiment = Experiment(ws, experiment_name)
root_run = myexperiment.start_logging()


# COMMAND ----------

# MAGIC %md
# MAGIC # II. Training

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

data = spark.read.json("wasb://sampledata@dcibviennadata.blob.core.windows.net/one_week.json") \
  .cache()

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline, PipelineModel

df = data \
  .filter(col("sessionStart") != 'true') \
  .filter(col("sessionStop") != 'true') \
  .filter(col("url") != "http://adressa.no") \
  .filter(col("activeTime") > 10) \
  .select("userId","url", "activeTime", "time") \
  .cache()

indexerContacts = StringIndexer(inputCol='userId', outputCol='userIdIndex', handleInvalid='keep').fit(df)
indexerRules = StringIndexer(inputCol='url', outputCol='itemIdIndex', handleInvalid='keep').fit(df)

ratings = indexerRules.transform(indexerContacts.transform(df)) \
  .select("userIdIndex","itemIdIndex","activeTime","time") \
  .withColumnRenamed('userIdIndex',"userId") \
  .withColumnRenamed('itemIdIndex',"itemId") \
  .withColumnRenamed('activeTime',"rating") \
  .withColumnRenamed('time',"timestamp") \
  .cache()

display(ratings.select('userId','itemId','rating','timestamp').orderBy('userId','itemId'))

# COMMAND ----------

display(ratings.select('userId','itemId','rating','timestamp').orderBy('userId','itemId'))

# COMMAND ----------

# Build the recommendation model using ALS on the rating data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
algo = ALS(userCol="userId", itemCol="itemId", implicitPrefs=True, coldStartStrategy="drop")
model = algo.fit(ratings)

# COMMAND ----------

# Evaluate the model by computing the RMSE on the rating data
predictions = model.transform(ratings)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

root_run.log('rmse', rmse)
print(f"Root-mean-square error = {str(rmse)}")

# COMMAND ----------

# Evaluate the model by computing ranking metrics on the rating data
from mmlspark.RankingAdapter import RankingAdapter
from mmlspark.RankingEvaluator import RankingEvaluator

output = RankingAdapter(mode='allUsers', k=5, recommender=algo) \
  .fit(ratings) \
  .transform(ratings)

metrics = ['ndcgAt','map','recallAtK','mrr','fcp']
metrics_dict = {
    metric: RankingEvaluator(k=3, metricName=metric).evaluate(output)
    for metric in metrics
}

for k, v in metrics_dict.items():
    root_run.log(k, v)    

metrics_dict    

# COMMAND ----------

# Recommend Subset Wrapper
def recommendSubset(self, df, timestamp):
  def Func(lines):
      out = [(lines[1][i],lines[2][i]) for i in range(len(lines[1]))]
      return lines[0], out

  tup = StructType([
    StructField('itemId', IntegerType(), True),
    StructField('rating', FloatType(), True)
  ])
  array_type = ArrayType(tup, True)
  active_items = df.filter(col("timestamp") > timestamp).select("itemId").distinct()
  users = df.select("userId").distinct()

  users_active_items = users.crossJoin(active_items)
  scored = self.transform(users_active_items)

  recs = scored \
    .groupBy(col('userId')) \
    .agg(collect_list(col("itemId")),collect_list(col("prediction"))) \
    .rdd \
    .map(Func) \
    .toDF() \
    .withColumnRenamed("_1","userId") \
    .withColumnRenamed("_2","recommendations") \
    .select(col("userId"),col("recommendations").cast(array_type))

  return recs

import pyspark
pyspark.ml.recommendation.ALSModel.recommendSubset = recommendSubset

# COMMAND ----------

recs = model.recommendSubset(ratings, 1483747200) \
  .cache()

recs.take(5)

# COMMAND ----------

# MAGIC %md
# MAGIC In order to turn new stories from cold items, to warm items, 1% of the recommendations servered should include a random new (cold) story. This population should also be used to provide a baseline to measure the online model performance.

# COMMAND ----------

# MAGIC %%writefile recommend.py
# MAGIC 
# MAGIC import pyspark
# MAGIC from pyspark.ml.recommendation import ALS
# MAGIC 
# MAGIC # Recommend Subset Wrapper
# MAGIC def recommendSubset(self, df, timestamp):
# MAGIC   def Func(lines):
# MAGIC     out = []
# MAGIC     for i in range(len(lines[1])):
# MAGIC       out += [(lines[1][i],lines[2][i])]
# MAGIC     return lines[0], out
# MAGIC 
# MAGIC   tup = StructType([
# MAGIC     StructField('itemId', IntegerType(), True),
# MAGIC     StructField('rating', FloatType(), True)
# MAGIC   ])
# MAGIC   array_type = ArrayType(tup, True)
# MAGIC   active_items = df.filter(col("timestamp") > timestamp).select("itemId").distinct()
# MAGIC   users = df.select("userId").distinct()
# MAGIC 
# MAGIC   users_active_items = users.crossJoin(active_items)
# MAGIC   scored = self.transform(users_active_items)
# MAGIC 
# MAGIC   recs = scored \
# MAGIC     .groupBy(col('userId')) \
# MAGIC     .agg(collect_list(col("itemId")),collect_list(col("prediction"))) \
# MAGIC     .rdd \
# MAGIC     .map(Func) \
# MAGIC     .toDF() \
# MAGIC     .withColumnRenamed("_1","userId") \
# MAGIC     .withColumnRenamed("_2","recommendations") \
# MAGIC     .select(col("userId"),col("recommendations").cast(array_type))
# MAGIC 
# MAGIC   return recs
# MAGIC 
# MAGIC import pyspark
# MAGIC pyspark.ml.recommendation.ALSModel.recommendSubset = recommendSubset
# MAGIC 
# MAGIC #Implement this function
# MAGIC def recommend(historic, timestamp):   
# MAGIC   algo = ALS(userCol="userId", itemCol="itemId", implicitPrefs=True, coldStartStrategy="drop")
# MAGIC   model = algo.fit(historic)  
# MAGIC   recs = model.recommendSubset(historic. timestamp)
# MAGIC   return recs

# COMMAND ----------

root_run.upload_file("outputs/recommend.py",'recommend.py')
root_run.complete()

# COMMAND ----------

# MAGIC %md
# MAGIC # III. Scoring

# COMMAND ----------

with open('recommend.py', 'r') as myfile:
    data=myfile.read()

exec(data)

recs = recommend(ratings,1483747200)
# display(recs.orderBy('userId'))

# COMMAND ----------

# Register as model
from azureml.core.model import Model
mymodel = Model.register(model_path = 'recommend.py', # this points to a local file
                       model_name = 'als', # this is the name the model is registered as, am using same name for both path and name.                 
                       description = "ADB trained model by Dan",
                       workspace = ws)

print(mymodel.name, mymodel.description, mymodel.version)

# COMMAND ----------

from azureml.core.model import Model

mymodel = Model.list(ws)[0]
mymodel.download('./o16n/',exists_ok=True)
print(mymodel.name, mymodel.description, mymodel.version)

with open('./o16n/recommend.py', 'r') as myfile:
    data=myfile.read()

exec(data)

recs = recommend(ratings,1692455636)
display(recs.orderBy('userId'))

# COMMAND ----------

account_name = "movies-ds-sql"
endpoint = f"https://{account_name}.documents.azure.com:443/"
master_key = "KUEKaC3ULYBvZUOLk4IaBKqXaNMS9TCSSwvSCezcOVMS01jalvEPk9oYxENu5vElabUeaYIyIppzDtIbvQJsYQ=="

writeConfig = {
  "Endpoint": endpoint,
  "Masterkey": master_key,
  "Database": 'recommendations',
  "Collection": 'news',
  "Upsert": "true"
}

# recs \
#   .withColumn("id",recs['userid'].cast("string")) \
#   .select("id", "recommendations.itemid")\
#   .write \
#   .format("com.microsoft.azure.cosmosdb.spark") \
#   .mode('overwrite') \
#   .options(**writeConfig) \
#   .save()

# COMMAND ----------

# MAGIC %md
# MAGIC # IV. Operationalization

# COMMAND ----------

# MAGIC %%writefile score_sparkml.py
# MAGIC 
# MAGIC import json
# MAGIC def init(local=False):
# MAGIC     global client, collection
# MAGIC     try:
# MAGIC       # Query them in SQL
# MAGIC       import pydocumentdb.document_client as document_client
# MAGIC 
# MAGIC       MASTER_KEY = '{key}'
# MAGIC       HOST = '{endpoint}'
# MAGIC       DATABASE_ID = "{database}"
# MAGIC       COLLECTION_ID = "{collection}"
# MAGIC       database_link = 'dbs/' + DATABASE_ID
# MAGIC       collection_link = database_link + '/colls/' + COLLECTION_ID
# MAGIC       
# MAGIC       client = document_client.DocumentClient(HOST, {'masterKey': MASTER_KEY})
# MAGIC       collection = client.ReadCollection(collection_link=collection_link)
# MAGIC     except Exception as e:
# MAGIC       collection = e
# MAGIC def run(input_json):      
# MAGIC 
# MAGIC     try:
# MAGIC       import json
# MAGIC 
# MAGIC       id = json.loads(json.loads(input_json)[0])['id']
# MAGIC       query = {'query': 'SELECT * FROM c WHERE c.id = "' + str(id) +'"' } #+ str(id)
# MAGIC 
# MAGIC       options = {}
# MAGIC 
# MAGIC       result_iterable = client.QueryDocuments(collection['_self'], query, options)
# MAGIC       result = list(result_iterable);
# MAGIC   
# MAGIC     except Exception as e:
# MAGIC         result = str(e)
# MAGIC     return json.dumps(str(result)) #json.dumps({{"result":result}})

# COMMAND ----------

with open('score_sparkml.py', 'r') as myfile:
    score_sparkml=myfile.read()

import json
score_sparkml = score_sparkml.replace("{key}",writeConfig['Masterkey']).replace("{endpoint}",writeConfig['Endpoint']).replace("{database}",writeConfig['Database']).replace("{collection}",writeConfig['Collection'])

exec(score_sparkml)

# COMMAND ----------

# MAGIC %%writefile myenv_sparkml.yml
# MAGIC 
# MAGIC name: myenv
# MAGIC channels:
# MAGIC   - defaults
# MAGIC dependencies:
# MAGIC   - pip:
# MAGIC     - numpy==1.14.2
# MAGIC     - scikit-learn==0.19.1
# MAGIC     - pandas
# MAGIC     # Required packages for AzureML execution, history, and data preparation.
# MAGIC     - --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-release/Preview/E7501C02541B433786111FE8E140CAA1
# MAGIC     - azureml-core
# MAGIC     - pydocumentdb

# COMMAND ----------

models = [mymodel]
runtime = "spark-py"
conda_file = 'myenv_sparkml.yml'
driver_file = "score_sparkml.py"

# image creation
from azureml.core.image import ContainerImage
myimage_config = ContainerImage.image_configuration(execution_script = driver_file, 
                                    runtime = runtime, 
                                    conda_file = conda_file)

image = ContainerImage.create(name = "news-als",
                                # this is the model object
                                models = [mymodel],
                                image_config = myimage_config,
                                workspace = ws)

# Wait for the create process to complete
image.wait_for_creation(show_output = True)

# COMMAND ----------

#create AKS compute
#it may take 20-25 minutes to create a new cluster

from azureml.core.compute import AksCompute, ComputeTarget

# Use the default configuration (can also provide parameters to customize)
prov_config = AksCompute.provisioning_configuration()

# Create the cluster
aks_target = ComputeTarget.create(workspace = ws, 
                                  name = aks_name, 
                                  provisioning_configuration = prov_config)

aks_target.wait_for_completion(show_output = True)

print(aks_target.provisioning_state)
print(aks_target.provisioning_errors)

# COMMAND ----------

from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.image import ContainerImage

#Set the web service configuration (using default here with app insights)
aks_config = AksWebservice.deploy_configuration(enable_app_insights=True)

# Webservice creation using single command, there is a variant to use image directly as well.
try:
  aks_service = Webservice.deploy_from_image(
    workspace=ws, 
    name=service_name,
    deployment_config = aks_config,
    image = image,
    deployment_target = aks_target
      )
  aks_service.wait_for_deployment(show_output=True)
except Exception:
    aks_service = Webservice.list(ws)[0]


# COMMAND ----------

import urllib
import time
import json

scoring_url = aks_service.scoring_uri
service_key = aks_service.get_keys()[0]

input_data = '["{\\"id\\":\\"1\\"}"]'.encode()

req = urllib.request.Request(scoring_url,data=input_data)
req.add_header("Authorization", f"Bearer {service_key}")
req.add_header("Content-Type","application/json")

tic = time.time()
with urllib.request.urlopen(req) as result:
    res = result.readlines()
    print(res)

toc = time.time()
t2 = toc - tic
print("Full run took %.2f seconds" % (toc - tic))

# COMMAND ----------


