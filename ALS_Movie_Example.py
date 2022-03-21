# Databricks notebook source
# MAGIC %md 
# MAGIC ### ALS Movie Example Databricks Notebook
# MAGIC ##### by Daniel Ciborowski, dciborow@microsoft.com
# MAGIC 
# MAGIC ##### Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC ##### Licensed under the MIT License.
# MAGIC 
# MAGIC ##### Setup
# MAGIC 1. Create new Cluster, DB 4.1, Spark 2.3.0, Python3
# MAGIC 1. Add Azure-cli via pypi - azure-cli
# MAGIC 1. Add AzureML via Pypi - azureml-sdk[databricks]
# MAGIC 1. Add pydocumentdb via Pypi
# MAGIC 1. Add CosmosDB uber jar via maven central - https://search.maven.org/artifact/com.microsoft.azure/azure-cosmosdb-spark_2.3.0_2.11/1.2.2

# COMMAND ----------

from azure.common.client_factory import get_client_from_cli_profile
from azure.mgmt.compute import ComputeManagementClient
import azure.mgmt.cosmosdb

import azureml.core
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment

import pydocumentdb
import pydocumentdb.document_client as document_client

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

# COMMAND ----------

prefix = dbutils.widgets.get("Prefix")
data = dbutils.widgets.get("Dataset")
algo = dbutils.widgets.get("Algo")

resource_group = prefix + "_" + data
workspace_name = prefix + "_"+data+"_aml"
workspace_region = "westus2"

#Columns
userCol=dbutils.widgets.get("Column User")
itemCol=dbutils.widgets.get("Column Item")
ratingCol=dbutils.widgets.get("Column Rating")

userColIndex = userCol.replace("Id","Index")
itemColIndex = itemCol.replace("Id","Index")

#CosmosDB
location = 'westus2'
account_name = prefix + "-" + data + "-ds-sql"
DOCUMENTDB_DATABASE = "recommendations"
DOCUMENTDB_COLLECTION = "user_recommendations_" + algo

#AzureML
history_name = 'spark-ml-notebook'
model_name = data+"-"+algo+"-Recommendations.mml"
service_name = data + "-" + algo
experiment_name = data + "_"+ algo +"_Experiment"

train_data_path = data + "Train"
test_data_path = data + "Test"

subscription_id = dbutils.widgets.get("Subscription")

# COMMAND ----------

# import the Workspace class and check the azureml SDK version
# exist_ok checks if workspace exists or not.
ws = Workspace.create(name = workspace_name,
                      subscription_id = subscription_id,
                      resource_group = resource_group, 
                      location = workspace_region,
                      exist_ok=True)

# persist the subscription id, resource group name, and workspace name in aml_config/config.json.
ws.write_config()

# COMMAND ----------

def find_collection(client, dbid, id):
        database_link = 'dbs/' + dbid
        collections = list(client.QueryCollections(
            database_link,
            {
                "query": "SELECT * FROM r WHERE r.id=@id",
                "parameters": [
                    { "name":"@id", "value": id }
                ]
            }
        ))

        return len(collections) > 0
def read_collection(client, dbid, id):
        try:
                database_link = 'dbs/' + dbid
                collection_link = database_link + '/colls/{0}'.format(id)

                return client.ReadCollection(collection_link)
        except errors.DocumentDBError as e:
            if e.status_code == 404:
               print('A collection with id \'{0}\' does not exist'.format(id))
            else: 
                raise errors.HTTPFailure(e.status_code)    

def read_database(client, id):
        try:
                database_link = 'dbs/' + id

                return client.ReadDatabase(database_link)
        except errors.DocumentDBError as e:
            if e.status_code == 404:
               print('A database with id \'{0}\' does not exist'.format(id))
            else: 
                raise errors.HTTPFailure(e.status_code)  
            
def find_database(client, id):
        databases = list(client.QueryDatabases({
            "query": "SELECT * FROM r WHERE r.id=@id",
            "parameters": [
                { "name":"@id", "value": id }
            ]
        }))

        return len(databases) > 0            

client = get_client_from_cli_profile(azure.mgmt.cosmosdb.CosmosDB)

async_cosmosdb_create = client.database_accounts.create_or_update(
    resource_group,
    account_name,
    {
        'location': location,
        'locations': [{
            'location_name': location
        }]
    }
)
account = async_cosmosdb_create.result()

my_keys = client.database_accounts.list_keys(
    resource_group,
    account_name
)
master_key = my_keys.primary_master_key
endpoint = "https://" + account_name + ".documents.azure.com:443/"
#db client
client = document_client.DocumentClient(endpoint, {'masterKey': master_key})

if find_database(client, DOCUMENTDB_DATABASE) == False:
  db = client.CreateDatabase({ 'id': DOCUMENTDB_DATABASE })
else:
  db = read_database(client, DOCUMENTDB_DATABASE)
# Create collection options
options = {
    'offerThroughput': 11000
}

# Create a collection
collection_definition = { 'id': DOCUMENTDB_COLLECTION, 'partitionKey': {'paths': ['/id'],'kind': 'Hash'} }
if find_collection(client,DOCUMENTDB_DATABASE,  DOCUMENTDB_COLLECTION) ==False:
  collection = client.CreateCollection(db['_self'], collection_definition, options)
else:
  collection = read_collection(client, DOCUMENTDB_DATABASE, DOCUMENTDB_COLLECTION)
secrets = {
  "Endpoint": endpoint,
  "Masterkey": master_key,
  "Database": DOCUMENTDB_DATABASE,
  "Collection": DOCUMENTDB_COLLECTION,
  "Upsert": "true"
}
import json
with open("secrets.json", "w") as file:
    json.dump(secrets, file)

# COMMAND ----------

# Download Movie Lens
basedataurl = "http://aka.ms" 
datafile = "MovieRatings.csv"

datafile_dbfs = os.path.join("/dbfs", datafile)

if os.path.isfile(datafile_dbfs):
    print("found {} at {}".format(datafile, datafile_dbfs))
else:
    print("downloading {} to {}".format(datafile, datafile_dbfs))
    urllib.request.urlretrieve(os.path.join(basedataurl, datafile), datafile_dbfs)
    
data_all = sqlContext.read.format('csv')\
                     .options(header='true', delimiter=',', inferSchema='true', ignoreLeadingWhiteSpace='true', ignoreTrailingWhiteSpace='true')\
                     .load(datafile)    
data_all.printSchema()
display(data_all)

# COMMAND ----------

train, test = data_all.cache().randomSplit([0.75, 0.25], seed=123)

print("train ({}, {})".format(train.cache().count(), len(train.columns)))
print("test ({}, {})".format(test.cache().count(), len(test.columns)))

train_data_path_dbfs = os.path.join("/dbfs", train_data_path)
test_data_path_dbfs = os.path.join("/dbfs", test_data_path)

train.write.mode('overwrite').parquet(train_data_path)
test.write.mode('overwrite').parquet(test_data_path)
print("train and test datasets saved to {} and {}".format(train_data_path_dbfs, test_data_path_dbfs))

# COMMAND ----------

#from tqdm import tqdm
model_dbfs = os.path.join("/dbfs", model_name)

# start a training run by defining an experiment
myexperiment = Experiment(ws, experiment_name)
root_run = myexperiment.start_logging()

# COMMAND ----------

indexerContacts = StringIndexer(inputCol=userCol, outputCol=userColIndex, handleInvalid='keep').fit(data_all)
indexerRules = StringIndexer(inputCol=itemCol, outputCol=itemColIndex, handleInvalid='keep').fit(data_all)

als = ALS(maxIter=5, userCol=userColIndex, itemCol=itemColIndex, ratingCol=ratingCol, coldStartStrategy="drop")

# put together the pipeline
pipe = Pipeline(stages=[indexerContacts, indexerRules, als])

# COMMAND ----------

# Regularization Rates
regs = [1, 0.1, 0.01, 0.001]
paramGrid = ParamGridBuilder().addGrid(als.regParam, regs).build()

evaluator = RegressionEvaluator(metricName="rmse", labelCol=ratingCol, predictionCol="prediction")
cv = CrossValidator(estimator=pipe, evaluator=evaluator, estimatorParamMaps=paramGrid)
cvModel = cv.fit(train)
i = 0

# COMMAND ----------

# record a bunch of reg values in a ALS model
for reg in regs:
    print(reg)
    # create a bunch of child runs
    with root_run.child_run("reg-" + str(reg)) as run:
        rmse = cvModel.avgMetrics[i]
        print("Root-mean-square error = " + str(rmse))
        
        # log reg, rmse and feature names in run history
        run.log("reg", reg)
        run.log("rmse", rmse)
        run.log_list("columns", train.columns)
        i += 1

# COMMAND ----------

# save model
cvModel.bestModel.write().overwrite().save(model_name)
print("Model saved")

# upload the serialized model into run history record
mdl, ext = model_name.split(".")
model_zip = mdl + ".zip"
shutil.make_archive(mdl, 'zip', model_dbfs)
root_run.upload_file("outputs/" + model_name, model_zip)        

# now delete the serialized model from local folder since it is already uploaded to run history 
# shutil.rmtree(model_dbfs)
os.remove(model_zip)
        
# Declare run completed
root_run.complete()

##NOTE: service deployment always gets the model from the current working dir.
print("copy model from dbfs to local")
model_local = "file:" + os.getcwd() + "/" + model_name
dbutils.fs.cp(model_name, model_local, True)

# COMMAND ----------

pred = cvModel.transform(test)
display(pred)

re = RegressionEvaluator(metricName="rmse", labelCol=ratingCol,
                                predictionCol="prediction")
rmse = re.evaluate(pred)
print("Root-mean-square error = " + str(rmse))

# COMMAND ----------

import json
with open('secrets.json') as json_data:
  writeConfig = json.load(json_data)

  recs = cvModel.bestModel.stages[2].recommendForAllUsers(10)
  recs.withColumn("id",recs[userColIndex].cast("string")).select("id", "recommendations."+ itemColIndex)\
    .write.format("com.microsoft.azure.cosmosdb.spark").mode('overwrite').options(**writeConfig).save()

# COMMAND ----------

#%%writefile score_sparkml.py

score_sparkml = """

import json
def init(local=False):
    global client, collection
    try:
      # Query them in SQL
      import pydocumentdb.document_client as document_client

      MASTER_KEY = '{key}'
      HOST = '{endpoint}'
      DATABASE_ID = "{database}"
      COLLECTION_ID = "{collection}"
      database_link = 'dbs/' + DATABASE_ID
      collection_link = database_link + '/colls/' + COLLECTION_ID
      
      client = document_client.DocumentClient(HOST, {'masterKey': MASTER_KEY})
      collection = client.ReadCollection(collection_link=collection_link)
    except Exception as e:
      collection = e
def run(input_json):      

    try:
      import json

      id = json.loads(json.loads(input_json)[0])['id']
      query = {'query': 'SELECT * FROM c WHERE c.id = "' + str(id) +'"' } #+ str(id)

      options = {}

      result_iterable = client.QueryDocuments(collection['_self'], query, options)
      result = list(result_iterable);
  
    except Exception as e:
        result = str(e)
    return json.dumps(str(result)) #json.dumps({{"result":result}})
"""#.format(model_name=model_name)

import json
with open('secrets.json') as json_data:
  writeConfig = json.load(json_data)
  score_sparkml = score_sparkml.replace("{key}",writeConfig['Masterkey']).replace("{endpoint}",writeConfig['Endpoint']).replace("{database}",writeConfig['Database']).replace("{collection}",writeConfig['Collection'])

  exec(score_sparkml)

  with open("score_sparkml.py", "w") as file:
      file.write(score_sparkml)

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

from azureml.core.webservice import AciWebservice, Webservice
#aci = azure container instance
aci_config = AciWebservice.deploy_configuration(
    cpu_cores = 1, 
    memory_gb = 1, 
    tags = {'name':'Spark ML Databricks sample'}, 
    description = 'This is a great example.')

# COMMAND ----------

from azureml.core.model import Model
mymodel = Model.register(model_path = model_name, # this points to a local file
                       model_name = model_name, # this is the name the model is registered as, am using same name for both path and name.                 
                       description = "ADB trained model by Dan",
                       workspace = ws)

print(mymodel.name, mymodel.description, mymodel.version)

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


# COMMAND ----------

try:
  myservice = AciWebservice(ws, name=service_name)
except:
  myservice = Webservice.deploy_from_model(
    workspace=ws, 
    name=service_name,
    deployment_config = aci_config,
    models = models,
    image_config = myimage_config
      )

  myservice.wait_for_deployment(show_output=True)

print(yaml.dump(myservice.__dict__, default_flow_style=False))

# COMMAND ----------

json2 = '["{\\"id\\":\\"5616\\"}"]'.encode()
res1_service = myservice.run(input_data = json2)
print(res1_service)

# COMMAND ----------

