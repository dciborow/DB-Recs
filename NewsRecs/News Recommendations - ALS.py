# Databricks notebook source
# MAGIC %md 
# MAGIC ### News Recommendation ALS Example Databricks Notebook
# MAGIC ##### by Daniel Ciborowski, dciborow@microsoft.com
# MAGIC 
# MAGIC ##### Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC ##### Licensed under the MIT License.
# MAGIC 
# MAGIC ##### Setup
# MAGIC 1. Create new Cluster, DB 4.1, Spark 2.3.0, Python3
# MAGIC 1. (Optional for Ranking Metrics) From Maven add to cluster the following jar: Azure:mmlspark:0.15
# MAGIC 
# MAGIC In a news recommendation scenario, items have an active lifespan when they should be recommended. After this time has expired old stories are not recommended, and new news stories replace the expired ones. When recommending new stories, only active stories should be recommended. This example shows how to train a model using historical data, and make recommendations for the latest news stories.
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

# Create Sample Data
raw = [
  {'userId': 1, 'itemId': 1, 'rating':  random.randint(0, 10), 'timestamp': 1462277923},
  {'userId': 2, 'itemId': 1, 'rating':  random.randint(0, 10), 'timestamp': 1463455636},
  {'userId': 3, 'itemId': 1, 'rating':  random.randint(0, 10), 'timestamp': 1464277923},
  {'userId': 4, 'itemId': 1, 'rating':  random.randint(0, 10), 'timestamp': 1465277923},
  {'userId': 5, 'itemId': 1, 'rating':  random.randint(0, 10), 'timestamp': 1466277923},
  {'userId': 1, 'itemId': 2, 'rating':  random.randint(0, 10), 'timestamp': 1467277923},
  {'userId': 2, 'itemId': 2, 'rating':  random.randint(0, 10), 'timestamp': 1468277923},
  {'userId': 3, 'itemId': 2, 'rating':  random.randint(0, 10), 'timestamp': 1469277923},
  {'userId': 4, 'itemId': 2, 'rating':  random.randint(0, 10), 'timestamp': 1471277923},
  {'userId': 5, 'itemId': 2, 'rating':  random.randint(0, 10), 'timestamp': 1472277923},
  {'userId': 1, 'itemId': 3, 'rating':  random.randint(0, 10), 'timestamp': 1473277923},
  {'userId': 2, 'itemId': 3, 'rating':  random.randint(0, 10), 'timestamp': 1474277923},
  {'userId': 3, 'itemId': 3, 'rating':  random.randint(0, 10), 'timestamp': 1475277923},
  {'userId': 4, 'itemId': 3, 'rating':  random.randint(0, 10), 'timestamp': 1476277923},
  {'userId': 5, 'itemId': 3, 'rating':  random.randint(0, 10), 'timestamp': 1477277923},
  {'userId': 1, 'itemId': 4, 'rating':  random.randint(0, 10), 'timestamp': 1478277923},
  {'userId': 2, 'itemId': 4, 'rating':  random.randint(0, 10), 'timestamp': 1479277923},
  {'userId': 3, 'itemId': 4, 'rating':  random.randint(0, 10), 'timestamp': 1481277923},
  {'userId': 4, 'itemId': 4, 'rating':  random.randint(0, 10), 'timestamp': 1482277923},
  {'userId': 5, 'itemId': 4, 'rating':  random.randint(0, 10), 'timestamp': 1483277923},  
  {'userId': 1, 'itemId': 5, 'rating':  random.randint(0, 10), 'timestamp': 1484277923},
  {'userId': 2, 'itemId': 5, 'rating':  random.randint(0, 10), 'timestamp': 1485277923},
  {'userId': 3, 'itemId': 5, 'rating':  random.randint(0, 10), 'timestamp': 1486277923},
  {'userId': 4, 'itemId': 5, 'rating':  random.randint(0, 10), 'timestamp': 1487277923},
  {'userId': 5, 'itemId': 5, 'rating':  random.randint(0, 10), 'timestamp': 1492455636},   
]

day1 = pd.DataFrame(raw)
day2=pd.DataFrame(raw)
day2['itemId'] = day2['itemId']+10
day2['timestamp'] = day2['timestamp']+100000000
day3=pd.DataFrame(raw)
day3['itemId'] = day3['itemId']+20
day3['timestamp'] = day3['timestamp']+200000000
day4=pd.DataFrame(raw)
day4['itemId'] = day4['itemId']+30
day4['timestamp'] = day4['timestamp']+300000000

data = day1 \
  .append(day2) \
  .append(day3) \
  .append(day4) \
  .sample(frac=0.75, replace=False)

spark = SparkSession.builder.getOrCreate()
ratings = spark.createDataFrame(data)
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
print("Root-mean-square error = " + str(rmse))

# COMMAND ----------

# Evaluate the model by computing ranking metrics on the rating data
from mmlspark.RankingAdapter import RankingAdapter
from mmlspark.RankingEvaluator import RankingEvaluator

output = RankingAdapter(mode='allUsers', k=5, recommender=algo) \
  .fit(ratings) \
  .transform(ratings)

metrics = ['ndcgAt','map','recallAtK','mrr','fcp']
metrics_dict = {}
for metric in metrics:
    metrics_dict[metric] = RankingEvaluator(k=3, metricName=metric).evaluate(output)

metrics_dict    

# COMMAND ----------

# Recommend Subset Wrapper
def recommendSubset(self, df, timestamp):
  def Func(lines):
    out = []
    for i in range(len(lines[1])):
      out += [(lines[1][i],lines[2][i])]
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

# Recommend most recent items for all users
recs = model.recommendSubset(ratings, 1662277923)

display(recs.orderBy('userId'))

# COMMAND ----------

# MAGIC %md
# MAGIC In order to turn new stories from cold items, to warm items, 1% of the recommendations servered should include a random new (cold) story. This population should also be used to provide a baseline to measure the online model performance.

# COMMAND ----------

# MAGIC %md
# MAGIC # Repeat with a larger dataset.
# MAGIC 
# MAGIC 1 Week of data collection - 923 articles (in Norwegian), 15,514 users, average article length is 518.6 words 

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

# COMMAND ----------

display(ratings.select('userId','itemId','rating','timestamp').orderBy('userId','itemId'))

# COMMAND ----------

display(ratings.select('userId','itemId','rating','timestamp').orderBy('userId','itemId'))

# COMMAND ----------

# Build the recommendation model using ALS on the rating data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
algo = ALS(userCol="userId", itemCol="itemId", implicitPrefs=True, coldStartStrategy="drop")
model = algo.fit(ratings)

# Evaluate the model by computing the RMSE on the rating data
predictions = model.transform(ratings)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Evaluate the model by computing ranking metrics on the rating data
from mmlspark.RankingAdapter import RankingAdapter
from mmlspark.RankingEvaluator import RankingEvaluator

output = RankingAdapter(mode='allUsers', k=5, recommender=algo) \
  .fit(ratings) \
  .transform(ratings)

metrics = ['ndcgAt','map','recallAtK','mrr','fcp']
metrics_dict = {}
for metric in metrics:
    metrics_dict[metric] = RankingEvaluator(k=3, metricName=metric).evaluate(output)

print(metrics_dict)

# Recommend most recent items for all users
recs = model.recommendSubset(ratings, 1483747200) \
  .cache()

recs.take(5)

# COMMAND ----------


