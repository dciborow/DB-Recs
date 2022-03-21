# Databricks notebook source
# MAGIC %md
# MAGIC ### Add Maven package as new Libaray in Databricks Workspace and attach to cluster...
# MAGIC 
# MAGIC #### Package
# MAGIC com.microsoft.ml.spark:mmlspark_2.11:0.14.dev0+10.g56206725 
# MAGIC #### Repository (under advanced)
# MAGIC https://mmlspark.azureedge.net/maven

# COMMAND ----------

import pyspark
from mmlspark.RecommendationEvaluator import RecommendationEvaluator
from pyspark.ml.tuning import *
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS
from mmlspark.TrainValidRecommendSplit import TrainValidRecommendSplit

# COMMAND ----------

def getRatings():
        cSchema = StructType([StructField("user", IntegerType()),
                              StructField("item", IntegerType()),
                              StructField("rating", IntegerType()),
                              StructField("notTime", IntegerType())])

        return pyspark.sql.SparkSession.builder.getOrCreate().createDataFrame([
            (0, 1, 4, 4),
            (0, 3, 1, 1),
            (0, 4, 5, 5),
            (0, 5, 3, 3),
            (0, 7, 3, 3),
            (0, 9, 3, 3),
            (0, 10, 3, 3),
            (1, 1, 4, 4),
            (1, 2, 5, 5),
            (1, 3, 1, 1),
            (1, 6, 4, 4),
            (1, 7, 5, 5),
            (1, 8, 1, 1),
            (1, 10, 3, 3),
            (2, 1, 4, 4),
            (2, 2, 1, 1),
            (2, 3, 1, 1),
            (2, 4, 5, 5),
            (2, 5, 3, 3),
            (2, 6, 4, 4),
            (2, 8, 1, 1),
            (2, 9, 5, 5),
            (2, 10, 3, 3),
            (3, 2, 5, 5),
            (3, 3, 1, 1),
            (3, 4, 5, 5),
            (3, 5, 3, 3),
            (3, 6, 4, 4),
            (3, 7, 5, 5),
            (3, 8, 1, 1),
            (3, 9, 5, 5),
            (3, 10, 3, 3)], cSchema)

# COMMAND ----------

ratings = getRatings()

als = ALS() \
    .setUserCol("user") \
    .setRatingCol('rating') \
    .setItemCol("item") \

paramGrid = ParamGridBuilder() \
    .addGrid(als.maxIter, [1, 2, 3, 5, 8]) \
    .build()

evaluator = RecommendationEvaluator().setSaveAll(True)

tvRecommendationSplit = TrainValidRecommendSplit() \
    .setEstimator(als) \
    .setEvaluator(evaluator) \
    .setEstimatorParamMaps(paramGrid) \
    .setTrainRatio(0.8) \
    .setUserCol(als.getUserCol()) \
    .setRatingCol(als.getRatingCol()) \
    .setItemCol(als.getItemCol())

tvmodel = tvRecommendationSplit.fit(ratings)

usersRecs = tvmodel.recommendForAllUsers(3)

print("User Recs: ")
print(usersRecs.take(1))
print("Validation Metrics: ")
print(tvmodel.validationMetrics)

metrics = evaluator._call_java("getMetricsList").toString()
print("Metrics List: ")
print(metrics)

# COMMAND ----------


