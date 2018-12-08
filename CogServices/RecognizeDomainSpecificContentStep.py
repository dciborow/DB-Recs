import os

from mmlspark import RecognizeDomainSpecificContent
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import SparkSession

urls = SparkSession.builder.getOrCreate().read.parquet("wasbs://azureml@"+def_blob_store.account_name+".blob.core.windows.net/raw_data/cog_services/celebs/urls/")

dbutils.widgets.get("VISION_API_KEY") 
VISION_API_KEY = getArgument("VISION_API_KEY")

dbutils.widgets.get("model") 
model = getArgument("model")

dbutils.widgets.get("region") 
region = getArgument("region")

celebs = RecognizeDomainSpecificContent()\
          .setSubscriptionKey(VISION_API_KEY)\
          .setModel(model)\
          .setUrl("https://"+region+".api.cognitive.microsoft.com/vision/v2.0/")\
          .setImageUrlCol("url")\
          .setOutputCol("celebs")

#Extract the first celebrity we see from the structured response
firstCeleb = SQLTransformer(statement="SELECT *, celebs.result.celebrities[0].name as firstCeleb FROM __THIS__")

output = PipelineModel(stages=[celebs, firstCeleb]).transform(urls)
output.write.parquet("wasbs://azureml@"+def_blob_store.account_name+".blob.core.windows.net/raw_data/cog_services/celebs/celeb/", mode='overwrite')
output
