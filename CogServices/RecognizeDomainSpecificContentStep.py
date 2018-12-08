import os

from mmlspark import RecognizeDomainSpecificContent
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import SparkSession

dbutils.widgets.get("account_name") 
account_name = getArgument("account_name")

dbutils.widgets.get("input_path") 
input_path = getArgument("input_path")

dbutils.widgets.get("output_path") 
output_path = getArgument("output_path")

wasb_path    = "wasbs://azureml@"+account_name+".blob.core.windows.net/"

urls = SparkSession.builder.getOrCreate().read.parquet(wasb_path + input_path)

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
output.write.parquet(wasb_path+output_path, mode='overwrite')
output
