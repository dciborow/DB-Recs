from mmlspark import BingImageSearch
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

#Get the inputs for the Databricks ML Step
dbutils.widgets.get("account_name") 
account_name = getArgument("account_name")

dbutils.widgets.get("output_path") 
output_path = getArgument("output_path")

wasb_path    = "wasbs://azureml@"+account_name+".blob.core.windows.net/"

dbutils.widgets.get("BING_IMAGE_SEARCH_KEY") 
BING_IMAGE_SEARCH_KEY = getArgument("BING_IMAGE_SEARCH_KEY")

dbutils.widgets.get("input_query") 
input_query = getArgument("input_query")

imgsPerBatch = 10 #the number of images Bing will return for each query
offsets = [(i*imgsPerBatch,) for i in range(100)] # A list of offsets, used to page into the search results

bingParameters = SparkSession.builder.getOrCreate().createDataFrame(offsets, ["offset"])

bingSearch = BingImageSearch()\
  .setSubscriptionKey(BING_IMAGE_SEARCH_KEY)\
  .setOffsetCol("offset")\
  .setQuery(input_query)\
  .setCount(imgsPerBatch)\
  .setOutputCol("images")

#Transformer to that extracts and flattens the richly structured output of Bing Image Search into a simple URL column
getUrls = BingImageSearch.getUrlTransformer("images", "url")

output = PipelineModel(stages=[bingSearch, getUrls]).transform(bingParameters).cache()
output.write.parquet(wasb_path + output_path, mode='overwrite')
output
