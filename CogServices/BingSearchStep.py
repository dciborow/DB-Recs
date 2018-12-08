from mmlspark import BingImageSearch
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

#Get the inputs for the Databricks ML Step
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

urls = PipelineModel(stages=[bingSearch, getUrls]).transform(bingParameters).cache()
urls.write.parquet("wasbs://azureml@"+def_blob_store.account_name+".blob.core.windows.net/raw_data/cog_services/celebs/urls/", mode='overwrite')
urls
