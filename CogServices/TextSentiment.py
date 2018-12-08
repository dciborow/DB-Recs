from mmlspark import *
from pyspark.ml.feature import SQLTransformer
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

account_name = def_blob_store.account_name
wasb_path    = "wasbs://azureml@"+account_name+".blob.core.windows.net/"
input_path   = wasb_path + "raw_data/cog_services/celebs/text/"
output_path  = wasb_path + "raw_data/cog_services/celebs/sentiment/"

text = SparkSession.builder.getOrCreate().read.parquet(input_path)

dbutils.widgets.get("TEXT_API_KEY") 
TEXT_API_KEY = getArgument("TEXT_API_KEY")

dbutils.widgets.get("text_col") 
text_col = getArgument("text_col")

dbutils.widgets.get("output_col_sentiment") 
output_col_sentiment = getArgument("output_col_sentiment")

sentimentTransformer = TextSentiment()\
    .setTextCol(text_col)\
    .setUrl("https://"+region+".api.cognitive.microsoft.com/text/analytics/v2.0/sentiment")\
    .setSubscriptionKey(TEXT_API_KEY)\
    .setOutputCol(output_col_sentiment)

#Extract the sentiment score from the API response body
getSentiment = SQLTransformer(statement="SELECT *, "+output_col_sentiment+"[0].score as sentimentScore FROM __THIS__")

sentiment = PipelineModel(stages=[sentimentTransformer, getSentiment]).transform(text)
sentiment.write.parquet(output_path, mode='overwrite')
sentiment
