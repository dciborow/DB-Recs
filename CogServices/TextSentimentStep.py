from mmlspark import *
from pyspark.ml.feature import SQLTransformer
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


dbutils.widgets.get("account_name")
account_name = getArgument("account_name")

dbutils.widgets.get("input_path")
input_path = getArgument("input_path")

dbutils.widgets.get("output_path")
output_path = getArgument("output_path")

wasb_path = f"wasbs://azureml@{account_name}.blob.core.windows.net/"

text = SparkSession.builder.getOrCreate().read.parquet(wasb_path + input_path)

dbutils.widgets.get("TEXT_API_KEY")
TEXT_API_KEY = getArgument("TEXT_API_KEY")

dbutils.widgets.get("region")
region = getArgument("region")

dbutils.widgets.get("text_col")
text_col = getArgument("text_col")

dbutils.widgets.get("output_col_sentiment")
output_col_sentiment = getArgument("output_col_sentiment")

sentimentTransformer = (
    TextSentiment()
    .setTextCol(text_col)
    .setUrl(
        f"https://{region}"
        + ".api.cognitive.microsoft.com/text/analytics/v2.0/sentiment"
    )
    .setSubscriptionKey(TEXT_API_KEY)
    .setOutputCol(output_col_sentiment)
)


#Extract the sentiment score from the API response body
getSentiment = SQLTransformer(
    statement=f"SELECT *, {output_col_sentiment}"
    + "[0].score as sentimentScore FROM __THIS__"
)


output = PipelineModel(stages=[sentimentTransformer, getSentiment]).transform(text)
output.write.parquet(wasb_path + output_path, mode='overwrite')
output
