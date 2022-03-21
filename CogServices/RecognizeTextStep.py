from mmlspark import RecognizeText, UDFTransformer
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import SparkSession

account_name = def_blob_store.account_name
wasb_path = f"wasbs://azureml@{account_name}.blob.core.windows.net/"
input_path = f'{wasb_path}raw_data/cog_services/celebs/celeb/'
output_path = f'{wasb_path}raw_data/cog_services/celebs/text/'

image_urls = SparkSession.builder.getOrCreate().read.parquet(input_path)

dbutils.widgets.get("VISION_API_KEY")
VISION_API_KEY = getArgument("VISION_API_KEY")

dbutils.widgets.get("region")
region = getArgument("region")

dbutils.widgets.get("mode")
mode = getArgument("mode")

dbutils.widgets.get("image_url_col")
image_url_col = getArgument("image_url_col")

dbutils.widgets.get("output_col_rec_text")
output_col_rec_text = getArgument("output_col_rec_text")

dbutils.widgets.get("concurrency")
concurrency = int(getArgument("concurrency"))

mid_col = "ocr"

recognizeText = (
    RecognizeText()
    .setSubscriptionKey(VISION_API_KEY)
    .setUrl(
        f"https://{region}"
        + ".api.cognitive.microsoft.com/vision/v2.0/recognizeText"
    )
    .setImageUrlCol(image_url_col)
    .setMode(mode)
    .setOutputCol(mid_col)
    .setConcurrency(concurrency)
)

def getTextFunction(ocrRow):
    if ocrRow is None: return None
    return "\n".join([line.text for line in ocrRow.recognitionResult.lines])

# this transformer wil extract a simpler string from the structured output of recognize text

getText = UDFTransformer().setUDF(udf(getTextFunction)).setInputCol(mid_col).setOutputCol(output_col_rec_text)

output = PipelineModel(stages=[recognizeText, getText]).transform(image_urls)
output.write.parquet(output_path, mode='overwrite')
output
