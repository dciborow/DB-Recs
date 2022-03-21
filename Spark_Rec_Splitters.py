# Databricks notebook source
# MAGIC %md 
# MAGIC ### Chronological Splitter Example Databricks Notebook
# MAGIC ##### by Le Zhang: zhle@microsoft.com, Daniel Ciborowski: dciborow@microsoft.com
# MAGIC 
# MAGIC ##### Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC ##### Licensed under the MIT License.

# COMMAND ----------

from pyspark.sql.functions import col, row_number
def ChronoSplit(self, ratio = .75, filter_by = 'user',
                col_user = 'UserId', col_item = 'ItemId', col_timestamp = 'Timestamp'):
  """Chronological Splitter - split data (items are ordered by timestamps for each customer) by
    timestamps.
  Args:
      ratio (float): Ratio for splitting data. 
        It splits data into two halfs and the ratio argument indicates the ratio of training data set;        
      filter_by (str): either "user" or "item", depending on which of the two is to filter with min_rating.
      col_user (str): column name of user IDs.
      col_item (str): column name of item IDs.
      col_timestamp (str): column name of timestamps.
  """
  split_by_column = col_user if filter_by == "user" else col_item
  
  rating_grouped = self.groupBy(split_by_column).agg({col_timestamp: 'count'})
  
  window_spec = pyspark.sql.Window.partitionBy(split_by_column).orderBy(col(col_timestamp).desc())  
  
  rating_rank = self.join(rating_grouped, on=split_by_column)\
                    .withColumn('rank', row_number().over(window_spec) / col('count(' + col_timestamp + ')'))\
                    .cache()

  left = rating_rank.filter(col('rank') <= ratio).drop('rank', 'count(' + col_timestamp + ')')
  right = rating_rank.filter(col('rank') > ratio).drop('rank', 'count(' + col_timestamp + ')')

  return left, right  

pyspark.sql.dataframe.DataFrame.ChronoSplit = ChronoSplit

# COMMAND ----------

import os
# Download Movie Lens
basedataurl = "http://aka.ms"
datafile = "MovieRatings.csv"

datafile_dbfs = os.path.join("/dbfs", datafile)

if os.path.isfile(datafile_dbfs):
  print(f"found {datafile} at {datafile_dbfs}")
else:
  print(f"downloading {datafile} to {datafile_dbfs}")
  urllib.request.urlretrieve(os.path.join(basedataurl, datafile), datafile_dbfs)

data_all = sqlContext.read.format('csv')\
                     .options(header='true', delimiter=',', inferSchema='true', ignoreLeadingWhiteSpace='true', ignoreTrailingWhiteSpace='true')\
                     .load(datafile)

# COMMAND ----------

train, test = data_all.ChronoSplit()
print(f"Train Count: {str(train.count())}")
print(f"Test Count:  {str(test.count())}")

# COMMAND ----------

