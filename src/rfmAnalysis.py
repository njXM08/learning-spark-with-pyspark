import sys
from random import random
from operator import add
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import confusion_matrix

from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.sql.types import ArrayType, StringType, DoubleType, IntegerType

from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorIndexer, IndexToString, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.functions import col
from pyspark.sql import functions as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

from pyspark.sql.functions import to_utc_timestamp, unix_timestamp, from_unixtime, to_timestamp, lit, datediff, col

# -------- convert data to dense vectors 
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

def RScore(x):
  if x <= 16:
    return 1
  elif x<= 50:
    return 2
  elif x<= 143:
    return 3
  else:
    return 4

def FScore(x):
  if x <= 1:
    return 4
  elif x <= 3:
    return 3
  elif x <= 5:
    return 2
  else:
    return 1

def MScore(x):
  if x <= 293:
    return 4
  elif x <= 648:
    return 3
  elif x <= 1611:
    return 2
  else:
    return 1

def describe_pd(df_in, columns, deciles=False):
  '''
  Function to union the basic stats results and deciles
  :param df_in: the input dataframe
  :param columns: the cloumn name list of the numerical variable
  :param deciles: the deciles output
  :return : the numerical describe info. of the input dataframe
  :author: Ming Chen and Wenqiang Feng
  :email: von198@gmail.com
  '''
  if deciles:
    percentiles = np.array(range(0, 110, 10))
  else:
    percentiles = [25, 50, 75]
    
  percs = np.transpose([np.percentile(df_in.select(x).collect(),
    percentiles) for x in columns])
  percs = pd.DataFrame(percs, columns=columns)
  percs['summary'] = [str(p) + '%' for p in percentiles]
  spark_describe = df_in.describe().toPandas()
  new_df = pd.concat([spark_describe, percs],ignore_index=True)
  new_df = new_df.round(2)
  return new_df[['summary'] + columns]

def get_dummy(df, indexCol, categoricalCols, continuousCols, labelCol, dropLast=False):
  '''
  Get dummy variables and concat with continuous variables for ml modeling.
  :param df: the dataframe
  :param categoricalCols: the name list of the categorical data
  :param continuousCols: the name list of the numerical data
  :param labelCol: the name of label column
  :param dropLast: the flag of drop last column
  :return: feature matrix
  :author: Wenqiang Feng
  :email: von198@gmail.com
  >>> df = spark.createDataFrame([
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c")
  ], ["id", "category"])

  >>>
  indexCol = 'id'
  categoricalCols = ['category']
  continuousCols = []
  labelCol = []
  >>> mat = get_dummy(df,indexCol,categoricalCols,continuousCols,
  labelCol)
  >>> mat.show()
  >>>
  +---+-------------+
  | id|features|
  +---+-------------+
  | 0|[1.0,0.0,0.0]|
  | 1|[0.0,0.0,1.0]|
  | 2|[0.0,1.0,0.0]|
  | 3|[1.0,0.0,0.0]|
  | 4|[1.0,0.0,0.0]|
  | 5|[0.0,1.0,0.0]|
  +---+-------------+
  '''
  indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) \
     for c in categoricalCols ]
  
  # default setting: dropLast=True
  encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
                 outputCol="{0}_encoded".format(indexer.getOutputCol()),dropLast=dropLast) \
                 for indexer in indexers ]
  assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders] + continuousCols, \
    outputCol="features")

  pipeline = Pipeline(stages=indexers + encoders + [assembler])
  model=pipeline.fit(df)
  data = model.transform(df)

  if indexCol and labelCol:
    # for supervised learning
    data = data.withColumn('label',col(labelCol))
    return data.select(indexCol,'features','label')
  elif not indexCol and labelCol:
    # for supervised learning
    data = data.withColumn('label',col(labelCol))
    return data.select('features','label')
  elif indexCol and not labelCol:
    # for unsupervised learning
    return data.select(indexCol,'features')
  elif not indexCol and not labelCol:
    # for unsupervised learning
    return data.select('features')


from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col, percent_rank, lit
from pyspark.sql.window import Window
from pyspark.sql import DataFrame, Row
from pyspark.sql.types import StructType
from functools import reduce # For Python 3.x
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import numpy as np

# lambda in rdd.map for "map": good for large features
def transData(data):
  return data.rdd.map(lambda r: [r[0],Vectors.dense(r[1:])]).toDF(['CustomerID','rfm'])

# Convert to float format
def string_to_float(x):
  return float(x)

def condition(r):
  if (0<= r <= 4):
    label = "low"
  elif(4< r <= 6):
    label = "medium"
  else:
    label = "high"
  
  return label

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.functions import count, round
from pyspark.sql.functions import mean, min, max, sum, datediff, to_date

# how to count non-null values in columns
def my_count(df_in):
  df_in.agg( *[ count(c).alias(c) for c in df_in.columns ] ).show()

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Recency, frequency, monetary value analysis")\
        .getOrCreate()

    spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")

    # infer schema in sql-favor: https://www.kaggle.com/sgus1318/winedata
    df_raw = spark.read.format('com.databricks.spark.csv').\
      options(header='true', inferschema='true').\
      load("../data/onlineRetail2.csv", header=True);

    # head(5) and schema
    df_raw.show(5)
    df_raw.printSchema()
    
    my_count(df_raw)

    df = df_raw.dropna(how='any')
    my_count(df)

    # ------------ feature engineering ---------------------
    
    # --------- TODO: issue with v. 3.0?? re-format datetime to "UTC" format --------------
    timeFmt = "MM/dd/yy HH:mm"
    df = df.withColumn('NewInvoiceDate', from_unixtime(unix_timestamp(df.InvoiceDate, timeFmt)))

    # unix_timestamp('date_str', 'MM/dd/yyy')

      # o_utc_timestamp(unix_timestamp(col('InvoiceDate'), \
      #timeFmt).cast('timestamp'), 'UTC'))

    # ---------- create new features -------------------------- 
    # 1). total price
    df = df.withColumn('TotalPrice', round( df.Quantity * df.UnitPrice, 2 ) )


    date_max = df.select(max('NewInvoiceDate')).toPandas()
    current = to_utc_timestamp( unix_timestamp(lit(str(date_max.iloc[0][0])), \
          'yy-MM-dd HH:mm').cast('timestamp'), 'UTC' )
    
    # 2). Calculatre Duration
    df = df.withColumn('Duration', datediff(lit(current), 'NewInvoiceDate'))
    print("duration calculated: ")
    df.show(5)

    # 3) R-F-M features
    recency = df.groupBy('CustomerID').agg(min('Duration').alias('Recency'))
    frequency = df.groupBy('CustomerID', 'InvoiceNo').count()\
          .groupBy('CustomerID')\
          .agg(count("*").alias("Frequency"))
    monetary = df.groupBy('CustomerID').agg(round(sum('TotalPrice'), 2).alias('Monetary'))

    rfm = recency.join(frequency,'CustomerID', how = 'inner')\
                 .join(monetary,'CustomerID', how = 'inner')

    print("RFM calculated: ")
    rfm.show(5)
    my_count(rfm)

    # ---------------- customer insights: determine the cutting points ------------------------
    cols = ['Recency','Frequency','Monetary']
    describe_pd(rfm, cols, 1)

    R_udf = udf(lambda x: RScore(x), StringType())
    F_udf = udf(lambda x: FScore(x), StringType())
    M_udf = udf(lambda x: MScore(x), StringType())
        
    #  segmentation -> binning
    rfm_seg = rfm.withColumn("r_seg", R_udf("Recency"))
    rfm_seg = rfm_seg.withColumn("f_seg", F_udf("Frequency"))
    rfm_seg = rfm_seg.withColumn("m_seg", M_udf("Monetary"))
    rfm_seg.show(5)

    rfm_seg = rfm_seg.withColumn('RFMScore', F.concat(F.col('r_seg'),F.col('f_seg'), F.col('m_seg')))
    rfm_seg.sort(F.col('RFMScore')).show(5)

    rfm_seg.groupBy('RFMScore').agg({'Recency':'mean', 'Frequency': 'mean', 'Monetary': 'mean'} )\
      .sort(F.col('RFMScore')).show(5)

    transformed= transData(rfm)
    print("RFM aggregated: ")
    transformed.show(5)

    scaler = MinMaxScaler(inputCol="rfm", outputCol="features")
    scalerModel = scaler.fit(transformed)
    scaledData = scalerModel.transform(transformed)
    scaledData.show(5,False)

    print("k-mean clustering")

    '''  time consuming
    cost = np.zeros(20)
    for k in range(2,20):
      print(k)
      kmeans = KMeans().setK(k).setSeed(1) \
      .setFeaturesCol("features")\
      .setPredictionCol("prediction")

      model = kmeans.fit(scaledData)
      predictions = model.transform(scaledData)

      evaluator = ClusteringEvaluator()

      silhouette = evaluator.evaluate(predictions)
      cost[k] = silhouette
    '''

    k = 3
    kmeans = KMeans().setK(k).setSeed(1)
    model = kmeans.fit(scaledData)
    
    # Make predictions
    predictions = model.transform(scaledData)
    predictions.show(5,False)

    spark.stop()


