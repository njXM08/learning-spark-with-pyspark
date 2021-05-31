# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName("pyspark-feature-engineering").getOrCreate()

# create data-frame from scratch: rows, col-names
dataset = spark.createDataFrame([(0, 18, 1.0, Vectors.dense([0.0, 10.0, 0.5]), 1.0),
                                 (1, 10, 1.0, Vectors.dense([5.0, 1.0, 7.5]), 1.0)],
                                ["id", "hour", "mobile", "userFeatures", "clicked"])

# assemble a few columns (features) into a single feature
assembler = VectorAssembler(inputCols=["hour", "mobile", "userFeatures"], outputCol="features")
output = assembler.transform(dataset)

print("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
output.select("features", "clicked").show(truncate=False)

df = spark.createDataFrame([(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")], 
                           ["id", "category"])
df.show()

from pyspark.ml.feature import OneHotEncoder, StringIndexer
stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)
print('After StringIndexer:')
indexed.show()

# default setting: dropLast=True -> eliminate the leastly occurring category
encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec", dropLast=False)
encoder = encoder.fit(indexed)
encoded = encoder.transform(indexed)

# 3 categories: one-hot vector of length-3 
# -> categoryVec: (len, [index], [value at index])
print('After OneHotEncoder:')
encoded.show()

from pyspark.ml import Pipeline

# list of StringIndexer to convert categorical columns to integer indices
categoricalCols = ['category']
indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
              for c in categoricalCols ]

# OneHotEncoder default setting: dropLast=True 
# one OHE for each StringIndexer
encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), \
                           outputCol="{0}_encoded".format(indexer.getOutputCol()), dropLast=False) for indexer in indexers ]

assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders], 
                            outputCol="features")

# complete pipeline as a list of (cascaded) transforming objects
pipeline = Pipeline(stages=indexers + encoders + [assembler])

model=pipeline.fit(df)
data = model.transform(df)
data.show()

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
  indexCol = 'id'
  categoricalCols = ['category']
  continuousCols = []
  labelCol = []
  mat = get_dummy(df,indexCol,categoricalCols,continuousCols,labelCol)
  mat.show()

  >>>
  +---+-------------+
  | id| features|
  +---+-------------+
  | 0|[1.0,0.0,0.0]|
  | 1|[0.0,0.0,1.0]|
  | 2|[0.0,1.0,0.0]|
  | 3|[1.0,0.0,0.0]|
  | 4|[1.0,0.0,0.0]|
  | 5|[0.0,1.0,0.0]|
  +---+-------------+
  '''

  from pyspark.ml import Pipeline
  from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
  from pyspark.sql.functions import col

  indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
    for c in categoricalCols ]

  # default setting: dropLast=True
  encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
    outputCol="{0}_encoded".format(indexer.getOutputCol()), dropLast=dropLast)
    for indexer in indexers ]
  assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder
   in encoders] + continuousCols, outputCol="features")
  
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

df = spark.createDataFrame([
(0, "a"),
(1, "b"),
(2, "c"),
(3, "a"),
(4, "a"),
(5, "c")
], ["id", "category"])

indexCol = 'id'
categoricalCols = ['category']
continuousCols = []
labelCol = []
mat = get_dummy(df,indexCol,categoricalCols,continuousCols,labelCol)
mat.show()

"""Scaler"""

from pyspark.ml.feature import Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler
scaler_type = 'Standard'
if scaler_type=='Normal':
  scaler = Normalizer(inputCol="features", outputCol="scaledFeatures", p=1.0)
elif scaler_type=='Standard':
  scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
elif scaler_type=='MinMaxScaler':
  scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
elif scaler_type=='MaxAbsScaler':
  scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")

from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
df = spark.createDataFrame([
(0, Vectors.dense([1.0, 0.5, -1.0]),),
(1, Vectors.dense([2.0, 1.0, 1.0]),),
(2, Vectors.dense([4.0, 10.0, 2.0]),)], ["id", "features"])

df.show()
pipeline = Pipeline(stages=[scaler])
model =pipeline.fit(df)
data = model.transform(df)
data.show(truncate=False)

#from pyspark.sql.functions import avg, round
#df2 = df.withColumn("col4", round(df["features"], 4)).withColumnRenamed("col4","new_features")
#df2.show()

"""Normalizer"""

from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors
dataFrame = spark.createDataFrame([
(0, Vectors.dense([1.0, 0.5, -1.0]),),
(1, Vectors.dense([2.0, 1.0, 1.0]),),
(2, Vectors.dense([4.0, 10.0, 2.0]),)
], ["id", "features"])

# Normalize each Vector using $L^1$ norm.
normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
l1NormData = normalizer.transform(dataFrame)   # sum = 1?
print("Normalized using L^1 norm")
l1NormData.show()

# Normalize each Vector using $L^\infty$ norm: x_i = x_i / max{x_i}
lInfNormData = normalizer.transform(dataFrame, {normalizer.p: float("inf")})
print("Normalized using L^inf norm")
lInfNormData.show()

"""StandardScaler"""

from pyspark.ml.feature import Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler
from pyspark.ml.linalg import Vectors
dataFrame1 = spark.createDataFrame([
(0, Vectors.dense([1.0, 0.5, -1.0])),
(1, Vectors.dense([2.0, 1.0, 1.0])),
(2, Vectors.dense([4.0, 10.0, 2.0]))
], ["id", "features"])

dataFrame = spark.createDataFrame([
(0, Vectors.dense([1.0])),
(1, Vectors.dense([2.0])),
(2, Vectors.dense([4.0])),
], ["id", "features"])

# x_i <- x_i/sigma (Not true!!)  
# due to unbiased estimate of sigma sqrt(1/(N-1) sum((x_i - mu)^2))
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
withStd=True, withMean=True)
scaleredData = scaler.fit((dataFrame)).transform(dataFrame)
scaleredData.show(truncate=False)

"""MinMaxScaler: to a specific range [min, max] ~ (x-Xmin)/(Xmax-Xmin)*(max-min) + min"""

from pyspark.ml.feature import Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler
from pyspark.ml.linalg import Vectors
dataFrame = spark.createDataFrame([
(0, Vectors.dense([1.0, 0.5, -1.0]),),
(1, Vectors.dense([2.0, 1.0, 1.0]),),
(2, Vectors.dense([4.0, 10.0, 2.0]),)
], ["id", "features"])

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scaledData = scaler.fit((dataFrame)).transform(dataFrame)
scaledData.show(truncate=False)

# (3,[],[]) means that a length-3 vector with all zero values!

"""MaxAbsScaler: feature (column) scaling -> x_i/max({|x_i|})"""

from pyspark.ml.feature import Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler
from pyspark.ml.linalg import Vectors
dataFrame = spark.createDataFrame([
(0, Vectors.dense([1.0, 0.5, -1.0]),),
(1, Vectors.dense([2.0, 1.0, 1.0]),),
(2, Vectors.dense([4.0, 10.0, 2.0]),)
], ["id", "features"])
scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")
scaledData = scaler.fit((dataFrame)).transform(dataFrame)
scaledData.show(truncate=False)

"""PCA"""

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
(Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
(Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data, ["features"])

# number of kept components
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")

model = pca.fit(df)
result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)

"""DCT"""

from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors
df = spark.createDataFrame([
(Vectors.dense([0.0, 1.0, -2.0, 3.0]),),
(Vectors.dense([-1.0, 2.0, 4.0, -7.0]),),
(Vectors.dense([14.0, -2.0, -5.0, 1.0]),)], ["features"])

# DCT transform
dct = DCT(inverse=False, inputCol="features", outputCol="featuresDCT")
dctDf = dct.transform(df)
dctDf.select("featuresDCT").show(truncate=False)