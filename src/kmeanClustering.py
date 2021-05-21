import sys
from random import random
from operator import add
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sbs
from matplotlib.ticker import MaxNLocator

from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.sql.types import ArrayType, StringType, DoubleType, IntegerType
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.ml.feature import VectorIndexer, IndexToString, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')
    print(cm)
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()

  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")
  
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

# -------- convert data to dense vectors 
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

def modelsummary(model):
    print ("Note: the last row : 3 coeffients + bias")
    print ("##","-------------------------------------------------")
    print ("##"," Estimate Std.Error | t Values | P-value")

    coef = np.append(list(model.coefficients), model.intercept)
    Summary=model.summary

    for i in range(len(Summary.pValues)):
        print ("##",'{:10.6f}'.format(coef[i]),\
            '{:10.6f}'.format(Summary.coefficientStandardErrors[i]),\
            '{:8.3f}'.format(Summary.tValues[i]),\
            '{:10.6f}'.format(Summary.pValues[i]))

    print ("##",'---')
    print ("##","Mean squared error: % .6f" \
            % Summary.meanSquaredError, ", RMSE: % .6f" \
            % Summary.rootMeanSquaredError )
    print ("##","Multiple R-squared: %f" % Summary.r2, ", \
            Total iterations: %i"% Summary.totalIterations)


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


# lambda in rdd.map for "map": good for large features
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1])]).toDF(['features'])

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

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("K-means clustering")\
        .getOrCreate()

    # infer schema in sql-favor: https://www.kaggle.com/sgus1318/winedata
    df = spark.read.format('com.databricks.spark.csv').\
      options(header='true', inferschema='true').\
      load("../data/iris.csv", header=True);

    # head(5) and schema
    df.show(5)
    df.printSchema()
    df.describe().show()
    
    transformed = transData(df)
    transformed.show(5)
    
    featureIndexer =VectorIndexer(inputCol="features", \
      outputCol="indexedFeatures", \
      maxCategories=4).fit(transformed)
    featureIndexer.transform(transformed).show(5, True)
    
    data = featureIndexer.transform(transformed)

    # it’s hard to choose the optimal number of the clusters by using the elbow
    # method.
    cost = np.zeros(20)
    for k in range(2,20):
      kmeans = KMeans().setK(k).setSeed(1)

      model = kmeans.fit(data)
      predictions = model.transform(data) 

      evaluator = ClusteringEvaluator()

      silhouette = evaluator.evaluate(predictions)
      cost[k] = silhouette

    print(cost)

    fig, ax = plt.subplots(1,1, figsize =(8,6))
    ax.plot(range(2,20),cost[2:20])
    ax.set_xlabel('k')
    ax.set_ylabel('cost')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    spark.stop()


