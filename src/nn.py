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

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorIndexer, IndexToString, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col

import matplotlib.pyplot as plt
import numpy as np
import itertools


# -------- convert data to dense vectors 
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

# lambda in rdd.map for "map": good for large features
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]), r[-1]]).toDF(['features', 'label'])

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
        .appName("Multi-class classification")\
        .getOrCreate()

    # infer schema in sql-favor: https://www.kaggle.com/sgus1318/winedata
    df = spark.read.format('com.databricks.spark.csv').\
      options(header='true', inferschema='true').\
      load("../data/WineData.csv", header=True);

    # head(5) and schema
    df.show(5)
    df.printSchema()
    
    string_to_float_udf = udf(string_to_float, DoubleType())
    quality_udf = udf(lambda x: condition(x), StringType())
    df = df.withColumn("quality", quality_udf("quality"))

    df.show(5,True)

    transformed = transData(df)
    transformed.show(5)

    labelIndexer = StringIndexer(inputCol='label', outputCol='indexedLabel').fit(transformed)
    labelIndexer.transform(transformed).show(5, True)

    featureIndexer =VectorIndexer(inputCol="features", \
      outputCol="indexedFeatures", \
      maxCategories=4).fit(transformed)
    featureIndexer.transform(transformed).show(5, True)

    (trainingData, testData) = transformed.randomSplit([0.7, 0.3])
    trainingData.show(5,False)
    testData.show(5,False)

    # specify layers for the neural network:
    # input layer of size 11 (features), two intermediate of size 5 and 4
    # and output of size 7 (classes)
    layers = [11, 5, 4, 4, 3 , 3]

    # create the trainer and set its parameters
    FNN = MultilayerPerceptronClassifier(labelCol="indexedLabel", \
    featuresCol="indexedFeatures", maxIter=10, layers=layers, \
    blockSize=128, seed=1234)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
    labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    from pyspark.ml import Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, FNN, labelConverter])
    # train the model
    # Train model. This also runs the indexers.
    model = pipeline.fit(trainingData)

    predictions = model.transform(testData)
    predictions.select("features","label","predictedLabel").show(5)

    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", \
      predictionCol="prediction", metricName="accuracy")
    
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    spark.stop()


