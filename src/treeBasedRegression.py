import sys
from random import random
from operator import add
import numpy as np
import pandas as pd
import sklearn.metrics

from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.sql.types import ArrayType, StringType, DoubleType, IntegerType

from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.feature import VectorIndexer
from pyspark.sql.functions import col

# -------- convert data to dense vectors 
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

# lambda in rdd.map for "map": good for large features
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]), r[-1]]).toDF(['features', 'label'])

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("DecisionTreeRegression")\
        .getOrCreate()

    # infer schema in sql-favor
    df = spark.read.format('com.databricks.spark.csv').\
      options(header='true', inferschema='true').\
      load("../data/Advertising.csv", header=True);

    # head(5) and schema
    df.show(5,True)
    df.printSchema()
    
    # ---- transform data: toDenseVectors
    transformed= transData(df)
    transformed.show(5)

    # ------ Automatically identify categorical features, and index them.
    # maxCategories defined so features with > maxCategories distinct values are treated
    # as continuous.
    featureIndexer = VectorIndexer(inputCol="features", \
      outputCol="indexedFeatures",\
      maxCategories=4).fit(transformed)
    data = featureIndexer.transform(transformed)

    # no difference if #category > maxCategories, interpreted as non-categorical features
    data.show(5)

    # Split the data into training and test sets (30% held out for testing) 
    (trainingData, testData) = transformed.randomSplit([0.7, 0.3])

    # Train a DecisionTree model.
    dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

    pipeline = Pipeline(stages=[featureIndexer, dt])
    model = pipeline.fit(trainingData)
    
    print('feature importances: \n', model.stages[1].featureImportances)

    predictions = model.transform(testData)
    predictions.select("features", "label", "prediction").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", \
        metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    # use sklearn.metrics for evaluation after converting toPandas
    y_true = predictions.select("label").toPandas()
    y_pred = predictions.select("prediction").toPandas()
    r2_score = sklearn.metrics.r2_score(y_true, y_pred)
    print('r2_score: {0}'.format(r2_score))


    print ("\n#---------random forest regression -------------------------------------")
    rf = RandomForestRegressor() # featuresCol="indexedFeatures",numTrees=2, maxDepth=2, seed=42
    pipeline_g = Pipeline(stages=[featureIndexer, rf])
    model_g = pipeline_g.fit(trainingData)
    print('trees:\n ', model_g.stages[-1].trees)
    
    predictions_g = model_g.transform(testData)
    predictions_g.select("features", "label", "prediction").show(5)

    # Select (prediction, true label) and compute test error
    rmse = evaluator.evaluate(predictions_g)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    

    print ("\n#---------gradient boosting tree regression -------------------------------------")
    rf = GBTRegressor() #numTrees=2, maxDepth=2, seed=42 
    pipeline_g = Pipeline(stages=[featureIndexer, rf])
    model_g = pipeline_g.fit(trainingData)
    print('trees:\n ', model_g.stages[-1].trees)
    
    predictions_g = model_g.transform(testData)
    predictions_g.select("features", "label", "prediction").show(5)

    # Select (prediction, true label) and compute test error
    rmse = evaluator.evaluate(predictions_g)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    
    spark.stop()


