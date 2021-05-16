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
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorIndexer
from pyspark.sql.functions import col

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

# lambda in rdd.map for "map": good for large features
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]), r[-1]]).toDF(['features', 'label'])

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("LinearRegression")\
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

    # Linear regressor
    lr = LinearRegression()

    pipeline = Pipeline(stages=[featureIndexer, lr])
    model = pipeline.fit(trainingData)
    
    # model summary
    modelsummary(model.stages[-1])

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


    # https://spark.apache.org/docs/latest/ml-classification-regression.html#generalized-linear-regression
    # Generalized Gaussian distribution instead of normal distribution
    # in the context of least-squaures
    print ("#---------GLR regression -------------------------------------")
    glr = GeneralizedLinearRegression(family="gaussian", link="identity",\
        maxIter=10, regParam=0.3)
    pipeline_g = Pipeline(stages=[featureIndexer, glr])
    model_g = pipeline_g.fit(trainingData)
    
    # model summary
    # modelsummary(model_g.stages[-1])

    predictions_g = model_g.transform(testData)
    predictions_g.select("features", "label", "prediction").show(5)

    # Select (prediction, true label) and compute test error
    rmse = evaluator.evaluate(predictions_g)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    spark.stop()


