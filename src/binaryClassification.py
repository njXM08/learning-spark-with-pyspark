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

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]), r[-1]]).toDF(['features', 'label'])

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("BinaryClassification")\
        .getOrCreate()

    # infer schema in sql-favor: https://www.kaggle.com/janiobachmann/bank-marketing-dataset
    df = spark.read.format('com.databricks.spark.csv').\
      options(header='true', inferschema='true').\
      load("../data/bank.csv", header=True);

    # head(5) and schema
    df.drop('day','month','poutcome').show(5)
    df.printSchema()
    
    
    # convert categorical columns to one-hot vectors
    catcols = ['job','marital','education','default', 'housing','loan','contact','poutcome']
    num_cols = ['balance', 'duration','campaign','pdays','previous',] 
    labelCol = 'deposit'
    data = get_dummy(df, None, catcols, num_cols, labelCol)
    data.show(5)

    # Index labels, adding metadata to the label column 
    labelIndexer = StringIndexer(inputCol='label', outputCol='indexedLabel').fit(data)
    labelIndexer.transform(data).show(5, True)

    # Automatically identify categorical features, and index them. 
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", \
        maxCategories=4).fit(data)
    featureIndexer.transform(data).show(5, True)

    # Split the data into training and test sets (40% held out for testing)
    (trainingData, testData) = data.randomSplit([0.6, 0.4])
    trainingData.show(5,False)
    testData.show(5,False)

    logr = LogisticRegression(featuresCol='indexedFeatures', labelCol='indexedLabel')

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol= "predictedLabel", labels=labelIndexer.labels)

    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, logr, labelConverter])
    model = pipeline.fit(trainingData)

    predictions = model.transform(testData)
    predictions.select("features","label","predictedLabel").show(5)

    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", \
        predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    
    # logistic regression model check
    lrModel = model.stages[2]
    trainingSummary = lrModel.summary

    trainingSummary.roc.show(5)

    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

    '''
    # Set the model threshold to maximize F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head(5)
    bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)'])\
      .select('threshold').head()['threshold']
    # lr.setThreshold(bestThreshold)
    '''

    class_temp = predictions.select("label").groupBy("label")\
    .count().sort('count', ascending=False).toPandas()

    class_temp = class_temp["label"].values.tolist()
    class_names = map(str, class_temp)
    print(class_names, class_temp)

    y_true = predictions.select("label")
    y_true.show(5)
    y_true = y_true.toPandas()

    y_pred = predictions.select("predictedLabel")
    y_pred.show(5)
    y_pred = y_pred.toPandas()
    cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_temp)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_temp, normalize=True, title='Confusion matrix, without normalization')
    plt.show()

    spark.stop()


