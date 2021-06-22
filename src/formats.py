import sys
from random import random
from operator import add

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
import pyspark.sql.functions as F

if __name__ == "__main__":
    '''spark1 = SparkSession\
        .builder\
        .appName("dataframe manipulation")\
        .getOrCreate()
    '''

    # avro format needs external jar: must match to the installed spark version
    # example: spark 3.1.1
    spark = SparkSession\
        .builder\
        .appName("dataframe manipulation")\
        .config("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.1.1")\
        .getOrCreate()

    # 1). read csv/json
    dp = pd.read_csv('../data/data.csv')

    ds = spark.read.csv(path='../data/data.csv', sep=',', \
        encoding='UTF-8', comment=None, header=True, inferSchema=True)
    
    print('two columns: ', dp[['name','age']].head(4))
    ds[['name','age']].show(4)

    # columns for pd.df or spark.df
    print('column names: ')
    print(dp.columns)
    print(ds.columns)

    # dtypes: pd interprets col with "NA" as object, 
    # ps inteprets col with "NA" as string
    print('column dtypes: ')
    print(dp.dtypes)
    print(ds.dtypes)
    
    # 2). write to and read from parquet format: built-in support
    ds.write.mode("overwrite").parquet("../data/parquet/")
    
    ds_parquet = spark.read.parquet("../data/parquet/")
    print("read-in parquet: ")
    ds_parquet.show(4)
    
    # 3). no built-in support for avro
    ds.write.mode("overwrite").format('avro').save("../data/avro/tst.avro")

    ds_avro = spark.read.format('avro').load("../data/avro/tst.avro")
    print("read-in avro: ")
    ds_avro.show(4)

    # 4). built-in support for orc
    ds.write.mode("overwrite").orc("../data/orc/")

    ds_orc = spark.read.orc("../data/orc/")
    print("read-in orc: ")
    ds_orc.show(4)

    spark.stop()