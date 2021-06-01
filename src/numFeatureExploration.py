import sys
from random import random
from operator import add

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, skewness, kurtosis

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

        new_df = pd.concat([spark_describe, percs], ignore_index=True)
        new_df = new_df.round(2)  # round 2 columns

    return new_df[['summary'] + columns]

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("DataExploration")\
        .getOrCreate()

    df = spark.read.csv(path='../data/Credit.csv', sep=',', \
        encoding='UTF-8', comment=None, header=True, inferSchema=True)

    num_cols = ['Income','Limit']
    df.select(num_cols).describe().show()

    var = 'Income'
    df.select(skewness(var),kurtosis(var)).show()

    df1 = describe_pd(df, num_cols)    
    print(df1)
    
    spark.stop()


