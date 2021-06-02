import sys
from random import random
from operator import add

from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
import pyspark.sql.functions as F

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("dataframe manipulation")\
        .getOrCreate()

    # 1). create df from list: three cols - str, int, int
    my_list = [['a', 1, 2], ['b', 2, 3],['c', 3, 4]]
    col_name = ['A', 'B', 'C']

    pd_df = pd.DataFrame(my_list,columns= col_name)

    # default value makes the list as rows.
    #   A  B  C
    #0  a  1  2
	#1  b  2  3
	#2  c  3  4
    print('pandas df from list: ', pd_df)

    df = spark.createDataFrame(my_list, col_name).show()
    
    # 2). create df from dict
    d = {'A': [0, 1, 0], 'B': [1, 0, 1], 'C': [1, 0, 0]}
    col_name = ['A', 'B', 'C']

    pd_df = pd.DataFrame(d, columns= col_name)

    # default value makes the list as rows.
    #   A  B  C
    #0  a  1  2
	#1  b  2  3
	#2  c  3  4
    print('pandas df from dict: ', pd_df)

    # (list as values, dict-keyas as col_name)
    df = spark.createDataFrame(np.array(list(d.values())).T.tolist(), list(d.keys())).show()
    
    # 3). read csv/json
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
    # ps inteprets col with "NA" as tring
    print('column dtypes: ')
    print(dp.dtypes)
    print(ds.dtypes)
    
    # 4). Fill-Null
    my_list = [['male', 1, None], ['female', 2, 3],['male', 3, 4]]
    dp = pd.DataFrame(my_list,columns=['A', 'B', 'C'])
    ds = spark.createDataFrame(my_list, ['A', 'B', 'C'])

    #print(dp.dtypes)
    #print(ds.dtypes)   # col is "bigInt" irrespective of "Null"

    dp = dp.fillna(-99)
    ds = ds.fillna(-99)
    
    # 5) replace values in a column
    # pandas' df -> caution: you need to chose specific col
    dp.A.replace(['male', 'female'],[1, 0], inplace=True)

    # caution: Mixed type replacements are not supported
    ds1 = ds.na.replace(['male','female'],['1','0'])
    #ds1.show()
    #print(ds1.columns)

    # rename columns
    dp.columns = ['a', 'b', 'c']
    ds.toDF('a', 'b', 'c').show()

    # rename with mapping dict
    mapping = {'a':'age', 'b':'builderNum', 'c': 'C'}
    dp = dp.rename(columns = mapping)
    print('rename of pandas df: ', dp.head())

    mapping1 = {'A': 'age', 'B': 'builderNum'}
    new_names = [mapping1.get(col,col) for col in ds.columns]
    # ds.toDF(*new_names).show(4)
    
    # withColumnRenamed to rename one column -> no change to ds
    ds.withColumnRenamed('A', 'age').show()

    # drop columns 
    drop_col_names = ['C']
    print(dp.drop(drop_col_names, axis=1).head())
    # ds.drop(*drop_col_names).show()

    print(dp[dp.age<2].head())
    ds[ds.B<=2].show()

    # 6). new column examples: new feature
    dp['age_p_bn'] = dp['age']/	sum(dp.age)
    dp['age+10'] = dp.age.apply(lambda x: x+10)
    print(dp)

    ds.withColumn('B_p_sC', ds.B/ds.groupBy().agg(F.sum('C')).collect()[0][0]).show()

    ds.withColumn('exp_B', F.log(ds.B+200)).show()
    
    # 7). join
    leftp = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
      'B': ['B0', 'B1', 'B2', 'B3'],
      'C': ['C0', 'C1', 'C2', 'C3'],
      'D': ['D0', 'D1', 'D2', 'D3']},
      index=[0, 1, 2, 3])
    rightp = pd.DataFrame({'A': ['A0', 'A1', 'A6', 'A7'],
      'F': ['B4', 'B5', 'B6', 'B7'],
      'G': ['C4', 'C5', 'C6', 'C7'],
      'H': ['D4', 'D5', 'D6', 'D7']},
      index=[4, 5, 6, 7])

    lefts = spark.createDataFrame(leftp)
    rights = spark.createDataFrame(rightp)

    lefts.show()
    rights.show()

    print('left-join: ')
    newpl = leftp.merge(rightp,on='A',how='left') 
    print(newpl)

    print('right-join: ')
    newpr = leftp.merge(rightp,on='A',how='right') 
    print(newpr)
    
    print('inner-join: ')
    newpi = leftp.merge(rightp,on='A',how='inner') 
    print(newpi)

    print('full-join: ')
    newpf = leftp.merge(rightp, on='A', how='outer') 
    print(newpf)
    
    lefts.join(rights,on='A',how='left').orderBy('A',ascending=True).show()
    
    # 8). concat columns
    my_list = [('a', 2, 3), ('b', 5, 6), ('c', 8, 9), 
    ('a', 4, 5)]
    col_name = ['col1', 'col2', 'col3']

    dp = pd.DataFrame(my_list,columns=col_name)
    ds = spark.createDataFrame(my_list,schema=col_name)
    dp['concat'] = dp.apply(lambda x:'%s%s'%(x['col1'],x['col2']),axis=1)
    print('concated cols: ', dp)

    ds.withColumn('concat',F.concat('col1','col2')).show()
    
    # groupBy
    print(dp.groupby(['col1']).agg({'col2':'min','col3':'mean'}))

    ds.groupBy(['col1']).agg({'col2': 'min', 'col3': 'avg'}).show()
    
    # 9) windowing
    d = {'A':['a','b','c','d'],'B':['m','m','n','n'],'C':[1,2,3,6]}
    dp = pd.DataFrame(d)
    ds = spark.createDataFrame(dp)

    dp['rank'] = dp.groupby('B')['C'].rank('dense',ascending=False)
    print(dp)

    # rank
    from pyspark.sql.window import Window
    w = Window.partitionBy('B').orderBy(ds.C.desc())
    ds = ds.withColumn('rank',F.rank().over(w))
    
    # rank vs. rank_dense: difference for multiple equal values
    # 1 1 3 4 4 5 vs. 1 1 2 3 3 4 
    # 
    '''
    dp['Rank_dense'] = dp['Score'].rank(method='dense',ascending =False)
    dp['Rank'] = dp['Score'].rank(method='min',ascending =False)
    
    import pyspark.sql.functions as F
    from pyspark.sql.window import Window
    w = Window.orderBy(ds.Score.desc())
    ds = ds.withColumn('Rank_spark_dense',F.dense_rank().over(w))
    ds = ds.withColumn('Rank_spark',F.rank().over(w))
    ds.show()
    '''
    
    spark.stop()