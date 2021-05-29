import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

start_m =100
wager = 5
bets = 100
trials = 1000
trans = np.vectorize(lambda t: -wager if t <=0.51 else wager)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1,1,1)
end_m = []
for i in range(trials):
  money = reduce(lambda c, x: c + [c[-1] + x], trans(np.random.random(bets)), [start_m])
  end_m.append(money[-1])

plt.plot(money)
plt.ylabel('Player Money in $')
plt.xlabel('Number of bets')
plt.title(("John starts the game with $ %.2f and ends with $ %.2f")%(start_m, sum(end_m)/len(end_m)))
plt.show()


from pyspark.sql import SparkSession
from pyspark.mllib.random import RandomRDDs
import pandas as pd

spark = SparkSession\
      .builder\
      .appName("Monte-Carlo simulation")\
      .getOrCreate()

spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")
stock_ds = spark.read.format('com.databricks.spark.csv').\
    options(header='true', inferschema='true').\
    load("BABA.csv", header=True);

stock_ds.show(5)
stock = pd.read_csv("BABA.csv")
stock['Date'] = pd.to_datetime(stock['Date'])
print(stock.tail(5))

# from matplotlib import ticker
# data visualization
width = 10
height = 6
data = stock
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(1,1,1)
ax.plot(data.Date, data.Close, label='Close')
ax.plot(data.Date, data.High, label='High')
# ax.plot(data.Date, data.Low, label='Low')
ax.set_xlabel('Date')
ax.set_ylabel('price ($)')
ax.legend()
ax.set_title('Stock price: BABA', y=1.01)
#plt.xticks(rotation=70)
plt.show()
# Plot everything by leveraging the very powerful matplotlib package
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(1,1,1)
ax.plot(data.Date, data.Volume, label='Volume')
#ax.plot(data.Date, data.High, label='High')
# ax.plot(data.Date, data.Low, label='Low')
ax.set_xlabel('Date')
ax.set_ylabel('Volume')
ax.legend()
ax.set_title('Stock volume: BABA', y =1.01)
#plt.xticks(rotation=70)
plt.show()

#---------- Compound Annual Growth Rate
days = (stock.Date.iloc[-1] - stock.Date.iloc[0]).days
cagr = ((((stock['Adj Close'].iloc[-1]) / stock['Adj Close'].iloc[0])) ** (365.0/days)) - 1 
print ('CAGR =',str(round(cagr,4)*100)+"%")
mu = cagr

# ------------- annual volatility
stock['Returns'] = stock['Adj Close'].pct_change()
vol = stock['Returns'].std()*np.sqrt(252)


import math 
from pyspark.sql.functions import round

# get spark context from session
sc = spark.sparkContext

S = stock['Adj Close'].iloc[-1] #starting stock price (i.e. last available real stock price)
T = 5 #Number of trading days
mu = cagr   #Return
vol = vol   #Volatility
trials = 10000

# create 'trials' normally distributed vectors: size-[trials, T] 
mat = RandomRDDs.normalVectorRDD(sc, trials, T, seed=1)

# convert from RDD matrix to numpy matrix
mat_n = np.mat(mat.collect())

from pyspark.mllib.linalg import DenseVector

a = mu/T
b = vol/math.sqrt(T)
v = mat.map(lambda x: a + (b - a)* x)

# convert RDD to dataframe: in single feature, or in features
#df = v.map(lambda x: (x, )).toDF()
df = mat.map(lambda a : a.tolist()).toDF()

df.show(5)

from pyspark.sql.functions import lit
S = stock['Adj Close'].iloc[-1]
price = df.withColumn('init_price' ,lit(S))
price.show(5)

# duplicate a column with a new column name
price = price.withColumn('day_0', price['init_price'])
price.show(5)


from pyspark.sql.functions import round, col

for name in price.columns[:-2]: 
  price = price.withColumn('day'+name, round(col(name)*col('init_price'),2))
  price = price.withColumn('init_price',col('day'+name))

price.show(5)

selected_col = [name for name in price.columns if 'day_' in name]
simulated = price.select(selected_col)
simulated.describe().show()


data_plt = simulated.toPandas()
days = pd.date_range(stock['Date'].iloc[-1], periods= T+1,freq='B').date
width = 10
height = 6
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(1,1,1)
days = pd.date_range(stock['Date'].iloc[-1], periods= T+1,freq='B').date
for i in range(trials):
  plt.plot(days, data_plt.iloc[i])

ax.set_xlabel('Date')
ax.set_ylabel('price ($)')
ax.set_title('Simulated Stock price: BABA', y=1.01)
plt.show()

spark.stop()