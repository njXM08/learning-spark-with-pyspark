# learning-spark-with-pyspark
Learning pyspark for ETL and ML based on the note/code https://runawayhorse001.github.io/LearningApacheSpark/pyspark.pdf 

Testing versions:
* python: 3.6.7
* spark: 3.1.1-bin-hadoop2.7

## Machine learning
### regression 
* linear regression: Advertising.csv
* regularized linear regression
### classification
* binary classification: bank.csv (bank marketing dataset)
* multi-class classification: WineData.csv (wine quality dataset, by binning quality values)
### clustering
* k-means clustering: iris.csv (iris type dataset)
### Recency, frequency, monetary: customer insight
* RFM analysis: onlineRetail2.csv
### Elephas for tf.Keras MNIST 
* Google colab notebook: use Elephas to train tf.keras model in conjunction with pyspark
### Feature engineering
* createDataFrame, VectorAssembler, StringIndexer, OneHotEncoder, Scaler
### numerical feature exploration
### dataframe processing: pandas vs. pyspark
* create dataframe, read csv, columns' renaming, drop-na, fill-na, groupby, join


