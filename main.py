from pyspark.sql import SparkSession
from machinelearn import *
from relativepicture import *
from k_means import *

spark = SparkSession.builder \
    .appName("final") \
    .getOrCreate()

spark = SparkSession.builder.appName("final").getOrCreate()

df = spark.read.csv("hdfs://master:9000/final/AirQualityUCI.csv", header=True, inferSchema=True, sep=";")

k_means(df)
heatMap(df)
linearRegression(df)
randomForest(df)
decisionTree(df)
supportVectorRegression(df)
