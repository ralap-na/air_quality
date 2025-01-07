from pyspark.sql.functions import col, regexp_replace, avg, when
from matplotlib.ticker import MaxNLocator
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd

def coRelativePic(df):

    create_relative_pic(df, 'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)')
    # create_relative_pic(df, 'NMHC(GT)', 'CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)')
    # create_relative_pic(df, 'C6H6(GT)', 'CO(GT)', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)')
    # create_relative_pic(df, 'NOx(GT)', 'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NO2(GT)')
    # create_relative_pic(df, 'NO2(GT)', 'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)')

def replaceInvalid(df):
    new_column_names = [col.replace(".", "_") for col in df.columns]

    df_renamed = df.toDF(*new_column_names)
    df_renamed = df_renamed.drop('_c15', '_c16')
    df_cleaned = df_renamed.dropna(how="all")
    df_cleaned = df_cleaned.select([regexp_replace(col(c), ',', '.').alias(c) for c in df_cleaned.columns])

    columns_to_process = [col for col in df_cleaned.columns if col not in ['Time', 'Date']]

    avg_values = df_cleaned.select(
    [avg(when(col(c) != -200, col(c))).alias(c) for c in columns_to_process]
).collect()[0]

    for c, avg_val in zip(columns_to_process, avg_values):
        df_cleaned = df_cleaned.withColumn(c, when(col(c) == -200, avg_val).otherwise(col(c)))
    return df_cleaned

def toPanda(df_cleaned):
    df_cleaned = df_cleaned.select(col('CO(GT)'),col('PT08_S1(CO)'), col('NMHC(GT)'), col('C6H6(GT)'), col('NOx(GT)'), col('NO2(GT)'), col('T'), col('RH'), col('AH'))
    df_pandas = df_cleaned.toPandas()

    cols_to_convert = df_pandas.columns

    df_pandas[cols_to_convert] = df_pandas[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    return df_pandas

def create_relative_pic(df, col1, col2, col3, col4, col5):
    
    df_cleaned = replaceInvalid(df)
    df_pandas = toPanda(df_cleaned) 

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.scatter(df_pandas[col1], df_pandas[col2], color='blue')
    plt.title(col1 + " vs " + col2)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=10))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=10))

    plt.subplot(2, 2, 2)
    plt.scatter(df_pandas[col1], df_pandas[col3], color='green')
    plt.title(col1 + " vs " + col3)
    plt.xlabel(col1)
    plt.ylabel(col3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=10))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=10))

    plt.subplot(2, 2, 3)
    plt.scatter(df_pandas[col1], df_pandas[col4], color='red')
    plt.title(col1 + " vs " + col4)
    plt.xlabel(col1)
    plt.ylabel(col4)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=10))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=10))

    plt.subplot(2, 2, 4)
    plt.scatter(df_pandas[col1], df_pandas[col5], color='purple')
    plt.title(col1 + " vs " + col5)
    plt.xlabel(col1)
    plt.ylabel(col5)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=10))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=10))

    plt.tight_layout()
    plt.savefig('hdfs://master:9000/final/output/' + col1 + ".png")

def heatMap(df):
    df_cleaned = replaceInvalid(df)
    df_pandas = toPanda(df_cleaned)

    df_haep = df_pandas.apply(pd.to_numeric, errors='coerce')

    plt.figure(figsize=(12, 8))

    sns.heatmap(df_haep.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

    plt.savefig('hdfs://master:9000/final/output/heatmap.png')

__all__ = ['heatMap', 'coRelativePic'] 