from pyspark.sql.functions import col, regexp_replace

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import matplotlib.pyplot as plt

# Drop the unwanted columns

def k_means(df):
    data_df = dataProcessing(df)
    generateKMeansPictures(data_df)

def dataProcessing(df) :
    columns_to_drop = ["_c15", "_c16"]
    df = df.drop(*columns_to_drop)

    # Replace commas in multiple columns
    columns_to_fix = ["CO(GT)", "C6H6(GT)", "T", "RH", "AH"]
    for col_name in columns_to_fix:
        df = df.withColumn(col_name, regexp_replace(col(col_name), ",", ".").cast("double"))

    # Replace special characters in column names
    cleaned_columns = [col.replace(".", "_") for col in df.columns]
    df = df.toDF(*cleaned_columns)
    df = df.replace(-200, None)
    df = df.na.drop()

    df = df.withColumn(
        "Time", 
        col("Time").substr(1, 2).cast("int")
    )

    df.show()

    assembler = VectorAssembler(inputCols=["Time", "CO(GT)", "PT08_S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08_S2(NMHC)", "NOx(GT)", "PT08_S3(NOx)", "NO2(GT)", "PT08_S4(NO2)", "PT08_S5(O3)", "T", "RH", "AH"], outputCol="features")
    data_df = assembler.transform(df)

    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(data_df)
    data_df = scaler_model.transform(data_df)

    data_df.show()
    return data_df

def generateKMeansPictures(df):
    # Computing Silhouette for k calues from 2 to 20
    Silhouette = []
    evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="scaled_features", \
                                    metricName="silhouette", distanceMeasure="squaredEuclidean")
    for i in range(2, 21):
        kmeans_mod = KMeans(featuresCol="scaled_features", k=i)
        KMeans_fit = kmeans_mod.fit(data_df)
        predictions = KMeans_fit.transform(data_df)
        score = evaluator.evaluate(predictions)
        Silhouette.append(score)
        print("Silhouette Score: ", score)

    # Plotting the WSSSE values
    plt.plot(range(2, 21), Silhouette)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within Set Sum of Squared Errors(WSSSE)")
    plt.title("Elbow Method for Optimal k")
    plt.grid()
    plt.savefig('hdfs://master:9000/final/output/wssse.png')

    # Define the K-Means model with the optimal k value
    optimal_k = Silhouette.index(max(Silhouette)) + 2
    kmeans = KMeans(k=optimal_k, featuresCol="scaled_features", predictionCol="cluster")
    kmeans= kmeans.fit(data_df)

    # Assign the data points to clusters
    predictions = kmeans.transform(data_df)
    predictions.show()

    output = KMeans_fit.transform(data_df)
    Silhouette = evaluator.evaluate(output)
    print(f"WSSSE: {Silhouette}")

    # Converting to Pandas dataframe
    pandas_df = predictions.toPandas()

    # Visualizing the results
    plt.scatter(pandas_df["CO(GT)"], pandas_df["T"], c=pandas_df["cluster"], cmap="rainbow")
    plt.xlabel("CO(GT)")
    plt.ylabel("T")
    plt.title("K-Means Clustering")
    plt.colorbar().set_label("Cluster")
    plt.savefig('hdfs://master:9000/final/output/co_t.png')

    plt.scatter(pandas_df["NMHC(GT)"], pandas_df["T"], c=pandas_df["cluster"], cmap="rainbow")
    plt.xlabel("NMHC(GT)")
    plt.ylabel("T")
    plt.title("K-Means Clustering")
    plt.colorbar().set_label("Cluster")
    plt.savefig('hdfs://master:9000/final/output/nmhc_t.png')

    plt.scatter(pandas_df["C6H6(GT)"], pandas_df["T"], c=pandas_df["cluster"], cmap="rainbow")
    plt.xlabel("C6H6(GT)")
    plt.ylabel("T")
    plt.title("K-Means Clustering")
    plt.colorbar().set_label("Cluster")
    plt.savefig('hdfs://master:9000/final/output/c6h6_t.png')

    plt.scatter(pandas_df["NOx(GT)"], pandas_df["T"], c=pandas_df["cluster"], cmap="rainbow")
    plt.xlabel("NOx(GT)")
    plt.ylabel("T")
    plt.title("K-Means Clustering")
    plt.colorbar().set_label("Cluster")
    plt.savefig('hdfs://master:9000/final/output/nox_t.png')

    plt.scatter(pandas_df["NO2(GT)"], pandas_df["T"], c=pandas_df["cluster"], cmap="rainbow")
    plt.xlabel("NO2(GT)")
    plt.ylabel("T")
    plt.title("K-Means Clustering")
    plt.colorbar().set_label("Cluster")
    plt.savefig('hdfs://master:9000/final/output/no2_t.png')

__all__ = ['k_means'] 