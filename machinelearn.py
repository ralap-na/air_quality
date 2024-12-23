from pyspark.sql.functions import col, regexp_replace, avg, when

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

def linearRegression(df):

    df_cleaned = replaceInvalid(df)
    df_pandas = toPanda(df_cleaned)
    X_train, X_test, y_train, y_test = createTrainAndTest(df_pandas)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("LinearRegression R²:", r2_score(y_test, y_pred))
    print("LinearRegression MSE:", mean_squared_error(y_test, y_pred))

def randomForest(df):

    df_cleaned = replaceInvalid(df)
    df_pandas = toPanda(df_cleaned)
    X_train, X_test, y_train, y_test = createTrainAndTest(df_pandas)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)

    print("RandomForest R²:", r2_score(y_test, y_pred_rf))
    print("RandomForest MSE:", mean_squared_error(y_test, y_pred_rf))

def decisionTree(df):

    df_cleaned = replaceInvalid(df)
    df_pandas = toPanda(df_cleaned)
    X_train, X_test, y_train, y_test = createTrainAndTest(df_pandas)

    dt = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    dt.fit(X_train, y_train)

    y_pred_dt = dt.predict(X_test)
    print("DecisionTree R²:", r2_score(y_test, y_pred_dt))
    print("DecisionTree MSE:", mean_squared_error(y_test, y_pred_dt))

def supportVectorRegression(df):
    df_cleaned = replaceInvalid(df)
    df_pandas = toPanda(df_cleaned)
    X_train, X_test, y_train, y_test = createTrainAndTest(df_pandas)

    svr_model = SVR(kernel='linear', C=1.0)
    svr_model.fit(X_train, y_train) 

    y_pred = svr_model.predict(X_test)
    print("SVR R²:", r2_score(y_test, y_pred))
    print("SVR MSE:", mean_squared_error(y_test, y_pred))

def logisticRegression(df):
    df_cleaned = replaceInvalid(df)
    df_pandas = toPanda(df_cleaned)
    X_train, X_test, y_train, y_test = createTrainAndTest(df_pandas)

    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)
    print("logisticRegression R²:", r2_score(y_test, y_pred))
    print("logisticRegression MSE:", mean_squared_error(y_test, y_pred))

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
    df_cleaned = df_cleaned.select(col('CO(GT)'), col('PT08_S1(CO)'), col('NMHC(GT)'), col('C6H6(GT)'), col('NOx(GT)'), col('NO2(GT)'), col('T'), col('RH'), col('AH'))
    df_pandas = df_cleaned.toPandas()

    cols_to_convert = df_pandas.columns

    df_pandas[cols_to_convert] = df_pandas[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    return df_pandas

def createTrainAndTest(df_pandas):
    # X = df_pandas[['NMHC(GT)','PT08_S1(CO)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']]
    X = df_pandas[['PT08_S1(CO)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']]
    # X = df_pandas[['PT08_S1(CO)', 'C6H6(GT)']]
    # X = df_pandas[['C6H6(GT)', 'PT08_S1(CO)', 'NOx(GT)']]
    y = df_pandas['CO(GT)']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test

__all__ = ['linearRegression', 'randomForest', 'decisionTree', 'supportVectorRegression', 'logisticRegression'] 