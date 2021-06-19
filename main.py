from math import radians, sin, cos

from pyspark import SparkContext
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType, TimestampType
import pyspark.sql.functions as F

JFK_LAT = radians(40.6413)
JFK_LON = radians(-73.7781)
R = 6371
MAX_DISTANT = 2


def convert_to_hour(df):
    hour = F.udf(lambda x: x.hour, IntegerType())
    df = df.withColumn("hour_pickup", hour("lpep_pickup_datetime"))
    df = df.withColumn("hour_dropoff", hour("Lpep_dropoff_datetime"))
    return df


def convert_to_day(df):
    df = df.withColumn("dayofweek_pickup", F.dayofweek("lpep_pickup_datetime"))
    df = df.withColumn("dayofweek_dropoff", F.dayofweek("Lpep_dropoff_datetime"))
    return df


def ohe(df, input_cols, output_cols):
    encoder = OneHotEncoder(inputCols=input_cols, outputCols=output_cols)
    model = encoder.fit(df)
    df_encoded = model.transform(df)
    return df_encoded


def pre_process(df):
    df = df.withColumn('lpep_pickup_datetime', df['lpep_pickup_datetime'].cast(TimestampType()))
    df = df.withColumn('Lpep_dropoff_datetime', df['Lpep_dropoff_datetime'].cast(TimestampType()))

    df.printSchema()
    return df


def distance_airport(df):
    df = df.withColumn("rlat_pickup", F.radians(F.col("Pickup_latitude")))\
        .withColumn("rlon_pickup", F.radians(F.col("Pickup_longitude")))\
        .withColumn("rlat_dropoff", F.radians(F.col("Dropoff_latitude")))\
        .withColumn("rlon_dropoff", F.radians(F.col("Dropoff_longitude")))\
        .withColumn("dist_JFK_pickup",
                    F.acos(
                        F.sin(F.col("rlat_pickup")) * sin(JFK_LAT)
                        + F.cos(F.col("rlat_pickup")) * cos(JFK_LAT) * F.cos(F.col("rlon_pickup") - JFK_LON)
                    ) * R)\
        .withColumn("dist_JFK_dropoff",
                    F.acos(
                        F.sin(F.col("rlat_pickup")) * sin(JFK_LAT)
                        + F.cos(F.col("rlat_dropoff")) * cos(JFK_LAT) * F.cos(F.col("rlon_dropoff") - JFK_LON)) * R
                    )\
        .withColumn("is_JFK_airport",
                    F.when((F.col("dist_JFK_pickup") <= MAX_DISTANT) | (F.col("dist_JFK_dropoff") <= MAX_DISTANT), 1)
                    .otherwise(0)
                    )\
        .drop("rlat_pickup", "rlon_pickup", "rlat_dropoff", "rlon_dropoff", "dist_JFK_pickup", "dist_JFK_dropoff")
    return df


def main():
    sc = SparkContext("local", "NYC Taxi app")
    sqlContext = SQLContext(sc)

    df = sqlContext.read.format('com.databricks.spark.csv') \
        .options(header='true', inferschema='true') \
        .load('data/green_tripdata_2013-09.csv')
    df.registerTempTable("trips")

    df = pre_process(df)
    df_hour = convert_to_hour(df)
    df_day = convert_to_day(df_hour)

    input_cols = ["hour_pickup", "hour_dropoff", "dayofweek_pickup", "dayofweek_dropoff"]
    output_cols = ["hour_pickup_vec", "hour_dropoff_vec", "dayofweek_pickup_vec", "dayofweek_dropoff_vec"]
    df_encoded = ohe(df_day, input_cols, output_cols)

    df_duration = df_encoded.withColumn(
        'duration_trip',
        F.unix_timestamp("Lpep_dropoff_datetime") - F.unix_timestamp('lpep_pickup_datetime')
    )

    df_result = distance_airport(df_duration)
    df_result.show()



if __name__ == '__main__':
    main()
