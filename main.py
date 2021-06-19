from pyspark import SparkContext
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType, TimestampType
import pyspark.sql.functions as F


def convert_to_hour(df):
    hour = F.udf(lambda x: x.hour, IntegerType())
    df = df.withColumn("hour_pickup", hour("lpep_pickup_datetime"))
    df = df.withColumn("hour_dropoff", hour("Lpep_dropoff_datetime"))
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
    df_hour = convert_to_hour(df)
    df_hour.show()

    input_cols = ["hour_pickup", "hour_dropoff"]
    output_cols = ["hour_pickup_vec", "hour_dropoff_vec"]
    df_encoded = ohe(df_hour, input_cols, output_cols)
    df_encoded.show()


def main():
    sc = SparkContext("local", "NYC Taxi app")
    sqlContext = SQLContext(sc)

    df = sqlContext.read.format('com.databricks.spark.csv') \
        .options(header='true', inferschema='true') \
        .load('data/green_tripdata_2013-09.csv')
    df.registerTempTable("trips")
    sqlContext.sql("select * from trips limit 5").show()
    pre_process(df)


if __name__ == '__main__':
    main()
