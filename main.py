
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext("local", "NYC Taxi app")
sqlContext = SQLContext(sc)
df = sqlContext.read.format('com.databricks.spark.csv')\
    .options(header='true', inferschema='true')\
    .load('data/green_tripdata_2013-09.csv')
df.registerTempTable("trips")
sqlContext.sql("select * from trips limit 5").show()



