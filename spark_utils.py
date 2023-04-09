from pyspark import SparkConf
from pyspark.sql import SparkSession


def create_spark_session() -> SparkSession:
    conf = SparkConf()\
        .set("spark.driver.memory", "2g")\
        .set("spark.sql.autoBroadcastJoinThreshold", "-1")\
        .set("spark.driver.extraJavaOptions", "-Dlog4j.configuration=log4j2.properties")

    spark_session = SparkSession\
        .builder\
        .master("local[4]")\
        .config(conf=conf)\
        .appName("Aggregate Transform Tutorial") \
        .getOrCreate()

    spark_session.sparkContext.setCheckpointDir("checkpoint")

    return spark_session
