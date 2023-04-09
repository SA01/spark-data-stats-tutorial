from typing import Callable
import time

from pyspark.sql import DataFrame, SparkSession, Observation
import pyspark.sql.functions as F

from spark_utils import create_spark_session


def calculate_pickup_locations_popularity(zones_data: DataFrame) -> Callable[[DataFrame], DataFrame]:
    def calculate_pickup_locations_popularity_inner(df: DataFrame) -> DataFrame:
        return df\
            .groupBy("PULocationID")\
            .agg(
                F.count(F.lit(1)).alias("trips"),
                F.avg("trip_distance").alias("avg_distance"),
                F.avg("total_amount").alias("avg_total_amount")
            ) \
            .join(zones_data, (F.col("PULocationID") == F.col("LocationID"))) \
            .select("Zone", "Borough", "trips", "avg_distance", "avg_total_amount")

    return calculate_pickup_locations_popularity_inner


def calculate_routes_popularity(zones_data: DataFrame) -> Callable[[DataFrame], DataFrame]:
    def calculate_routes_popularity_inner(df: DataFrame) -> DataFrame:
        return df.groupBy("PULocationID", "DOLocationID") \
            .agg(
                F.count(F.lit(1)).alias("trips"),
                F.avg("trip_distance").alias("avg_distance"),
                F.avg("total_amount").alias("avg_total_amount")
            ) \
            .join(zones_data, (F.col("PULocationID") == F.col("LocationID"))) \
            .withColumnRenamed("Zone", "Pickup-Zone") \
            .withColumnRenamed("Borough", "Pickup-Borough") \
            .drop("LocationID") \
            .join(zones_data, (F.col("DOLocationID") == F.col("LocationID"))) \
            .withColumnRenamed("Zone", "Dropoff-Zone") \
            .withColumnRenamed("Borough", "Dropoff-Borough") \
            .drop("LocationID") \
            .select(
                "Pickup-Zone", "Pickup-Borough", "Dropoff-Zone", "Dropoff-Borough",
                "trips", "avg_distance", "avg_total_amount"
            )

    return calculate_routes_popularity_inner


def run_with_logging(spark: SparkSession):
    exec_time = time.strftime("exec_time=%Y%m%d%H%M%S", time.localtime())

    zones_data = spark.read.option("header", "true").csv("data/metadata/taxi+_zone_lookup.csv")
    print(f"Loaded {zones_data.count()} zones")

    trips_data = spark.read.parquet("data/trips/*/*.parquet")
    print(f"Loaded {trips_data.count()} rows")

    popular_pickups = calculate_pickup_locations_popularity(zones_data=zones_data)(trips_data)
    print(f"Popular Pickups data size: {popular_pickups.count()}")
    popular_pickups.write.option("header", "true").csv(f"output/popular_pickups/{exec_time}")

    popular_routes = calculate_routes_popularity(zones_data=zones_data)(trips_data)
    print(f"Popular routes size: {popular_routes.count()}")
    popular_routes.write.option("header", "true").csv(f"output/popular_routes/{exec_time}")


def run_with_cache(spark: SparkSession):
    exec_time = time.strftime("exec_time=%Y%m%d%H%M%S", time.localtime())

    zones_data = spark.read.option("header", "true").csv("data/metadata/taxi+_zone_lookup.csv")
    print(f"Loaded {zones_data.count()} zones")

    trips_data = spark.read.parquet("data/trips/*/*.parquet")
    print(f"Loaded {trips_data.count()} rows")

    popular_pickups = calculate_pickup_locations_popularity(zones_data=zones_data)(trips_data).cache()
    print(f"Popular Pickups data size: {popular_pickups.count()}")
    popular_pickups.write.option("header", "true").csv(f"output/popular_pickups/{exec_time}")

    popular_routes = calculate_routes_popularity(zones_data=zones_data)(trips_data).cache()
    print(f"Popular routes size: {popular_routes.count()}")
    popular_routes.write.option("header", "true").csv(f"output/popular_routes/{exec_time}")


def run_with_checkpoint(spark: SparkSession):
    exec_time = time.strftime("exec_time=%Y%m%d%H%M%S", time.localtime())

    zones_data = spark.read.option("header", "true").csv("data/metadata/taxi+_zone_lookup.csv")
    print(f"Loaded {zones_data.count()} zones")

    trips_data = spark.read.parquet("data/trips/*/*.parquet")
    print(f"Loaded {trips_data.count()} rows")

    popular_pickups = calculate_pickup_locations_popularity(zones_data=zones_data)(trips_data).checkpoint(eager=True)
    print(f"Popular Pickups data size: {popular_pickups.count()}")
    popular_pickups.write.option("header", "true").csv(f"output/popular_pickups/{exec_time}")

    popular_routes = calculate_routes_popularity(zones_data=zones_data)(trips_data).checkpoint(eager=True)
    print(f"Popular routes size: {popular_routes.count()}")
    popular_routes.write.option("header", "true").csv(f"output/popular_routes/{exec_time}")


def run_with_no_logging(spark: SparkSession):
    exec_time = time.strftime("exec_time=%Y%m%d%H%M%S", time.localtime())

    zones_data = spark.read.option("header", "true").csv("data/metadata/taxi+_zone_lookup.csv")
    trips_data = spark.read.parquet("data/trips/*/*.parquet")

    popular_pickups = calculate_pickup_locations_popularity(zones_data=zones_data)(trips_data)
    popular_pickups.write.option("header", "true").csv(f"output/popular_pickups/{exec_time}")

    popular_routes = calculate_routes_popularity(zones_data=zones_data)(trips_data)
    popular_routes.write.option("header", "true").csv(f"output/popular_routes/{exec_time}")


def run_with_observers(spark: SparkSession):
    exec_time = time.strftime("exec_time=%Y%m%d%H%M%S", time.localtime())

    raw_data_observation = Observation("raw data")
    popular_pickups_observation = Observation("popular pickups data")
    popular_routes_observation = Observation("popular routes data")

    zones_data = spark.read.option("header", "true").csv("data/metadata/taxi+_zone_lookup.csv")
    trips_data = spark.read.parquet("data/trips/*/*.parquet")\
        .observe(raw_data_observation, F.count(F.lit(1)).alias("num_rows"))

    popular_pickups = calculate_pickup_locations_popularity(zones_data=zones_data)(trips_data)\
        .observe(popular_pickups_observation, F.count(F.lit(1)).alias("num_rows"))
    popular_pickups.write.option("header", "true").csv(f"output/popular_pickups/{exec_time}")

    popular_routes = calculate_routes_popularity(zones_data=zones_data)(trips_data)\
        .observe(popular_routes_observation, F.count(F.lit(1)).alias("num_rows"))
    popular_routes.write.option("header", "true").csv(f"output/popular_routes/{exec_time}")

    print(f"Raw data count: {raw_data_observation.get}")
    print(f"popular pickups data size: {popular_pickups_observation.get}, "
          f"popular routes data size: {popular_routes_observation.get}")


def run_with_accumulators(spark: SparkSession):
    exec_time = time.strftime("exec_time=%Y%m%d%H%M%S", time.localtime())

    zones_count_acc = spark.sparkContext.accumulator(0)
    trips_count_acc = spark.sparkContext.accumulator(0)
    pickup_loc_count_acc = spark.sparkContext.accumulator(0)
    routes_count_acc = spark.sparkContext.accumulator(0)

    @F.udf
    def increment_zones_count_accumulator(some_col_val):
        zones_count_acc.add(1)
        return some_col_val

    @F.udf
    def increment_trips_count_accumulator(some_col_val):
        trips_count_acc.add(1)
        return some_col_val

    @F.udf
    def increment_pickup_loc_count_accumulator(some_col_val):
        pickup_loc_count_acc.add(1)
        return some_col_val

    @F.udf
    def increment_routes_count_accumulator(some_col_val):
        routes_count_acc.add(1)
        return some_col_val

    zones_data = spark.read.option("header", "true").csv("data/metadata/taxi+_zone_lookup.csv")\
        .withColumn("LocationID", increment_zones_count_accumulator(F.col("LocationID")))\

    trips_data = spark.read.parquet("data/trips/*/*.parquet") \
        .withColumn("PULocationID", increment_trips_count_accumulator(F.col("PULocationID")))

    popular_pickups = calculate_pickup_locations_popularity(zones_data=zones_data)(trips_data)\
        .withColumn("trips", increment_pickup_loc_count_accumulator(F.col("trips")))
    popular_pickups.write.option("header", "true").csv(f"output/popular_pickups/{exec_time}")

    popular_routes = calculate_routes_popularity(zones_data=zones_data)(trips_data)\
        .withColumn("trips", increment_routes_count_accumulator(F.col("trips")))
    popular_routes.write.option("header", "true").csv(f"output/popular_routes/{exec_time}")

    print(f"Zones data count: {zones_count_acc.value}")
    print(f"Trips data count: {trips_count_acc.value}")
    print(f"Pickup location data count: {pickup_loc_count_acc.value}")
    print(f"Routes data count: {routes_count_acc.value}")


def observers_with_extended_stats(spark: SparkSession):
    raw_data_observation = Observation("raw data")

    trips_data = spark.read.parquet("data/trips/*/*.parquet") \
        .observe(
            raw_data_observation,
            F.count(F.lit(1)).alias("num_rows"),
            F.sum("total_amount").alias("total_amount"),
            F.avg("total_amount").alias("avg_fare")
        )

    print(trips_data.count())
    [print(kvp) for kvp in raw_data_observation.get.items()]


if __name__ == '__main__':
    start_time = time.time()
    spark = create_spark_session()

    # observers_with_extended_stats(spark=spark)

    # run_with_logging(spark=spark)
    # run_with_cache(spark=spark)
    # run_with_checkpoint(spark=spark)
    # run_with_no_logging(spark=spark)
    # run_with_observers(spark=spark)
    run_with_accumulators(spark=spark)

    print(f"Took: {time.time() - start_time} seconds")

    time.sleep(10000)

