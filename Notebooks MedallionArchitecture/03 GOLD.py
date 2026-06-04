# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.functions import round as spark_round

print("══ NYC Taxi — GOLD ══")

nyctaxi_silver = spark.table("bigdata_final.taxis.silver")

print(f"📦 Registros en SILVER: {nyctaxi_silver.count():,}")

# COMMAND ----------

nyctaxi_by_hour = nyctaxi_silver.groupBy(
    "pickup_year", "pickup_month", "pickup_dayofweek", "pickup_hour"
).agg(
    F.count("*").alias("total_trips"),
    F.avg("trip_distance").alias("avg_distance"),
    F.avg("total_amount").alias("avg_fare"),
    F.sum("total_amount").alias("total_revenue"),
    F.avg("passenger_count").alias("avg_passengers"),
    F.percentile_approx("trip_duration_minutes", 0.5).alias("median_duration")
)

nyctaxi_by_hour.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bigdata_final.taxis.gold_by_hour")

print(f"✅ by_hour escrito: {nyctaxi_by_hour.count():,} filas")
nyctaxi_by_hour.orderBy("pickup_hour").show(5)

# COMMAND ----------

nyctaxi_by_zone = nyctaxi_silver \
    .withColumn("pickup_zone_lat", spark_round(col("pickup_latitude"), 2)) \
    .withColumn("pickup_zone_lng", spark_round(col("pickup_longitude"), 2)) \
    .groupBy("pickup_zone_lat", "pickup_zone_lng") \
    .agg(
        F.count("*").alias("pickup_count"),
        F.avg("trip_distance").alias("avg_distance"),
        F.sum("total_amount").alias("total_revenue")
    ).orderBy(col("pickup_count").desc())

nyctaxi_by_zone.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bigdata_final.taxis.gold_by_zone")

print(f"✅ by_zone escrito: {nyctaxi_by_zone.count():,} zonas")
nyctaxi_by_zone.show(5)

# COMMAND ----------

nyctaxi_by_vendor = nyctaxi_silver.groupBy("vendor_id").agg(
    F.count("*").alias("total_trips"),
    F.avg("trip_distance").alias("avg_distance"),
    F.avg("total_amount").alias("avg_fare"),
    F.avg("passenger_count").alias("avg_passengers"),
    F.sum("total_amount").alias("total_revenue"),
    F.avg("tip_amount").alias("avg_tip")
)

nyctaxi_by_vendor.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bigdata_final.taxis.gold_by_vendor")

print(f"✅ by_vendor escrito: {nyctaxi_by_vendor.count():,} vendors")
nyctaxi_by_vendor.show()

# COMMAND ----------

nyctaxi_temporal = nyctaxi_silver.groupBy(
    "pickup_month", "pickup_dayofweek", "time_period"
).agg(
    F.count("*").alias("trip_count"),
    F.avg("total_amount").alias("avg_fare"),
    F.sum("total_amount").alias("revenue")
).orderBy("pickup_month", "pickup_dayofweek")

nyctaxi_temporal.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bigdata_final.taxis.gold_temporal_patterns")

print(f"✅ temporal_patterns escrito: {nyctaxi_temporal.count():,} filas")
nyctaxi_temporal.show(10)

# COMMAND ----------

nyctaxi_payment = nyctaxi_silver.groupBy("payment_type").agg(
    F.count("*").alias("payment_count"),
    F.avg("total_amount").alias("avg_amount"),
    F.avg("tip_amount").alias("avg_tip"),
    F.sum("total_amount").alias("total_revenue")
)

nyctaxi_payment.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bigdata_final.taxis.gold_payment_analysis")

print(f"✅ payment_analysis escrito: {nyctaxi_payment.count():,} tipos de pago")
nyctaxi_payment.show()

# COMMAND ----------

nyctaxi_fact = nyctaxi_silver.select(
    "vendor_id", "pickup_datetime", "dropoff_datetime",
    "passenger_count", "trip_distance", "trip_duration_minutes",
    "fare_amount", "extra", "mta_tax", "tip_amount",
    "tolls_amount", "total_amount",
    "payment_type", "rate_code_id",
    "pickup_latitude", "pickup_longitude",
    "dropoff_latitude", "dropoff_longitude",
    "pickup_year", "pickup_month", "pickup_dayofweek", "pickup_hour",
    "time_period", "trip_type", "fare_per_mile", "fare_per_minute"
)

nyctaxi_fact.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("pickup_year", "pickup_month") \
    .saveAsTable("bigdata_final.taxis.gold_fact_table")

print(f"✅ fact_table escrita: {nyctaxi_fact.count():,} registros")
print("📂 Particionada por: pickup_year / pickup_month")

# COMMAND ----------

tablas = {
    "nyctaxi_by_hour":           "bigdata_final.taxis.gold_by_hour",
    "nyctaxi_by_zone":           "bigdata_final.taxis.gold_by_zone",
    "nyctaxi_by_vendor":         "bigdata_final.taxis.gold_by_vendor",
    "nyctaxi_temporal_patterns": "bigdata_final.taxis.gold_temporal_patterns",
    "nyctaxi_payment_analysis":  "bigdata_final.taxis.gold_payment_analysis",
    "nyctaxi_fact_table":        "bigdata_final.taxis.gold_fact_table",
}

print("══ GOLD — Resumen Final ══")
for nombre, ruta in tablas.items():
    count = spark.table(ruta).count()
    print(f"  ✅ {nombre}: {count:,} filas")

print("\n🏁 GOLD completo")