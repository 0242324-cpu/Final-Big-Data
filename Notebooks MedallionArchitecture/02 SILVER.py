# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, unix_timestamp, round, when,
    year, month, dayofweek, hour
)

print("✅ Imports OK")

# COMMAND ----------

print("══ NYC Taxi — SILVER ══")

nyctaxi_bronze = spark.table("bigdata_final.taxis.bronze")

print(f"📦 Registros en BRONZE: {nyctaxi_bronze.count():,}")
nyctaxi_bronze.printSchema()

# COMMAND ----------

nyctaxi_clean = nyctaxi_bronze.filter(
    col("pickup_datetime").isNotNull() &
    col("dropoff_datetime").isNotNull() &
    col("trip_distance").isNotNull() &
    col("pickup_latitude").isNotNull() &
    col("pickup_longitude").isNotNull() &
    col("total_amount").isNotNull()
)

print(f"🧹 Después de eliminar nulos: {nyctaxi_clean.count():,}")

# COMMAND ----------

nyctaxi_clean = nyctaxi_clean.filter(
    (col("trip_distance") > 0) &
    (col("passenger_count") > 0) &
    (col("passenger_count") <= 8) &
    (col("total_amount") >= 0) &
    (col("pickup_latitude").between(-90, 90)) &
    (col("pickup_longitude").between(-180, 180)) &
    (col("dropoff_latitude").between(-90, 90)) &
    (col("dropoff_longitude").between(-180, 180))
)

print(f"✅ Después de filtrar inválidos: {nyctaxi_clean.count():,}")

# COMMAND ----------

# Duración del viaje
nyctaxi_silver = nyctaxi_clean.withColumn(
    "trip_duration_minutes",
    (unix_timestamp(col("dropoff_datetime")) -
     unix_timestamp(col("pickup_datetime"))) / 60
).filter(col("trip_duration_minutes") > 0)

# Componentes de fecha/hora
nyctaxi_silver = nyctaxi_silver \
    .withColumn("pickup_year",      year(col("pickup_datetime"))) \
    .withColumn("pickup_month",     month(col("pickup_datetime"))) \
    .withColumn("pickup_dayofweek", dayofweek(col("pickup_datetime"))) \
    .withColumn("pickup_hour",      hour(col("pickup_datetime")))

# Tarifa por milla y por minuto
nyctaxi_silver = nyctaxi_silver \
    .withColumn("fare_per_mile",
        when(col("trip_distance") > 0,
             round(col("fare_amount") / col("trip_distance"), 2)
        ).otherwise(0)
    ) \
    .withColumn("fare_per_minute",
        when(col("trip_duration_minutes") > 0,
             round(col("fare_amount") / col("trip_duration_minutes"), 2)
        ).otherwise(0)
    )

# Categoría de viaje por distancia
nyctaxi_silver = nyctaxi_silver.withColumn(
    "trip_type",
    when(col("trip_distance") < 1,              "Local")
    .when(col("trip_distance").between(1, 5),   "Medium")
    .when(col("trip_distance").between(5, 15),  "Long")
    .otherwise("Very_Long")
)

# Categoría de hora del día
nyctaxi_silver = nyctaxi_silver.withColumn(
    "time_period",
    when(col("pickup_hour").between(6, 9),   "Morning_Rush")
    .when(col("pickup_hour").between(17, 20), "Evening_Rush")
    .when(col("pickup_hour").between(0, 5),   "Night")
    .otherwise("Off_Peak")
)

print("✅ Features calculados")
nyctaxi_silver.select(
    "trip_duration_minutes", "pickup_hour", "fare_per_mile",
    "fare_per_minute", "trip_type", "time_period"
).show(5)

# COMMAND ----------

nyctaxi_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable("bigdata_final.taxis.silver")

print(f"✅ SILVER escrito: {nyctaxi_silver.count():,} registros")
print("📍 Tabla: bigdata_final.taxis.silver")

# COMMAND ----------

silver_check = spark.table("bigdata_final.taxis.silver")

print(f"📊 Registros en SILVER: {silver_check.count():,}")
print(f"📋 Columnas: {len(silver_check.columns)}")

# Vista rápida de distribución de features nuevos
silver_check.groupBy("trip_type").count().orderBy("trip_type").show()
silver_check.groupBy("time_period").count().orderBy("time_period").show()