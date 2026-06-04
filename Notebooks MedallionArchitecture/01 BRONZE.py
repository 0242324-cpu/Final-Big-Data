# Databricks notebook source
from functools import reduce
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import *

CATALOG = "bigdata_final"
LIMIT   = 4_000_000
spark.conf.set("spark.sql.shuffle.partitions", "8")

# ── Core utility ──────────────────────────────────────────────────
def read_to_limit(paths, read_fn, limit=LIMIT):
    collected = []
    total     = 0
    for path in paths:
        try:
            df    = read_fn(path)
            count = df.count()
            total += count
            print(f"  {path.split('/')[-1]:<35} {count:>10,}  |  total: {total:>12,}")
            collected.append(df)
            if total >= limit:
                print(f"\n  ✅ {total:,} rows across {len(collected)} files — stopping")
                break
        except Exception as e:
            print(f"  ⚠️  Skipped ({e})")
    return reduce(DataFrame.union, collected)

def write_bronze(df, schema_name):
    full_table = f"{CATALOG}.{schema_name}.bronze"
    (
        df
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source",      F.lit(schema_name))
        .coalesce(8)
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(full_table)
    )
    print(f"  💾 {full_table}\n")

# COMMAND ----------

# ── NYC Taxi (Delta — no file-by-file needed) ─────────────────────
print("══ NYC Taxi ══")


nyctaxi_df = (
    spark.read
    .format("delta")
    .load("dbfs:/databricks-datasets/nyctaxi/tables/nyctaxi_yellow")
    .limit(5_000_000)
)

count = nyctaxi_df.count()
print(f"  Rows: {count:,}")
write_bronze(nyctaxi_df, "taxis")

# COMMAND ----------

nyctaxi_df.count()