# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1: Bronze Layer - Data Ingestion
# MAGIC
# MAGIC This notebook ingests raw CSV data into Delta Lake format with metadata tracking.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

import os
from datetime import datetime

from pyspark.sql.functions import (
    col,
    current_timestamp,
    isnan,
    lit,
    monotonically_increasing_id,
)
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Source data path
RAW_DATA_PATH = "/Volumes/workspace/default/ensf612/ensf612project-data.csv"

# Target Delta table
BRONZE_TABLE_NAME = "bronze_vehicles"

# Source file metadata
SOURCE_FILE = os.path.basename(RAW_DATA_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read CSV Data

# COMMAND ----------

# Read CSV with header and infer schema
df_raw = (
    spark.read.option("header", "true").option("inferSchema", "true").csv(RAW_DATA_PATH)
)

# Display initial schema and sample data
print("Initial Schema:")
df_raw.printSchema()

print("\nSample Data (first 5 rows):")
df_raw.show(5, truncate=False)

print(f"\nTotal rows: {df_raw.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Column Names

# COMMAND ----------

# Rename columns to remove spaces and special characters (Delta Lake requirement)
# Replace spaces with underscores and trim leading/trailing spaces
column_mapping = {}
for old_col in df_raw.columns:
    # Remove leading/trailing spaces and replace spaces with underscores
    new_col = old_col.strip().replace(" ", "_")
    if old_col != new_col:
        column_mapping[old_col] = new_col

# Apply column renaming
df_cleaned = df_raw
for old_col, new_col in column_mapping.items():
    df_cleaned = df_cleaned.withColumnRenamed(old_col, new_col)

if column_mapping:
    print("Column names cleaned:")
    for old_col, new_col in column_mapping.items():
        print(f"  '{old_col}' -> '{new_col}'")
else:
    print("No column name changes needed")

print("\nCleaned Schema:")
df_cleaned.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Metadata Columns

# COMMAND ----------

# Add metadata columns
df_bronze = (
    df_cleaned.withColumn("ingestion_timestamp", current_timestamp())
    .withColumn("source_file", lit(SOURCE_FILE))
    .withColumn("record_id", monotonically_increasing_id())
)

# Reorder columns to have metadata first
metadata_cols = ["record_id", "ingestion_timestamp", "source_file"]
data_cols = [c for c in df_bronze.columns if c not in metadata_cols]
df_bronze = df_bronze.select(metadata_cols + data_cols)

# Display schema
print("Bronze Schema with Metadata:")
df_bronze.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Profiling

# COMMAND ----------

# Row count validation
total_rows = df_bronze.count()
print(f"Total rows ingested: {total_rows}")
print("Expected rows: ~24,199")

# Column schema inspection
print("\nColumn Schema:")
for field in df_bronze.schema.fields:
    print(f"  {field.name}: {field.dataType}")

# Missing value counts per column
print("\nMissing Value Counts:")
missing_counts = []
for col_name in df_bronze.columns:
    if col_name not in ["ingestion_timestamp", "source_file", "record_id"]:
        # Check if column is numeric before using isnan
        col_type = dict(df_bronze.dtypes)[col_name]
        if col_type in ["int", "bigint", "float", "double", "decimal"]:
            null_count = df_bronze.filter(
                col(col_name).isNull() | isnan(col(col_name))
            ).count()
        else:
            null_count = df_bronze.filter(col(col_name).isNull()).count()
        missing_counts.append(
            (col_name, null_count, total_rows, round(null_count / total_rows * 100, 2))
        )

missing_df = spark.createDataFrame(
    missing_counts, ["Column", "Null_Count", "Total_Rows", "Null_Percentage"]
)
missing_df.orderBy(col("Null_Count").desc()).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Delta Lake

# COMMAND ----------

# Write to Delta table (managed table in Unity Catalog)
df_bronze.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(BRONZE_TABLE_NAME)

print(f"Bronze table '{BRONZE_TABLE_NAME}' created successfully!")
print("Table is managed by Unity Catalog")

# Verify table creation
print("\nVerification - Sample from Delta table:")
spark.table(BRONZE_TABLE_NAME).show(5, truncate=False)

print(f"\nFinal row count: {spark.table(BRONZE_TABLE_NAME).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 80)
print("BRONZE LAYER INGESTION COMPLETE")
print("=" * 80)
print(f"Source: {RAW_DATA_PATH}")
print(f"Target: {BRONZE_TABLE_NAME}")
print(f"Total Records: {total_rows}")
print(f"Ingestion Timestamp: {datetime.now()}")
print("=" * 80)
