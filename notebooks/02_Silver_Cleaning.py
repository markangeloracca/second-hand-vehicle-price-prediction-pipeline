# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: Silver Layer - Data Cleaning & Validation
# MAGIC 
# MAGIC This notebook cleans and validates data from the Bronze layer.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, regexp_replace, trim, when, isnan, isnull, 
    split, element_at, length, upper, lower, 
    current_timestamp, lit, count, mean, median, 
    first, last, collect_list, size, array_contains
)
from pyspark.sql.types import *
from pyspark.sql.window import Window
import re
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Source and target tables
BRONZE_TABLE_NAME = "bronze_vehicles"
SILVER_TABLE_NAME = "silver_vehicles"

# Outlier thresholds
MIN_YEAR = 1990
MAX_YEAR = datetime.now().year + 1
MIN_PRICE = 1000
MAX_PRICE = 1000000
MIN_KILOMETRES = 0
MAX_KILOMETRES = 500000

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Bronze Data

# COMMAND ----------

df_bronze = spark.table(BRONZE_TABLE_NAME)
print(f"Bronze data loaded: {df_bronze.count()} rows")
df_bronze.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Extract Numeric Values from Kilometres

# COMMAND ----------

# Extract numeric value from "53052 km" format
df_cleaned = df_bronze.withColumn(
    "kilometres_numeric",
    regexp_replace(col("Kilometres"), r"[^0-9]", "").cast("int")
)

# Handle cases where extraction fails
df_cleaned = df_cleaned.withColumn(
    "kilometres_numeric",
    when(col("kilometres_numeric").isNull() | (col("kilometres_numeric") == 0), None)
    .otherwise(col("kilometres_numeric"))
)

print("Kilometres extraction sample:")
display(df_cleaned.select("Kilometres", "kilometres_numeric").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Extract Numeric Values from City/Highway Fuel Efficiency

# COMMAND ----------

# Function to extract numeric from fuel efficiency strings like "12.2L/100km" or "9.0L - 9.5L/100km"
def extract_fuel_efficiency(fuel_col):
    # Extract first number before "L" or range (take average of range)
    return when(
        col(fuel_col).isNotNull(),
        regexp_replace(
            regexp_replace(
                regexp_replace(col(fuel_col), r"L/100km", ""),
                r" - .*", ""  # Remove range part, keep first value
            ),
            r"[^0-9.]", ""
        ).cast("float")
    ).otherwise(None)

df_cleaned = df_cleaned \
    .withColumn("city_mpg_numeric", extract_fuel_efficiency("City")) \
    .withColumn("highway_mpg_numeric", extract_fuel_efficiency("Highway"))

print("Fuel efficiency extraction sample:")
display(df_cleaned.select("City", "city_mpg_numeric", "Highway", "highway_mpg_numeric").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Data Type Conversions

# COMMAND ----------

# Convert Year to integer
df_cleaned = df_cleaned.withColumn(
    "year_int",
    col("Year").cast("int")
)

# Price is already numeric, but ensure it's float
df_cleaned = df_cleaned.withColumn(
    "price_float",
    col("Price").cast("float")
)

print("Data type conversions sample:")
display(df_cleaned.select("Year", "year_int", "Price", "price_float").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Standardize Categorical Values

# COMMAND ----------

# Standardize Fuel Type: "Gas" -> "Gasoline"
df_cleaned = df_cleaned.withColumn(
    "fuel_type_standardized",
    when(upper(col("Fuel_Type")).contains("GASOLINE"), "Gasoline")
    .when(upper(col("Fuel_Type")).contains("GAS"), "Gasoline")
    .when(upper(col("Fuel_Type")).contains("PREMIUM"), "Premium Unleaded")
    .when(upper(col("Fuel_Type")).contains("DIESEL"), "Diesel")
    .when(upper(col("Fuel_Type")).contains("ELECTRIC"), "Electric")
    .when(upper(col("Fuel_Type")).contains("HYBRID"), "Hybrid")
    .otherwise(trim(upper(col("Fuel_Type"))))
)

# Standardize Transmission (remove extra spaces, standardize case)
df_cleaned = df_cleaned.withColumn(
    "transmission_standardized",
    trim(upper(col("Transmission")))
)

# Standardize Engine (extract key info - simplified)
df_cleaned = df_cleaned.withColumn(
    "engine_standardized",
    trim(upper(col("Engine")))
)

print("Categorical standardization sample:")
display(df_cleaned.select("Fuel_Type", "fuel_type_standardized",
                  "Transmission", "transmission_standardized").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Handle Missing Values - Kilometres

# COMMAND ----------

# Calculate median kilometres by Make/Model/Year group
window_spec = Window.partitionBy("Make", "Model", "year_int")

# Calculate median for each group
df_with_median = df_cleaned.withColumn(
    "median_km_by_group",
    median("kilometres_numeric").over(window_spec)
)

# Impute missing kilometres with group median, fallback to overall median
overall_median_km = df_cleaned.agg(median("kilometres_numeric").alias("median")).collect()[0]["median"]

df_cleaned = df_with_median.withColumn(
    "kilometres_imputed",
    when(col("kilometres_numeric").isNull(), 
         when(col("median_km_by_group").isNotNull(), col("median_km_by_group"))
         .otherwise(overall_median_km))
    .otherwise(col("kilometres_numeric"))
)

print(f"Overall median kilometres: {overall_median_km}")
print("Kilometres imputation sample:")
display(df_cleaned.select("Make", "Model", "year_int", "kilometres_numeric",
                  "kilometres_imputed").filter(col("kilometres_numeric").isNull()).orderBy(F.rand()).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Handle Missing Values - Engine/Transmission

# COMMAND ----------

# Impute Engine and Transmission with mode by Make/Model
window_make_model = Window.partitionBy("Make", "Model")

# Get mode (most frequent) for Engine
df_with_engine_mode = df_cleaned.withColumn(
    "engine_mode",
    first("engine_standardized", ignorenulls=True).over(window_make_model)
)

df_cleaned = df_with_engine_mode.withColumn(
    "engine_imputed",
    when(col("engine_standardized").isNull() | (col("engine_standardized") == ""), 
         col("engine_mode"))
    .otherwise(col("engine_standardized"))
)

# Get mode for Transmission
df_with_trans_mode = df_cleaned.withColumn(
    "transmission_mode",
    first("transmission_standardized", ignorenulls=True).over(window_make_model)
)

df_cleaned = df_with_trans_mode.withColumn(
    "transmission_imputed",
    when(col("transmission_standardized").isNull() | (col("transmission_standardized") == ""), 
         col("transmission_mode"))
    .otherwise(col("transmission_standardized"))
)

print("Engine/Transmission imputation sample:")
display(df_cleaned.select("Make", "Model", "Engine", "engine_imputed", "Transmission", "transmission_imputed")
        .orderBy(F.rand()).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Handle Missing Values - City/Highway MPG

# COMMAND ----------

# Calculate median by Make/Model for City and Highway
window_make_model_mpg = Window.partitionBy("Make", "Model")

df_with_mpg_median = df_cleaned.withColumn(
    "median_city_by_group",
    median("city_mpg_numeric").over(window_make_model_mpg)
).withColumn(
    "median_highway_by_group",
    median("highway_mpg_numeric").over(window_make_model_mpg)
)

# Overall medians
overall_median_city = df_cleaned.agg(median("city_mpg_numeric").alias("median")).collect()[0]["median"]
overall_median_highway = df_cleaned.agg(median("highway_mpg_numeric").alias("median")).collect()[0]["median"]

# Impute: if both missing, keep as null (will drop later), else impute with group median or overall median
df_cleaned = df_with_mpg_median.withColumn(
    "city_mpg_imputed",
    when(col("city_mpg_numeric").isNull(),
         when(col("median_city_by_group").isNotNull(), col("median_city_by_group"))
         .otherwise(overall_median_city))
    .otherwise(col("city_mpg_numeric"))
).withColumn(
    "highway_mpg_imputed",
    when(col("highway_mpg_numeric").isNull(),
         when(col("median_highway_by_group").isNotNull(), col("median_highway_by_group"))
         .otherwise(overall_median_highway))
    .otherwise(col("highway_mpg_numeric"))
)

print(f"Overall median City MPG: {overall_median_city}")
print(f"Overall median Highway MPG: {overall_median_highway}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Handle Missing Values - Passengers/Doors

# MAGIC Impute based on Body Type defaults

# COMMAND ----------

# Define defaults by Body Type
body_type_defaults = {
    "SUV": {"passengers": 5.0, "doors": 4.0},
    "Sedan": {"passengers": 5.0, "doors": 4.0},
    "Truck": {"passengers": 2.0, "doors": 2.0},
    "Coupe": {"passengers": 4.0, "doors": 2.0},
    "Wagon": {"passengers": 5.0, "doors": 4.0},
    "Hatchback": {"passengers": 5.0, "doors": 4.0}
}

# Impute Passengers
df_cleaned = df_cleaned.withColumn(
    "passengers_imputed",
    when(col("Passengers").isNull() | isnan(col("Passengers")),
         when(upper(col("Body_Type")).contains("SUV"), 5.0)
         .when(upper(col("Body_Type")).contains("TRUCK"), 2.0)
         .when(upper(col("Body_Type")).contains("COUPE"), 4.0)
         .otherwise(5.0))
    .otherwise(col("Passengers").cast("float"))
)

# Impute Doors
df_cleaned = df_cleaned.withColumn(
    "doors_imputed",
    when(col("Doors").isNull() | (col("Doors") == ""),
         when(upper(col("Body_Type")).contains("COUPE"), 2.0)
         .when(upper(col("Body_Type")).contains("TRUCK"), 2.0)
         .otherwise(4.0))
    .otherwise(
        when(col("Doors").rlike(r"\d+"), 
             regexp_replace(col("Doors"), r"[^0-9]", "").cast("float"))
        .otherwise(4.0)
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Outlier Removal

# COMMAND ----------

# Filter outliers
initial_count = df_cleaned.count()

df_cleaned = df_cleaned.filter(
    (col("year_int") >= MIN_YEAR) & 
    (col("year_int") <= MAX_YEAR) &
    (col("price_float") >= MIN_PRICE) & 
    (col("price_float") <= MAX_PRICE) &
    (col("kilometres_imputed") >= MIN_KILOMETRES) & 
    (col("kilometres_imputed") <= MAX_KILOMETRES)
)

final_count = df_cleaned.count()
removed_count = initial_count - final_count

print(f"Initial rows: {initial_count}")
print(f"Rows after outlier removal: {final_count}")
print(f"Rows removed: {removed_count} ({round(removed_count/initial_count*100, 2)}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Remove Duplicates

# COMMAND ----------

# Remove exact duplicates based on key fields
before_dedup = df_cleaned.count()

# Key fields for duplicate detection
key_fields = ["Make", "Model", "year_int", "kilometres_imputed", "price_float", 
              "Body_Type", "engine_imputed", "transmission_imputed"]

df_cleaned = df_cleaned.dropDuplicates(key_fields)

after_dedup = df_cleaned.count()
dup_count = before_dedup - after_dedup

print(f"Rows before deduplication: {before_dedup}")
print(f"Rows after deduplication: {after_dedup}")
print(f"Duplicates removed: {dup_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Create Silver Table Schema

# COMMAND ----------

# Select and rename columns for Silver layer
df_silver = df_cleaned.select(
    col("record_id"),
    col("ingestion_timestamp"),
    col("source_file"),
    col("Make"),
    col("Model"),
    col("year_int").alias("Year"),
    col("kilometres_imputed").alias("Kilometres"),
    col("Body_Type").alias("BodyType"),
    col("engine_imputed").alias("Engine"),
    col("transmission_imputed").alias("Transmission"),
    col("Drivetrain"),
    col("Exterior_Colour").alias("ExteriorColour"),
    col("Interior_Colour").alias("InteriorColour"),
    col("passengers_imputed").alias("Passengers"),
    col("doors_imputed").alias("Doors"),
    col("fuel_type_standardized").alias("FuelType"),
    col("city_mpg_imputed").alias("City"),
    col("highway_mpg_imputed").alias("Highway"),
    col("price_float").alias("Price"),
    current_timestamp().alias("silver_processing_timestamp")
)

# Drop rows where both City and Highway are still null (couldn't be imputed)
df_silver = df_silver.filter(
    col("City").isNotNull() | col("Highway").isNotNull()
)

print("Silver schema:")
df_silver.printSchema()
print(f"\nSilver row count: {df_silver.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12: Write to Delta Lake

# COMMAND ----------

# Write to Delta table (managed table in Unity Catalog)
df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(SILVER_TABLE_NAME)

print(f"Silver table '{SILVER_TABLE_NAME}' created successfully!")
print(f"Table is managed by Unity Catalog")

# Verify table creation
print("\nVerification - Sample from Silver table:")
spark.table(SILVER_TABLE_NAME).show(5, truncate=False)

print(f"\nFinal row count: {spark.table(SILVER_TABLE_NAME).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Summary

# COMMAND ----------

print("=" * 80)
print("SILVER LAYER CLEANING COMPLETE")
print("=" * 80)
print(f"Source: {BRONZE_TABLE_NAME}")
print(f"Target: {SILVER_TABLE_NAME}")
print(f"Initial rows: {initial_count}")
print(f"Final rows: {final_count}")
print(f"Rows removed (outliers): {removed_count}")
print(f"Duplicates removed: {dup_count}")
print(f"Processing timestamp: {datetime.now()}")
print("=" * 80)

# Show data quality metrics
print("\nData Quality Metrics:")
quality_metrics = df_silver.agg(
    count("*").alias("total_rows"),
    count(when(col("Kilometres").isNull(), 1)).alias("null_kilometres"),
    count(when(col("Engine").isNull() | (col("Engine") == ""), 1)).alias("null_engine"),
    count(when(col("Transmission").isNull() | (col("Transmission") == ""), 1)).alias("null_transmission"),
    count(when(col("City").isNull(), 1)).alias("null_city"),
    count(when(col("Highway").isNull(), 1)).alias("null_highway")
)
quality_metrics.show()

