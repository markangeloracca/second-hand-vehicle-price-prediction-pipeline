# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: Gold Layer - Feature Engineering
# MAGIC
# MAGIC This notebook creates model-ready features from cleaned Silver data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from datetime import datetime
from pyspark.sql import functions as F

from pyspark.sql.functions import (
    col,
    count,
    current_timestamp,
    lit,
    log,
    median,
    regexp_extract,
    trim,
    when,
)
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Source and target tables
SILVER_TABLE_NAME = "silver_vehicles"
GOLD_TABLE_NAME = "gold_vehicles"

# Train/test split
TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42

# Current year for age calculation
CURRENT_YEAR = datetime.now().year

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Silver Data

# COMMAND ----------

df_silver = spark.table(SILVER_TABLE_NAME)
print(f"Silver data loaded: {df_silver.count()} rows")
df_silver.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Derived Features

# COMMAND ----------

# Vehicle age
df_gold = df_silver.withColumn("vehicle_age", lit(CURRENT_YEAR) - col("Year"))

# Average fuel efficiency
df_gold = df_gold.withColumn(
    "avg_fuel_efficiency", (col("City") + col("Highway")) / 2.0
)

# Price per kilometre (log-transform if needed to handle skew)
df_gold = df_gold.withColumn(
    "price_per_km",
    when(col("Kilometres") > 0, col("Price") / col("Kilometres")).otherwise(None),
)

# Extract engine displacement from Engine column (if available)
# Pattern: look for numbers followed by "L" (e.g., "2.0L", "3.5L")
# Handle empty strings gracefully
df_gold = df_gold.withColumn(
    "engine_displacement",
    when(
        (regexp_extract(col("Engine"), r"(\d+\.?\d*)\s*L", 1) != "")
        & (regexp_extract(col("Engine"), r"(\d+\.?\d*)\s*L", 1).isNotNull()),
        regexp_extract(col("Engine"), r"(\d+\.?\d*)\s*L", 1).cast("float"),
    ).otherwise(None),
)

# Extract cylinder count from Engine column
# Handle empty strings gracefully
df_gold = df_gold.withColumn(
    "cylinder_count",
    when(
        col("Engine").rlike(r"(\d+)\s*cyl")
        & (regexp_extract(col("Engine"), r"(\d+)\s*cyl", 1) != ""),
        regexp_extract(col("Engine"), r"(\d+)\s*cyl", 1).cast("int"),
    )
    .when(
        col("Engine").rlike(r"V(\d+)")
        & (regexp_extract(col("Engine"), r"V(\d+)", 1) != ""),
        regexp_extract(col("Engine"), r"V(\d+)", 1).cast("int"),
    )
    .when(
        col("Engine").rlike(r"I-(\d+)")
        & (regexp_extract(col("Engine"), r"I-(\d+)", 1) != ""),
        regexp_extract(col("Engine"), r"I-(\d+)", 1).cast("int"),
    )
    .when(
        col("Engine").rlike(r"(\d+)\s*Cylinder")
        & (regexp_extract(col("Engine"), r"(\d+)\s*Cylinder", 1) != ""),
        regexp_extract(col("Engine"), r"(\d+)\s*Cylinder", 1).cast("int"),
    )
    .otherwise(None),
)

print("Derived features sample:")
display(df_gold.select(
    "Year",
    "vehicle_age",
    "City",
    "Highway",
    "avg_fuel_efficiency",
    "Price",
    "Kilometres",
    "price_per_km",
    "Engine",
    "engine_displacement",
    "cylinder_count",
).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Feature Selection

# COMMAND ----------

# Select features to keep (drop Exterior Colour, Interior Colour, Passengers, Doors, Engine)
df_gold = df_gold.select(
    col("record_id"),
    col("ingestion_timestamp"),
    col("source_file"),
    col("Make"),
    col("Model"),
    col("Year"),
    col("vehicle_age"),
    col("Kilometres"),
    col("BodyType"),
    col("engine_displacement"),
    col("cylinder_count"),
    col("Transmission"),
    col("Drivetrain"),
    col("FuelType"),
    col("City"),
    col("Highway"),
    col("avg_fuel_efficiency"),
    col("Price"),
    current_timestamp().alias("gold_processing_timestamp"),
)

print("Selected features:")
df_gold.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Handle Missing Values in Derived Features

# COMMAND ----------

# Impute missing values in derived features
# For engine_displacement and cylinder_count, use median by Make/Model

window_make_model = Window.partitionBy("Make", "Model")

# Impute engine_displacement
df_gold = df_gold.withColumn(
    "median_displacement", median("engine_displacement").over(window_make_model)
)
overall_median_displacement = df_gold.agg(
    median("engine_displacement").alias("median")
).collect()[0]["median"]

df_gold = df_gold.withColumn(
    "engine_displacement",
    when(
        col("engine_displacement").isNull(),
        when(
            col("median_displacement").isNotNull(), col("median_displacement")
        ).otherwise(overall_median_displacement),
    ).otherwise(col("engine_displacement")),
)

# Impute cylinder_count
df_gold = df_gold.withColumn(
    "median_cylinders", median("cylinder_count").over(window_make_model)
)
overall_median_cylinders = df_gold.agg(
    median("cylinder_count").alias("median")
).collect()[0]["median"]

df_gold = df_gold.withColumn(
    "cylinder_count",
    when(
        col("cylinder_count").isNull(),
        when(col("median_cylinders").isNotNull(), col("median_cylinders")).otherwise(
            overall_median_cylinders
        ),
    ).otherwise(col("cylinder_count")),
)

# Drop temporary columns
df_gold = df_gold.drop("median_displacement", "median_cylinders")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Train/Test Split

# COMMAND ----------

# Perform 80/20 split with random seed (BEFORE frequency encoding to avoid data leakage)
train_df, test_df = df_gold.randomSplit(
    [TRAIN_TEST_SPLIT, 1 - TRAIN_TEST_SPLIT], seed=RANDOM_SEED
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Frequency Encoding for Model

# COMMAND ----------

# Calculate frequency encoding for Model (high cardinality categorical)
# IMPORTANT: Calculate frequency on TRAINING data only to avoid data leakage
# This converts Model into a numerical feature based on how common each model is

# Calculate frequency of each Model in TRAINING data only
model_freq = train_df.groupBy("Model").agg(count("*").alias("model_frequency"))

# Join frequency back to both train and test dataframes
train_df = train_df.join(model_freq, "Model", "left")
test_df = test_df.join(model_freq, "Model", "left")

# Handle models in test set that weren't in training set (set frequency to 1)
train_df = train_df.withColumn(
    "model_frequency",
    when(col("model_frequency").isNull(), 1).otherwise(col("model_frequency")),
)
test_df = test_df.withColumn(
    "model_frequency",
    when(col("model_frequency").isNull(), 1).otherwise(col("model_frequency")),
)

# Log transform frequency to reduce impact of very common models
train_df = train_df.withColumn(
    "model_frequency_log",
    log(col("model_frequency") + 1),  # +1 to avoid log(0)
)
test_df = test_df.withColumn("model_frequency_log", log(col("model_frequency") + 1))

print("Model frequency encoding sample (training data):")
display(
    train_df.select("Make", "Model", "model_frequency", "model_frequency_log").limit(10)
)

# Calculate counts
train_count = train_df.count()
test_count = test_df.count()
total_count = train_count + test_count

print(f"\nTraining set size: {train_count} rows")
print(f"Test set size: {test_count} rows")
print(f"Train percentage: {round(train_count / total_count * 100, 2)}%")
print(f"Test percentage: {round(test_count / total_count * 100, 2)}%")

# Add split indicator
train_df = train_df.withColumn("split", lit("train"))
test_df = test_df.withColumn("split", lit("test"))

# Combine for storage (after frequency encoding)
df_gold_with_split = train_df.union(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Categorical Encoding Setup

# MAGIC Note: We'll prepare the data for encoding, but actual encoding will be done in the model training pipeline

# COMMAND ----------

# Identify categorical columns (Model removed - using frequency encoding instead)
categorical_cols = ["Make", "BodyType", "FuelType", "Transmission", "Drivetrain"]

# Identify numerical columns for scaling (includes model_frequency_log)
numerical_cols = [
    "Kilometres",
    "City",
    "Highway",
    "vehicle_age",
    "avg_fuel_efficiency",
    "engine_displacement",
    "cylinder_count",
    "model_frequency_log",
]

# Target variable
target_col = "Price"

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)
print("Target column:", target_col)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Prepare Data for ML Pipeline

# COMMAND ----------

# Ensure all categorical columns are strings and handle nulls
for cat_col in categorical_cols:
    df_gold_with_split = df_gold_with_split.withColumn(
        cat_col,
        when(col(cat_col).isNull() | (col(cat_col) == ""), "Unknown").otherwise(
            col(cat_col).cast("string")
        ),
    )

# Ensure all numerical columns are numeric and handle nulls/empty strings
# Check for empty strings before casting
for num_col in numerical_cols:
    df_gold_with_split = df_gold_with_split.withColumn(
        num_col,
        when(
            (col(num_col).isNotNull()) & (trim(col(num_col).cast("string")) != ""),
            col(num_col).cast("double"),
        ).otherwise(None),
    )

# Ensure target is numeric (handle empty strings safely)
df_gold_with_split = df_gold_with_split.withColumn(
    target_col,
    when(
        (col(target_col).isNotNull()) & (trim(col(target_col).cast("string")) != ""),
        col(target_col).cast("double"),
    ).otherwise(None),
)

# Filter out any rows with null target
df_gold_with_split = df_gold_with_split.filter(col(target_col).isNotNull())

print("Data preparation complete")
print(f"Final row count: {df_gold_with_split.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Write to Delta Lake

# COMMAND ----------

# Write to Delta table (managed table in Unity Catalog)
df_gold_with_split.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(GOLD_TABLE_NAME)

print(f"Gold table '{GOLD_TABLE_NAME}' created successfully!")
print("Table is managed by Unity Catalog")

# Verify table creation
print("\nVerification - Sample from Gold table:")
display(spark.table(GOLD_TABLE_NAME).orderBy(F.rand()).limit(5))

print(f"\nFinal row count: {spark.table(GOLD_TABLE_NAME).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Create Separate Train/Test Tables (Optional but useful)

# COMMAND ----------

# Create separate tables for train and test sets
train_table_name = f"{GOLD_TABLE_NAME}_train"
test_table_name = f"{GOLD_TABLE_NAME}_test"

# Write train set (managed table in Unity Catalog)
train_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(train_table_name)

# Write test set (managed table in Unity Catalog)
test_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(test_table_name)

print(
    f"Train table '{train_table_name}' created: {spark.table(train_table_name).count()} rows"
)
print(
    f"Test table '{test_table_name}' created: {spark.table(test_table_name).count()} rows"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 80)
print("GOLD LAYER FEATURE ENGINEERING COMPLETE")
print("=" * 80)
print(f"Source: {SILVER_TABLE_NAME}")
print(f"Target: {GOLD_TABLE_NAME}")
total_final = df_gold_with_split.count()
train_final = train_df.count()
test_final = test_df.count()
print(f"Total rows: {total_final}")
print(f"Training rows: {train_final}")
print(f"Test rows: {test_final}")
print("Features created:")
print("  - vehicle_age")
print("  - avg_fuel_efficiency")
print("  - price_per_km")
print("  - engine_displacement")
print("  - cylinder_count")
print("  - model_frequency_log (frequency encoding for Model)")
print(
    f"Categorical features: {len(categorical_cols)} (Model removed - using frequency encoding)"
)
print(f"Numerical features: {len(numerical_cols)}")
print(f"Processing timestamp: {datetime.now()}")
print("=" * 80)
