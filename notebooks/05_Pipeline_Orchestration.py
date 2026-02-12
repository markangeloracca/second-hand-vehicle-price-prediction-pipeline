# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 5: Pipeline Orchestration
# MAGIC
# MAGIC This notebook creates a complete sklearn pipeline and provides batch prediction functionality.
# MAGIC
# MAGIC Note: Using sklearn instead of PySpark ML for compatibility with Databricks Community Edition.
# MAGIC Model is kept in memory (not saved to disk) due to DBFS write restrictions on free tier.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from datetime import datetime

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Data tables
TRAIN_TABLE_NAME = "gold_vehicles_train"
TEST_TABLE_NAME = "gold_vehicles_test"

# Feature columns
CATEGORICAL_COLS = ["Make", "BodyType", "FuelType", "Transmission", "Drivetrain"]
NUMERICAL_COLS = [
    "Kilometres",
    "City",
    "Highway",
    "vehicle_age",
    "avg_fuel_efficiency",
    "engine_displacement",
    "cylinder_count",
    "model_frequency_log",
]
TARGET_COL = "Price"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Training Data

# COMMAND ----------

# Load train data from Delta table
train_spark_df = spark.table(TRAIN_TABLE_NAME)
print(f"Training data loaded: {train_spark_df.count()} rows")

# Convert to pandas
feature_cols = CATEGORICAL_COLS + NUMERICAL_COLS + [TARGET_COL]
train_df = train_spark_df.select(feature_cols).toPandas()

# Separate features and target
X_train = train_df[CATEGORICAL_COLS + NUMERICAL_COLS]
y_train = train_df[TARGET_COL]

print(f"Training features shape: {X_train.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Complete Pipeline

# COMMAND ----------

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CATEGORICAL_COLS,
        ),
        ("num", StandardScaler(), NUMERICAL_COLS),
    ]
)

# Create complete pipeline with Random Forest (best performing model)
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print("Pipeline created with stages:")
print("  1. ColumnTransformer (OneHotEncoder + StandardScaler)")
print("  2. RandomForestRegressor")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Pipeline

# COMMAND ----------

print("Training complete pipeline...")
start_time = datetime.now()

pipeline.fit(X_train, y_train)

training_time = (datetime.now() - start_time).total_seconds()
print(f"Pipeline training complete in {training_time:.2f} seconds")

# Note: Model is kept in memory. DBFS write is not available on Community Edition.
print("Model stored in memory (trained_pipeline variable)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Pipeline on Sample Data

# COMMAND ----------

# Test on a small sample from training data
test_sample = X_train.head(5)
sample_predictions = pipeline.predict(test_sample)

print("Sample predictions:")
sample_results = test_sample[["Make"]].copy()
sample_results["Actual_Price"] = y_train.head(5).values
sample_results["Predicted_Price"] = sample_predictions
print(sample_results.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Prediction Function

# COMMAND ----------


def batch_predict(input_table_name, output_table_name, model=None):
    """
    Perform batch predictions on new data.

    Parameters:
    - input_table_name: Name of the input Delta table
    - output_table_name: Name of the output Delta table for predictions
    - model: Trained sklearn pipeline (uses global 'pipeline' if not provided)
    """
    # Use provided model or global pipeline
    if model is None:
        model = pipeline

    print(f"Loading input data from {input_table_name}...")
    input_spark_df = spark.table(input_table_name)
    print(f"Input rows: {input_spark_df.count()}")

    # Convert to pandas
    input_df = input_spark_df.toPandas()

    # Prepare features - MUST be in exact same order as during training
    feature_cols = CATEGORICAL_COLS + NUMERICAL_COLS
    X = input_df[feature_cols].copy()

    print("Generating predictions...")
    predictions = model.predict(X)

    # Create output dataframe
    output_df = input_df.copy()
    output_df["prediction"] = predictions
    output_df["prediction_timestamp"] = datetime.now()
    output_df["model_version"] = "v1.0_sklearn"

    # Select output columns
    output_cols = [
        "record_id",
        "Make",
        "Model",
        "Year",
        "Kilometres",
        "Price",
        "prediction",
        "prediction_timestamp",
        "model_version",
    ]
    available_output_cols = [c for c in output_cols if c in output_df.columns]
    output_df = output_df[available_output_cols]

    # Convert back to Spark and save to Delta
    output_spark_df = spark.createDataFrame(output_df)
    output_spark_df.write.format("delta").mode("overwrite").saveAsTable(
        output_table_name
    )

    print(f"Predictions saved to '{output_table_name}' table")
    print(f"Total predictions: {len(output_df)}")

    return output_df


# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Batch Prediction on Test Set

# COMMAND ----------

PREDICTIONS_TABLE_NAME = "vehicle_price_predictions"

print("Running batch prediction on test set...")
predictions_result = batch_predict(
    TEST_TABLE_NAME, PREDICTIONS_TABLE_NAME, model=pipeline
)

print("\nSample predictions:")
print(predictions_result.head(10).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction Statistics

# COMMAND ----------

# Load predictions and calculate statistics
predictions_df = spark.table(PREDICTIONS_TABLE_NAME).toPandas()

# Calculate metrics
actual = predictions_df["Price"]
predicted = predictions_df["prediction"]

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("=" * 80)
print("PREDICTION STATISTICS")
print("=" * 80)
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"\nActual Price Range: ${actual.min():,.2f} - ${actual.max():,.2f}")
print(f"Predicted Price Range: ${predicted.min():,.2f} - ${predicted.max():,.2f}")
print(f"Average Actual Price: ${actual.mean():,.2f}")
print(f"Average Predicted Price: ${predicted.mean():,.2f}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Metadata

# COMMAND ----------

# Pipeline metadata
pipeline_metadata = {
    "model_storage": "In-memory (Databricks Community Edition limitation)",
    "created_timestamp": datetime.now().isoformat(),
    "categorical_features": CATEGORICAL_COLS,
    "numerical_features": NUMERICAL_COLS,
    "target_variable": TARGET_COL,
    "model_type": "RandomForestRegressor (sklearn)",
    "model_params": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 5,
    },
    "training_time_seconds": training_time,
}

print("Pipeline Metadata:")
for key, value in pipeline_metadata.items():
    print(f"  {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 80)
print("PIPELINE ORCHESTRATION COMPLETE")
print("=" * 80)
print("Pipeline model: Stored in memory (variable: pipeline)")
print(f"Test predictions table: {PREDICTIONS_TABLE_NAME}")
print("Batch prediction function: batch_predict()")
print("Model type: sklearn RandomForestRegressor")
print(f"Test set RMSE: ${rmse:,.2f}")
print(f"Test set R2: {r2:.4f}")
print(f"Completion timestamp: {datetime.now()}")
print("=" * 80)
