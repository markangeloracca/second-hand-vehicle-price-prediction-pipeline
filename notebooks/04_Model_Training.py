# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4: Model Training & Evaluation
# MAGIC
# MAGIC This notebook trains three models (Linear Regression, Random Forest, Gradient Boosting) using scikit-learn.
# MAGIC
# MAGIC Note: Using scikit-learn instead of PySpark ML for compatibility with Databricks Community Edition.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
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

# Cross-validation folds
CV_FOLDS = 3

# Store results for comparison
model_results = []

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load train and test data from Delta tables
train_spark_df = spark.table(TRAIN_TABLE_NAME)
test_spark_df = spark.table(TEST_TABLE_NAME)

print(f"Training set: {train_spark_df.count()} rows")
print(f"Test set: {test_spark_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert to Pandas

# COMMAND ----------

# Select only the columns we need and convert to pandas
feature_cols = CATEGORICAL_COLS + NUMERICAL_COLS + [TARGET_COL]

train_df = train_spark_df.select(feature_cols).toPandas()
test_df = test_spark_df.select(feature_cols).toPandas()

# Separate features and target
X_train = train_df[CATEGORICAL_COLS + NUMERICAL_COLS]
y_train = train_df[TARGET_COL]
X_test = test_df[CATEGORICAL_COLS + NUMERICAL_COLS]
y_test = test_df[TARGET_COL]

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Preprocessing Pipeline

# COMMAND ----------

# Create preprocessing pipeline
# - OneHotEncoder for categorical columns
# - StandardScaler for numerical columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
            CATEGORICAL_COLS,
        ),
        ("num", StandardScaler(), NUMERICAL_COLS),
    ]
)

print("Preprocessor created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 1: Linear Regression (Baseline)

# COMMAND ----------

start_time = time.time()

# Create pipeline with preprocessing and model
lr_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", LinearRegression())]
)

# Train model
print("Training Linear Regression model...")
lr_pipeline.fit(X_train, y_train)

lr_training_time = time.time() - start_time

# Make predictions
lr_predictions = lr_pipeline.predict(X_test)

# Evaluate
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

# Store results
model_results.append(
    {
        "Model": "LinearRegression",
        "RMSE": lr_rmse,
        "MAE": lr_mae,
        "R2": lr_r2,
        "Training_Time_s": lr_training_time,
    }
)

print(
    f"Linear Regression - RMSE: {lr_rmse:.2f}, MAE: {lr_mae:.2f}, R2: {lr_r2:.4f}, Time: {lr_training_time:.2f}s"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 2: Random Forest

# COMMAND ----------

start_time = time.time()

# Create pipeline with preprocessing and model
rf_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ]
)

# Hyperparameter grid
param_grid = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__max_depth": [5, 10, 15],
    "regressor__min_samples_leaf": [1, 5],
}

# GridSearchCV for hyperparameter tuning
print("Training Random Forest model with hyperparameter tuning...")
grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=CV_FOLDS,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

rf_training_time = time.time() - start_time

# Get best model
best_rf_model = grid_search.best_estimator_
best_rf_params = grid_search.best_params_

# Make predictions
rf_predictions = best_rf_model.predict(X_test)

# Evaluate
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

# Store results
model_results.append(
    {
        "Model": "RandomForest",
        "RMSE": rf_rmse,
        "MAE": rf_mae,
        "R2": rf_r2,
        "Training_Time_s": rf_training_time,
    }
)

print(
    f"Random Forest - RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}, R2: {rf_r2:.4f}, Time: {rf_training_time:.2f}s"
)
print(f"Best params: {best_rf_params}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 3: Gradient Boosting

# COMMAND ----------

start_time = time.time()

# Create pipeline with preprocessing and model
gb_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(random_state=42)),
    ]
)

# Hyperparameter grid
param_grid = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__max_depth": [3, 5, 7],
    "regressor__learning_rate": [0.05, 0.1, 0.2],
}

# GridSearchCV for hyperparameter tuning
print("Training Gradient Boosting model with hyperparameter tuning...")
grid_search = GridSearchCV(
    gb_pipeline,
    param_grid,
    cv=CV_FOLDS,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

gb_training_time = time.time() - start_time

# Get best model
best_gb_model = grid_search.best_estimator_
best_gb_params = grid_search.best_params_

# Make predictions
gb_predictions = best_gb_model.predict(X_test)

# Evaluate
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_r2 = r2_score(y_test, gb_predictions)

# Store results
model_results.append(
    {
        "Model": "GradientBoosting",
        "RMSE": gb_rmse,
        "MAE": gb_mae,
        "R2": gb_r2,
        "Training_Time_s": gb_training_time,
    }
)

print(
    f"Gradient Boosting - RMSE: {gb_rmse:.2f}, MAE: {gb_mae:.2f}, R2: {gb_r2:.4f}, Time: {gb_training_time:.2f}s"
)
print(f"Best params: {best_gb_params}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison

# COMMAND ----------

# Create comparison DataFrame
comparison_df = pd.DataFrame(model_results)
comparison_df = comparison_df.sort_values("RMSE")

print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)

# Determine best model
best_model_name = comparison_df.iloc[0]["Model"]
best_model_rmse = comparison_df.iloc[0]["RMSE"]

print(f"\nBest Model: {best_model_name} (RMSE: {best_model_rmse:.2f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Model Comparison Results

# COMMAND ----------

# Save comparison to Delta table
comparison_spark_df = spark.createDataFrame(comparison_df)
comparison_spark_df.write.format("delta").mode("overwrite").saveAsTable(
    "model_comparison"
)

print("Model comparison saved to 'model_comparison' table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 80)
print("MODEL TRAINING COMPLETE")
print("=" * 80)
print("Total models trained: 3")
print(f"Best model: {best_model_name}")
print(f"Best RMSE: {best_model_rmse:.2f}")
print(f"Training timestamp: {datetime.now()}")
print("=" * 80)
