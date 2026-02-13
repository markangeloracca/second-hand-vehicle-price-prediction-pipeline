[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/xh_awDha)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=21861833&assignment_repo_type=AssignmentRepo)
# 612-Project
ENSF 612 - 2025 - Code Repository


# Second-Hand Vehicle Price Prediction Pipeline

## Engineering Problem
Develop a scalable ML pipeline to predict second-hand vehicle prices in the Toronto market, addressing real-world pricing accuracy challenges. The solution will handle diverse features, missing data, and process thousands of records efficiently.

## Tools & Platform
| Component    | Technology | Purpose |
| -------- | ------- | ------- |
| **Execution Platform**  | Databricks Free Tier    |  |
| **Data Processing** | PySpark | Distributed DataFrame operations |
| **Feature Encoding**    | StringIndexer/OneHotEncoder | Categorical feature transformation |
| **Feature Scaling** | StandardScaler | Numerical feature normalization |
| **Storage & Pipeline** | Delta Lake on Databricks | ACID transactions, medallion architecture |

## Machine Learning Models
| Model    | Approach | Purpose |
| -------- | ------- | ------- |
| **Linear Regression** | Simple linear relationships | Baseline performance |
| **Random Forest** | Ensemble of decision trees | Non-linear patterns |
| **Gradient Boosting** | Sequential tree boosting | Enhanced accuracy |

## Dataset Description
**Source**: Kaggle - Used Vehicles (Toronto 2023, Farhan Hossein) | **Records**: 24,199 vehicles | **Geography**: Toronto area (within 25km of downtown) | **Features**: Price (target), make, model, year, mileage, condition, fuel type, transmission, body type, drivetrain from Autotrader.ca

## Big Data Engineering Relevance
| Concept | Implementation |
| -------- | ------- |
| **Distributed Processing **   | PySpark patterns on Databricks (scalable to larger data) |
| **Pipeline Architecture** | Medallion pattern (Bronze→Silver→Gold) on Delta Lake |
| **Data Reliability** | ACID transactions for versioning & governance |
| **Scalability** | ML pipeline extensible to larger datasets beyond free tier |
| **Reproducibility** | MLflow experiment tracking on Databricks |



