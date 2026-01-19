# Semiconductor Wafer Quality Prediction (ML)

Wafer-level quality prediction using multivariate time-series sensor data from semiconductor manufacturing.

This project builds a machine learning pipeline to predict wafer quality using real production sensor data from two different machines. Each wafer contains time-series signals from multiple sensors, but only one quality label is given per wafer. Because of this, the project focuses heavily on feature engineering and careful evaluation rather than complex models.

The main goal is to convert raw time-series sensor data into meaningful wafer-level features, handle different types of sensor behavior, and evaluate models in a way that makes sense for imbalanced manufacturing data.

---

## Problem Overview

During manufacturing, sensor data is recorded continuously as a wafer goes through multiple machines. This results in detailed time-series signals. However, quality inspection happens only once, after the wafer is fully processed. Because of this, each wafer has a single quality label that cannot be matched to individual sensor readings over time.

This creates several challenges:

- Multivariate time-series data mapped to a single wafer-level label
- Sensors with highly diverse behaviors
  - Continuous signals
  - Sparse or event-triggered sensors
  - Near-constant or binary-like channels
- Strong class imbalance (approximately 20% defective wafers)
- Redundant and highly correlated sensor channels

---

## Dataset Description

### Equipment Sensor Data

**Equipment 1**
- 24 sensors
- 176 timestamps per wafer
- Approximately 170,000 total rows

**Equipment 2**
- 32 sensors
- 176 timestamps per wafer
- Approximately 230,000 total rows

Each row corresponds to:
(lot, wafer, timestamp, sensor_*)

### Response Data

- One row per wafer
- Columns:
  - `response`: continuous quality score
  - `class` : categorical label (`good` / `bad`)
- Class distribution:
  - Good: about 80%
  - Bad: about 20%

The `class` label is a thresholded version of the continuous `response` score.

---

## Exploratory Data Analysis (EDA)

### Sensor Behavior
- Mixture of continuous, sparse, and near-constant sensors
- Significant scale differences across sensors
- Strong sensor redundancy indicated by high correlations

### Time-Series Structure
- No missing values
- No duplicated `(lot, wafer, timestamp)` tuples
- All wafers contain exactly 176 timestamps

### Label Insights
- Severe class imbalance
- Certain lots consistently produce defective wafers
- Extreme response outliers correspond to true failures rather than noise

**Conclusion**
Raw time-series data should not be modeled directly. Sensor-specific aggregation into wafer-level features is essential.

---

## Feature Engineering

Time-series signals are independently summarized per wafer for each piece of equipment.

### Statistical Features
For each sensor
- Mean, standard deviation, minimum, maximum
- Median and range
- 25th and 75th percentiles

### Shape and Trend Features
To capture temporal dynamics
- Linear slope
- First-derivative variance
- Number of peaks
- Rising and falling step counts

### Sparse-Sensor Activation Features
For zero-heavy sensors
- Zero ratio
- Activation count
- Longest active duration
- Mean value during activation periods

### Feature Cleaning
- Removal of near-constant features
- Removal of duplicated features

**Final feature matrix**
- 971 wafers
- 556 cleaned features
- Combined features from both equipment sources and response labels

---

## Key Challenges and Design Decisions

### Challenge 1: Time-series data with wafer-level labels
Each wafer contains 176 timestamps of multivariate sensor readings, but only a single wafer-level quality label is available. Sequence models were avoided because labels could not be meaningfully aligned to individual timesteps.

**Decision**
Time-series signals were aggregated into wafer-level features using sensor-specific statistical and shape-based summaries.

---

### Challenge 2: Diverse sensor behavior
Some sensors exhibited smooth continuous behavior, while others were sparse or near-constant.

**Decision**
Sensors were grouped by behavior, and different feature extraction strategies were applied, including statistical summaries for continuous sensors and activation-based features for sparse sensors.

---

### Challenge 3: Severe class imbalance
Approximately 20% of wafers were defective.

**Decision**
Model evaluation prioritized recall and F1-score for the defective class rather than overall accuracy to reflect manufacturing risk.

---

### Challenge 4: High feature redundancy
Many engineered features were strongly correlated or near-constant.

**Decision**
Redundant and low-variance features were removed prior to modeling to reduce noise and mitigate overfitting.

---

## Model Training and Evaluation

Models were trained using a stratified train/test split (80/20).
The task was formulated as a binary classification problem, with defective wafers treated as the positive class.

### Models Evaluated
- Decision Tree
- Support Vector Machine
- Random Forest
- Gradient Boosting
- XGBoost

### Evaluation Metrics
- Precision, Recall, and F1-score (focused on defective wafers)
- ROC-AUC
- Confusion Matrix analysis

---

| Model | Precision (Bad) | Recall (Bad) | F1 | ROC-AUC|

| Decision Tree (tuned) | 0.48 | 0.79 | 0.60 | 0.81 |
| SVM (tuned) | 0.46 | 0.82 | 0.59 | 0.89 |
| Random Forest (tuned) | 0.54 | 0.82 | 0.65 | 0.90 |
| Gradient Boosting (tuned) | 0.81 | 0.65 | 0.72 | 0.92 |
| XGBoost (tuned) | 0.78 | 0.74 | 0.76 | 0.90 |

**Best Model - XGBoost**

XGBoost achieved the strongest balance between precision and recall for defective wafers, resulting in the highest F1-score while remaining robust to high-dimensional engineered features.

---

## Takeaways

- Real industrial time-series data requires problem-aware feature engineering
- Sensor diversity must be explicitly handled rather than averaged away
- Recall-oriented evaluation is critical in manufacturing quality control settings

## Future Work
- Regression modeling on continuous `response` values
- Use SHAP to analyze feature importance and better understand which sensor patterns contribute most to defective predictions
- Apply Optuna for systematic hyperparameter optimization, focusing on recall and F1-score for defective wafers
