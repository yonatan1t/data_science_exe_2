# Kickstarter Project Success Prediction (Homework Assignment 2)

## Overview

This repository contains the solution for **Homework Assignment 2** in Introduction to Data Science (Semester B, 2026). The goal is to predict the success of Kickstarter projects (`state_ind`: `1` for successful, `0` for failed) on a set of unlabelled new projects (`new_projects.csv`), trained on historical project data (`kickstarter_projects.csv`).

As per the assignment guidelines, only supervised classification models taught in the course were permitted:
- **Logistic Regression**
- **Classification Trees** (`DecisionTreeClassifier`)
- **K-Nearest Neighbors (k-NN)** (`KNeighborsClassifier`)

The evaluation metric used to benchmark and optimize all models was the **$F_1$ score** ($F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$).

---

## Dataset & Feature Engineering

### 1. Raw Data Characteristics
- **Training Set (`kickstarter_projects.csv`)**: 17,904 rows, 31 columns. Target variable: `state_ind` (~30.3% positive class rate).
- **Test Set (`new_projects.csv`)**: 1,990 rows, 30 columns.
- **Missing Values**: Present in `category` (1,632 missing in train, 191 in test) and `name_len`/`name_len_clean` (4 missing in train, 1 in test).

### 2. Feature Engineering & Preprocessing Pipeline
- **Financial Features**:
  - `df.currency_rate`: Handled missing values with default `1.0`.
  - `goal_usd`: Converted goal into USD using `goal * df.currency_rate`.
  - `log_goal_usd` & `log_goal`: Log-transformed monetary values ($\log(1+x)$) to stabilize right-skewed distributions.
  - `goal_usd_per_day`: Monetary requirement normalized by campaign duration (`goal_usd / launch_to_deadline_days`).
- **Text Features**:
  - Extracted string metadata from `name`: word count, character length, indicator for exclamation marks (`!`), question marks (`?`), and all-caps titles.
- **Date & Cyclic Features**:
  - Encoded `launched_at_month`, `launched_at_hr`, `deadline_month`, and `deadline_hr` using Sine/Cosine trigonometric transformations ($\sin, \cos$).
- **Categorical & Boolean Features**:
  - Converted `staff_pick` to binary integer (`0`/`1`).
  - Filled missing `category` values with `'Missing'`.
  - Applied `OneHotEncoder(handle_unknown='ignore')` to `country`, `currency`, `category`, and weekday variables.
- **Numerical Scaling**:
  - Applied `SimpleImputer(strategy='median')` followed by `StandardScaler()` to all continuous numeric features.

---

## Research & Model Benchmarking

We evaluated candidate models using **5-Fold Stratified Cross-Validation** on the training dataset. Beyond standard thresholding ($p \ge 0.5$), we performed probability threshold optimization to maximize the $F_1$ score.

### Cross-Validation Comparison Matrix

| Model | Hyperparameters / Setup | Default $F_1$ ($p=0.5$) | Best $F_1$ Score | Optimal Threshold ($p^*$) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `C=1.0, class_weight='balanced'` | **0.6551** | **0.6551** | **0.49** |
| **Logistic Regression** | `C=0.1, class_weight='balanced'` | 0.6542 | 0.6542 | 0.50 |
| **Logistic Regression** | `C=10.0, class_weight='balanced'` | 0.6544 | 0.6544 | 0.50 |
| **Logistic Regression** | `C=1.0, default weights` | 0.5999 | 0.6547 | 0.31 |
| **Classification Tree** | `max_depth=8, min_samples_leaf=10` | 0.6121 | 0.6332 | 0.30 |
| **Classification Tree** | `max_depth=12, min_samples_leaf=20` | 0.5791 | 0.6215 | 0.31 |
| **K-Nearest Neighbors** | `k=51, weights='distance'` | 0.4678 | 0.6034 | 0.29 |
| **K-Nearest Neighbors** | `k=25, weights='distance'` | 0.5141 | 0.5928 | 0.28 |

### Key Findings
1. **Logistic Regression with Balanced Class Weights** outperformed Decision Trees and k-NN by a significant margin. The balanced class weight penalizes false negatives appropriately, helping the linear decision boundary align well with high $F_1$ performance.
2. **Probability Threshold Tuning**: Adjusting the decision threshold from 0.50 to 0.49 yielded the optimal trade-off between precision and recall.

---

## Submission Artifacts

- **`123456789.py`**: Self-contained Python script implementing the entire data loading, feature engineering, model training, and inference pipeline.
- **`123456789.csv`**: Prediction CSV containing 1,990 rows and exactly 2 columns (`id`, `state_ind_pred`).
- **`123456789.xlsx`**: Excel file formatted for student ID details.

*(Note for future agents / users: Replace `123456789` with your actual Israeli ID number prior to submission).*

---

## Environment Setup
The project uses Python 3.12 and dependencies installed via official PyPI (`https://pypi.org/simple`):
- `pandas`
- `scikit-learn`
- `matplotlib`
- `openpyxl`
