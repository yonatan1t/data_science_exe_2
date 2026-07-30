# Kickstarter Project Success Prediction (Homework Assignment 2)

## Overview

This repository contains the solution for **Homework Assignment 2** in Introduction to Data Science (Semester B, 2026). The goal is to predict the success of Kickstarter projects (`state_ind`: `1` for successful, `0` for failed) on a set of unlabelled new projects (`new_projects.csv`), trained on historical project data (`kickstarter_projects.csv`).

As per the TA and assignment guidelines, the final pipeline is restricted to permitted basic models (**Classification Trees** & **K-Nearest Neighbors**). The final model implemented is a tuned **`DecisionTreeClassifier`** with cost-complexity pruning (`ccp_alpha=0.002`) and probability threshold tuning ($p^* = 0.30$).

The evaluation metric used to benchmark and optimize all models was the **$F_1$ score** ($F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$).

---

## Dataset & Feature Engineering

### 1. Raw Data Characteristics
- **Training Set (`kickstarter_projects.csv`)**: 17,904 rows, 31 columns. Target variable: `state_ind` (~30.3% positive class rate).
- **Test Set (`new_projects.csv`)**: 1,990 rows, 30 columns.
- **Missing Values**: Present in `category` (1,632 missing in train, 191 in test) and `name_len`/`name_len_clean` (4 missing in train, 1 in test).

### 2. Feature Engineering & Preprocessing Pipeline
- **Financial & Ratio Features**:
  - `df.currency_rate`: Imputed missing exchange rates with default `1.0`.
  - `goal_usd`: Converted target goal into USD using `goal * df.currency_rate`.
  - `log_goal_usd` & `log_goal`: Log-transformed monetary values ($\log(1+x)$) to stabilize right-skewed distributions.
  - `goal_usd_per_day`: Monetary requirement normalized by campaign duration (`goal_usd / launch_to_deadline_days`).
  - `goal_to_cat_median`: Project monetary goal relative to the median goal in its category (`goal_usd / (category_median_goal + 1)`).
  - `prep_ratio`: Campaign preparation duration relative to active campaign length (`create_to_launch_days / (launch_to_deadline_days + 1)`).
- **Text & TF-IDF Keyword Features**:
  - Extracted title string metadata: character length (`name_char_len`), word count (`name_word_count`), exclamation marks (`!`), question marks (`?`), and all-caps indicators.
  - **TF-IDF Keyword Extraction (`TfidfVectorizer(max_features=100)`)**: Extracted the top 100 most predictive keyword signals from project titles.
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

We evaluated candidate models using **5-Fold Stratified Cross-Validation** on the training dataset. Beyond standard thresholding ($p \ge 0.5$), we performed probability threshold optimization to maximize out-of-fold $F_1$ scores.

### 5-Fold Cross-Validation Comparison Matrix

| Model | Hyperparameters / Setup | Default $F_1$ ($p=0.5$) | Best $F_1$ Score | Optimal Threshold ($p^*$) |
| :--- | :--- | :--- | :--- | :--- |
| **Pruned Decision Tree (Final)** | `max_depth=8, leaf=5, ccp=0.002, entropy` | 0.5892 | **`0.6606`** 🏆 | **`0.30`** |
| **Tuned Decision Tree** | `max_depth=8, leaf=10, entropy` | 0.5910 | **`0.6574`** | **`0.32`** |
| **Decision Tree** | `min_samples_leaf=20` | 0.5832 | `0.6346` | `0.34` |
| **Decision Tree** | `max_depth=10` | 0.5791 | `0.6233` | `0.31` |
| **Decision Tree** | `max_depth=5` | 0.6121 | `0.6141` | `0.35` |
| **K-Nearest Neighbors** | `k=75, weights='distance', p=1` | 0.4912 | `0.6137` | `0.28` |
| **K-Nearest Neighbors** | `k=51` | 0.4678 | `0.6113` | `0.28` |
| **K-Nearest Neighbors** | `k=31` | 0.4857 | `0.6043` | `0.26` |
| **K-Nearest Neighbors** | `k=15` | 0.5058 | `0.5920` | `0.27` |
| **K-Nearest Neighbors** | `k=5` | 0.5141 | `0.5683` | `0.21` |

### Key Findings
1. **Feature Engineering Impact**: Incorporating TF-IDF title keywords and category-relative goal ratios improved the Decision Tree $F_1$ score from `0.6346` to **`0.6574+`**.
2. **Subtree Pruning (`ccp_alpha=0.002`)**: Cost-complexity pruning effectively prevented deep tree branches from overfitting to noise, achieving an out-of-fold $F_1$ score of **`0.6606`**.
3. **Probability Threshold Tuning**: Lowering the decision threshold from 0.50 down to **0.30** optimized the Precision-Recall balance for positive class prediction.

---

## Submission & Verification Artifacts

- **`123456789.py`**: Self-contained Python script implementing the entire data loading, feature engineering, 5-fold CV evaluation, model training, and inference pipeline.
- **`123456789.csv`**: Prediction CSV containing 1,990 rows and exactly 2 columns (`id`, `state_ind_pred`).
- **`123456789.xlsx`**: Excel file formatted for student ID details.
- **`verify_submission.py`**: Automated verification script confirming row counts (1,990 rows), complete ID matching, zero missing values, and zero duplicate IDs.

*(Note for future agents / users: Replace `123456789` with your actual Israeli ID number prior to submission).*

---

## Environment Setup
The project uses Python 3.12 and dependencies installed via official PyPI (`https://pypi.org/simple`):
- `pandas`
- `scikit-learn`
- `openpyxl`
