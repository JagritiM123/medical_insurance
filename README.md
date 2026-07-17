# Medical Insurance Charges — Regression Model

## Overview
A regression model that predicts individual medical insurance charges based on **age**, **BMI**, and **smoking status**, built from scratch using NumPy and Pandas.

---

## Dataset
- 1,300+ data points, 6 columns
- Target variable: `charges`
- Train / test split: 70% / 30%
- All data points are unique with no missing values

---

## Pipeline

### 1. Data Validation
- Core filtering and sanity checks applied before analysis
- Columns affecting `charges` identified and transformed (justifications in code)

### 2. Preprocessing
- Categorical columns encoded to numerical values
- Feature scaling applied via **Z-score normalization**

### 3. Skew Correction
- `charges` histogram showed a right-skewed distribution
- **Log transformation** applied to normalize it for training
- Predictions converted back using the **antilog**

---

## Features Used

| Feature | Type |
|---|---|
| `age` | Numerical |
| `bmi` | Numerical |
| `smoker_yes` | Binary (1 = smoker, 0 = non-smoker) |

---

## Visualizations

| Plot | Purpose |
|---|---|
| Histogram | Shows skew in `charges`; motivates log transform |
| Pie chart | Proportional influence of age, BMI, and smoker status |
| Scatter plots | `charges_log` vs. `age`, `bmi`, `smoker_yes` — checks data spread |

---

## Tech Stack
- Python, NumPy, Pandas
- Matplotlib (visualizations)
- No scikit-learn — model built from scratch
