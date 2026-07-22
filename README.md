#  Medical Insurance Charges Prediction

> Predicting insurance charges using Multiple Linear Regression built
> entirely from scratch using NumPy — no scikit-learn.

---

##  Overview

A end-to-end machine learning project that predicts medical insurance
charges based on patient demographics. Every component — gradient
descent, feature scaling, train/test split, and evaluation — is
implemented manually using NumPy and Pandas.

---

##  Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) |
| Rows | 1338 |
| Features | age, sex, bmi, children, smoker, region |
| Target | charges |

---

##  Pipeline
### 1.  Exploratory Data Analysis
- Checked shape, dtypes, missing values and duplicates
- Discovered right skewed distribution in target variable
- Identified top correlated features via heatmap
- Visualized feature distributions via pie charts and scatter plots

### 2.  Preprocessing
- Applied **log transform** on `charges` to normalize skewed distribution
- Encoded categorical columns using `get_dummies`
- Created **interaction feature** `bmi_smoker = bmi × smoker_yes`

### 3. Feature Selection
| Feature | Reason |
|---------|--------|
| age | Strong positive linear relationship with charges |
| bmi | Higher BMI directly correlates with higher charges |
| smoker_yes | Single most impactful feature in the dataset |
| bmi_smoker | Captures combined effect of BMI and smoking |

### 4.  Train/Test Split
- **80% train / 20% test** split
- Done manually using `np.random.permutation`

### 5.  Feature Scaling
Applied standardization manually:

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

- Mean and std computed from **training data only**
- Same parameters applied to test data

### 6. 🧠 Model — Multiple Linear Regression

$$\hat{y} = b + w_1 \cdot age + w_2 \cdot bmi + w_3 \cdot smoker + w_4 \cdot bmi\_smoker$$

Vectorized form:

$$\hat{y} = X\theta$$

### 7. Gradient Descent

$$J(\theta) = \frac{1}{2m} \sum (\hat{y} - y)^2$$

$$\frac{\partial J}{\partial \theta} = \frac{1}{m} X^T (X\theta - y)$$

$$\theta = \theta - \alpha \cdot \frac{\partial J}{\partial \theta}$$

| Hyperparameter | Value |
|---------------|-------|
| Learning rate (α) | 0.1 |
| Epochs | 10000 |
| Initialization | zeros |

---

##  Results

| Metric | Value |
|--------|-------|
| R² (log scale) | 0.7547 |
| MSE (log scale) | 0.2087 |
| RMSE (dollar scale) | $8369.38 |

>  Model predicts insurance charges within **~8300** of the actual value on average.

---

## Key Learnings

- Built gradient descent and vectorization from scratch
- Understood why log transformation is needed for skewed targets
- Learned how interaction features capture combined feature effects
- Evaluated model in the correct scale it was trained on

---

##  Tech Stack
* Python
* Numpy
* Pandas
* Matplotlib
---

##  Project Structure
medicalinsurance/
│
├── medical_insurance.ipynb # main notebook
├── insurance.csv # dataset
└── README.md # project documentation
