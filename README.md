# E-Commerce Customer Behaviour and Purchase Prediction

**An End-to-End Machine Learning Project | CSE570 – Machine Learning with Python**

## Project Overview

This project focuses on analysing e-commerce customer purchasing behaviour and developing machine learning models to predict whether an existing customer will make another purchase within the next 90 days.

Using the **Online Retail II dataset**, the project follows an end-to-end machine learning workflow involving data collection, integration, data cleaning, exploratory data analysis (EDA), feature engineering, model development and evaluation.

The primary objective is to identify purchasing patterns and understand how historical customer behaviour can help predict future purchase activity.

## Problem Statement

E-commerce businesses need to understand customer purchasing behaviour to improve customer retention and develop effective marketing strategies.

This project addresses the problem through a supervised binary classification approach.

The prediction task is to determine whether a customer who has previously made purchases will make at least one additional qualifying purchase during the subsequent 90-day prediction window.

**Target Variable:** `RepeatPurchase_90D`

- `1` – Customer makes at least one qualifying purchase within the next 90 days.
- `0` – Customer does not make a qualifying purchase within the next 90 days.

## Project Objectives

- Analyse historical e-commerce transaction data.
- Perform data cleaning and preprocessing.
- Conduct univariate, bivariate and multivariate EDA.
- Identify customer purchasing patterns and trends.
- Engineer customer-level behavioural features.
- Develop and compare multiple classification models.
- Evaluate model performance using appropriate classification metrics.
- Interpret important features influencing model predictions.

## Dataset

**Dataset:** Online Retail II

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/502/online+retail+ii

The dataset contains transactions from a UK-based online retail business.

| Attribute | Details |
|---|---|
| Original combined records | 1,067,371 |
| Original features | 8 |
| Transaction period | December 2009 – December 2011 |
| Source format | Excel workbook |
| Dataset type | Transactional data |

### Original Dataset Features

| Feature | Description |
|---|---|
| Invoice | Invoice number |
| StockCode | Product identifier |
| Description | Product description |
| Quantity | Number of products purchased |
| InvoiceDate | Transaction date and time |
| Price | Unit price |
| Customer ID | Customer identifier |
| Country | Customer's country |

An additional transaction-level feature, `TotalAmount`, was created:

`TotalAmount = Quantity × Price`

## Project Workflow

The project is organised into seven Jupyter notebooks, following a sequential machine learning workflow.

| Notebook | Description |
|---|---|
| 1. Data Collection & Integration | Loading and combining the two Excel sheets |
| 2. Data Understanding & Initial Inspection | Understanding dataset structure, data types and missing values |
| 3. Data Cleaning | Handling duplicates, missing values, invalid records and data types |
| 4. Exploratory Data Analysis (EDA) | Univariate, bivariate and multivariate analysis |
| 5. Feature Engineering & Transformation | Creating customer-level features and the prediction target |
| 6. Model Development | Data splitting, preprocessing, model training and saving |
| 7. Model Evaluation & Interpretation | Performance comparison, feature importance and model interpretation |

## Data Cleaning and Preprocessing

The following operations were performed:

- Combined the two sheets of the original Excel workbook.
- Identified and removed duplicate records.
- Handled missing product descriptions.
- Retained missing Customer IDs in the transaction dataset for general transaction-level analysis.
- Converted `InvoiceDate` into datetime format.
- Converted `Customer ID` into a nullable integer type.
- Identified cancellation invoices and negative quantities.
- Removed invalid non-cancellation negative-quantity records.
- Removed zero-price and negative-price transactions from the qualifying paid-purchase dataset.
- Created the `TotalAmount` feature.

Outliers were inspected using statistical methods and visualisations rather than automatically removing all extreme values, as some may represent genuine transactions.

## Exploratory Data Analysis

EDA was performed to understand transaction characteristics and customer purchasing behaviour.

The analysis included:

- Categorical and numerical feature analysis.
- Descriptive statistics.
- Histograms and box plots.
- Skewness analysis and outlier inspection using the IQR method.
- Country-wise transaction analysis.
- Product and stock-code frequency analysis.
- Quantity versus transaction amount analysis.
- Price versus transaction amount analysis.
- Monthly, weekly and hourly transaction patterns.
- Customer-level purchasing behaviour.
- Correlation analysis and heatmaps.
- Statistical hypothesis testing.

These analyses helped identify purchasing patterns and provided a foundation for customer-level feature engineering.

## Feature Engineering

The modelling dataset was created using a historical observation period and a subsequent 90-day prediction period.

### Prediction Window

| Period | Dates |
|---|---|
| Observation period | 1 December 2009 – 9 September 2011 |
| Prediction period | 10 September 2011 – 8 December 2011 |
| Prediction horizon | 90 days |

Only customers with qualifying historical purchases were included in the customer-level modelling dataset.

### Engineered Features

Six customer-level features were created:

| Feature | Description |
|---|---|
| Recency | Number of days since the customer's last purchase |
| Frequency | Number of unique historical invoices |
| Monetary | Total historical spending |
| AverageOrderValue | Average value of historical invoices |
| ProductDiversity | Number of unique products purchased |
| AvgItemsPerOrder | Average quantity of items per invoice |

The target variable `RepeatPurchase_90D` was generated using customer purchase activity during the subsequent 90-day prediction period.

### Feature Transformation

Skewed numerical features were examined and `log1p` transformation was applied to:

- Frequency
- Monetary
- AverageOrderValue
- ProductDiversity
- AvgItemsPerOrder

Recency was retained without logarithmic transformation.

StandardScaler was incorporated into preprocessing pipelines for scale-sensitive algorithms.

## Machine Learning Models

This project uses supervised binary classification.

The following models were developed and evaluated:

1. Dummy Classifier – Baseline model
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. K-Nearest Neighbors (KNN)
6. Support Vector Machine (SVM)

### Data Splitting

The customer-level dataset contains 5,281 records and six model features.

The dataset was divided using:

- Training set: 80%
- Testing set: 20%
- Random state: 42
- Stratified splitting based on the target variable

Preprocessing was performed through scikit-learn pipelines for Logistic Regression, KNN and SVM.

Tree-based models were trained without feature scaling.

## Model Evaluation

Models were compared using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

### Initial Model Comparison

The following results were obtained before hyperparameter tuning.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7351 | 0.7119 | 0.6528 | 0.6811 | 0.7968 |
| Logistic Regression | 0.7266 | 0.6979 | 0.6507 | 0.6734 | 0.8076 |
| SVM | 0.7256 | 0.7049 | 0.6310 | 0.6659 | 0.7885 |
| KNN | 0.7001 | 0.6643 | 0.6223 | 0.6426 | 0.7441 |
| Decision Tree | 0.6386 | 0.5841 | 0.5764 | 0.5802 | 0.6313 |
| Dummy Classifier | 0.5667 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

### Initial Results

- Random Forest achieved the highest accuracy of approximately 73.51%.
- Random Forest achieved the highest F1-score of 0.6811.
- Logistic Regression achieved the highest ROC-AUC of 0.8076.
- The trained classification models outperformed the Dummy Classifier baseline.

Based on accuracy and F1-score, Random Forest is the current leading model candidate.

### Model Interpretation

Random Forest feature importance and Logistic Regression coefficients were used to understand model behaviour.

The Random Forest model identified Recency and Monetary as particularly influential features.

Logistic Regression coefficients provided additional insight into the direction of the associations between customer behaviour features and predicted repeat-purchase likelihood.

These feature interpretation results describe model behaviour and should not be interpreted as causal relationships.

## Technologies and Libraries

**Programming Language:** Python

**Development Environment:** Jupyter Notebook

**Libraries:**

- pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- scikit-learn
- joblib
- openpyxl

## Repository Structure

```text
CSE570_MachineLearningProject/
│
├── data/
│   ├── customer_features.csv
│   ├── online_retail_combined.csv
│   ├── online_retail_working.csv
│   └── test_dataset.csv
│
├── models/
│   ├── baseline.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── knn.pkl
│   └── svm.pkl
│
├── notebooks/
│   ├── 1. Data Collection & Integration.ipynb
│   ├── 2. Data Understanding & Initial Inspection.ipynb
│   ├── 3. Data Cleaning.ipynb
│   ├── 4. Exploratory Data Analysis.ipynb
│   ├── 5. Feature Engineering & Transformation.ipynb
│   ├── 6. Model Development.ipynb
│   └── 7. Model Evaluation & Interpretation.ipynb
│
├── outputs/
│
├── README.md
└── requirements.txt
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd CSE570_MachineLearningProject
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks folder and execute the notebooks in numerical order.

**Note:** Update dataset and model file paths in the notebooks if your local directory structure differs from the original development environment.

## Limitations

- The modelling dataset is based on a single observation cutoff and a 90-day prediction window.
- The evaluation uses a random train-test split rather than multiple chronological validation periods.
- The test set was used for model comparison and is therefore not an untouched final holdout.
- Hyperparameter tuning has not yet been performed.
- The dataset represents a historical retail business and may not reflect current e-commerce purchasing behaviour.
- Feature importance and model coefficients indicate associations rather than causal effects.

## Future Improvements

- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Evaluate models using additional chronological validation windows.
- Investigate threshold optimisation for repeat-purchase classification.
- Explore additional customer behavioural features.
- Compare model performance after tuning.
- Develop a simple prediction interface for customer-level predictions.
- Prepare a comprehensive project report.

## References

1. Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*.

## Academic Project

**Course:** CSE570 – Machine Learning with Python

**Project Type:** End-to-End Machine Learning Classification Project

**Specialisation:** Data Science and Analytics

---

*This project is being developed as part of an academic machine learning assignment. Model results and methodology may be updated as further validation and experimentation are completed.*
