# E-Commerce Customer Behaviour and Purchase Prediction

A machine learning project that uses historical e-commerce transaction
data to study customer purchasing behaviour and, ultimately, predict
whether an existing customer will make another **paid purchase within
the next 90 days**.

> **Project status:** Data integration, initial cleaning and exploratory
> data analysis (EDA) have been completed. Feature engineering, target
> construction, model development and evaluation are planned next.

## 1. Project Overview

Understanding which customers are likely to purchase again can help an
e-commerce business improve customer retention, plan targeted campaigns
and make better decisions.

This project uses the **Online Retail II** transaction dataset. The
original records are at transaction-line level. The planned modelling
workflow will aggregate eligible transactions into customer-level
features and construct a time-based binary target:

-   `1`: the customer makes at least one qualifying paid purchase during
    the following 90-day prediction window.
-   `0`: the customer does not make a qualifying paid purchase during
    that window.

The prediction will use information available before the prediction
window begins. This chronological design is intended to prevent data
leakage.

## 2. Dataset

**Dataset:** Online Retail II\
**Source:** UCI Machine Learning Repository\
**Dataset page:**
https://archive.ics.uci.edu/dataset/502/online+retail+ii

The source workbook contains two sheets covering different periods:

-   `Year 2009-2010`
-   `Year 2010-2011`

The sheets have the same structure and are combined vertically
(row-wise) for analysis.

### Original columns

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  `Invoice`                           Invoice identifier. Invoices
                                      beginning with `C` indicate
                                      cancellations.

  `StockCode`                         Product or stock identifier.

  `Description`                       Product description.

  `Quantity`                          Quantity recorded on the invoice
                                      line.

  `InvoiceDate`                       Date and time of the transaction.

  `Price`                             Unit price.

  `Customer ID`                       Customer identifier; some records
                                      do not have an identified customer.

  `Country`                           Customer's country.
  -----------------------------------------------------------------------

A derived column, `TotalAmount`, is calculated as:

``` python
df['TotalAmount'] = df['Quantity'] * df['Price']
```

## 3. Data Preparation and Cleaning

The two source sheets were combined into one transaction dataset. The
following cleaning operations were performed during the initial
preparation:

-   Removed exact duplicate records.
-   Filled missing product descriptions with `Unknown`.
-   Retained missing `Customer ID` values in the general transaction
    dataset because those records may still be useful for
    transaction-level analysis. Records without a customer identifier
    will be excluded when constructing customer-level modelling data.
-   Converted `InvoiceDate` to a datetime data type.
-   Converted `Customer ID` to Pandas nullable integer type (`Int64`).
-   Created `TotalAmount` from `Quantity × Price`.
-   Removed negative-quantity records that were not associated with
    cancellation invoices, based on the investigation performed.
-   Retained cancellation records in the transaction history, but they
    must be excluded from qualifying paid-purchase calculations.
-   Removed zero-unit-price records for the paid-purchase analysis.
-   Removed five negative-price records identified as `Adjust bad debt`.

The original combined dataset was preserved separately from the cleaned
working dataset.

**Important:** The cleaning decisions above describe the current project
workflow. The final modelling dataset will apply additional
purchase-specific filters and chronological rules.

## 4. Exploratory Data Analysis (EDA)

EDA was performed to understand the distributions, relationships and
patterns in the transaction data.

### Analyses performed

1.  **Univariate analysis**
    -   Categorical frequency distributions.
    -   Numerical descriptive statistics.
    -   Histograms and distribution plots.
    -   Skewness analysis.
2.  **Outlier analysis**
    -   Box plots.
    -   Interquartile Range (IQR) method.
    -   Outliers were investigated rather than automatically deleted,
        since extreme values may represent genuine bulk purchases or
        expensive products.
3.  **Bivariate analysis**
    -   Quantity versus TotalAmount.
    -   Price versus TotalAmount.
    -   Pearson correlation analysis.
4.  **Country-wise analysis**
    -   Country transaction-record frequency.
    -   Average and total transaction amounts.
    -   Country-level comparisons.
5.  **Time-based analysis**
    -   Monthly transaction activity.
    -   Day-of-week activity.
    -   Hourly transaction activity.
6.  **Customer behaviour analysis**
    -   Customer purchase frequency.
    -   Customer spending.
    -   Average transaction values.
7.  **Multivariate analysis**
    -   Numerical correlation heatmap.
    -   Country-wise monthly transaction amount heatmap.
8.  **Statistical analysis**
    -   An independent Welch's t-test was used to compare transaction
        amounts between the United Kingdom and other countries.

### EDA interpretation notes

-   Quantity and Price showed strong positive skewness in the cleaned
    transaction data.
-   TotalAmount showed negative skewness in the observed analysis.
-   The IQR method flagged extreme observations in Quantity, Price and
    TotalAmount. These were not automatically removed.
-   Quantity and TotalAmount showed a strong positive Pearson
    correlation in the analysis.
-   Price and TotalAmount showed a weak negative Pearson correlation in
    the analysis.

These are exploratory findings, not causal conclusions. In particular,
`TotalAmount` is mathematically derived from Quantity and Price, and
transaction-level records from the same customer are not statistically
independent.

## 5. Planned Machine Learning Objective

The main prediction task is **90-day repeat-purchase classification**.

For each eligible customer:

-   Build features from transactions in a historical observation window.
-   Define a later, non-overlapping 90-day prediction window.
-   Set the target to `1` if the customer makes at least one qualifying
    paid purchase during that future window; otherwise set it to `0`.

The exact observation-window dates and eligibility rules must be defined
before constructing the modelling dataset. Customers must have enough
historical and future observation coverage to be labelled correctly.

### Planned customer-level features

Potential features include:

  -----------------------------------------------------------------------
  Feature                             Meaning
  ----------------------------------- -----------------------------------
  Recency                             Time since the customer's most
                                      recent qualifying purchase in the
                                      observation window.

  Frequency                           Number of distinct purchase
                                      invoices in the observation window.

  Monetary                            Total qualifying purchase amount in
                                      the observation window.

  Average Order Value                 Average invoice-level purchase
                                      value.

  Total Quantity                      Total quantity purchased in the
                                      observation window.

  Purchase Interval                   Typical time between a customer's
                                      purchases, where sufficient history
                                      exists.
  -----------------------------------------------------------------------

Only features calculated from information available before the
prediction window will be used.

## 6. Planned Modelling Workflow

The next stages are:

1.  Create a purchase-only transaction dataset, excluding cancellations
    and invalid/non-paid records according to the project definition.
2.  Select chronological observation and 90-day prediction windows.
3.  Aggregate historical transactions into customer-level features.
4.  Create and inspect the binary target.
5.  Check class balance and investigate potential leakage.
6.  Split data using a time-aware strategy.
7.  Build preprocessing pipelines so transformations are fitted on
    training data only.
8.  Train a baseline and at least three classification models.
9.  Tune selected model hyperparameters using training data only.
10. Evaluate on held-out data.

### Planned models

Candidate models include:

-   Dummy Classifier (baseline)
-   Logistic Regression
-   Decision Tree
-   Random Forest
-   Gradient-boosting model, such as XGBoost, if available and
    appropriate

### Planned evaluation metrics

-   Precision
-   Recall
-   F1-score
-   ROC-AUC
-   PR-AUC
-   Confusion matrix

Accuracy will not be used alone, particularly if the target classes are
imbalanced. Model interpretation and feature importance will also be
examined.

## 7. Repository Structure

The project is being organized into separate notebooks and data folders.
A suggested structure is:

``` text
MLProject/
│
├── data/
│   ├── online_retail_combined.csv
│   └── online_retail_working.csv
│
├── notebooks/
│   ├── 01_Data_Inspection.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 04_Exploratory_Data_Analysis_(EDA).ipynb
│   └── 05_Feature_Engineering.ipynb
│
├── outputs/
├── models/
├── README.md
└── requirements.txt
```

Update the structure as additional notebooks, scripts and outputs are
created. Do not commit large datasets or sensitive files to a public
repository unless their licensing and distribution terms have been
checked.

## 8. Tools and Technologies

-   Python
-   Jupyter Notebook
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   SciPy
-   Scikit-learn

Additional libraries may be introduced during model development.

## 9. How to Run

1.  Clone or download this repository.

2.  Install the required packages:

    ``` bash
    pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
    ```

3.  Download the Online Retail II dataset from the UCI source linked
    above.

4.  Place the source workbook or prepared CSV files in the `data/`
    directory.

5.  Run the notebooks in the documented workflow order, updating file
    paths if necessary.

The current notebook uses a local Windows path in places. Replace
machine-specific paths with relative project paths before sharing the
project.

## 10. Limitations and Considerations

-   Customer IDs are missing for a portion of the source transactions.
    Such records cannot be assigned reliably to individual customers and
    are excluded from customer-level modelling.
-   Cancellation invoices and returns must not be counted as successful
    paid purchases in the target.
-   Zero-price records were excluded from the current paid-purchase
    analysis; this is a project definition and should be documented
    consistently.
-   Extreme values may be genuine business activity. They should be
    investigated and handled based on their meaning and the modelling
    method, not removed solely because they exceed an IQR threshold.
-   The transaction history has a finite date range. The observation and
    prediction windows must fit within the available dates.
-   Time-based splitting is required to ensure that evaluation reflects
    prediction on future periods.
-   Statistical significance does not necessarily imply practical
    importance.

## 11. Project Status

  Stage                                 Status
  ------------------------------------- --------------------------
  Dataset collection and integration    Completed
  Initial data inspection               Completed
  Initial data cleaning                 Completed
  Exploratory data analysis             Completed / under review
  Feature engineering                   Next
  90-day target construction            Pending
  Model development                     Pending
  Model evaluation and interpretation   Pending
  Final report and presentation         Pending

## 12. Acknowledgement

The transaction data used in this project is the **Online Retail II**
dataset provided through the UCI Machine Learning Repository. Please
refer to the original dataset page for its documentation, citation and
usage information:


https://archive.ics.uci.edu/dataset/502/online+retail+ii
