# E-commerce Purchase Intent — Logistic Regression

## Author & Internship
- **Author:** Urooj Fatima
- **Internship:** Machine Learning Internship
- **Company:** Learn Depth

## Project Overview
This project predicts whether an e-commerce website session will result in a purchase (`target=1`) or no purchase (`target=0`) using **Logistic Regression**.

### Dataset
- Rows: 1000
- Predictors: 6
- Target: `target`
- Missing values: 0
- Duplicate rows: 0
- Target balance: 500 class-0 and 500 class-1 rows

### Features
| Feature | Meaning |
|---|---|
| pages_viewed | Number of pages viewed during the session |
| session_minutes | Session duration in minutes |
| products_viewed | Number of products viewed |
| cart_additions | Number of cart additions |
| discount_seen | Discount/promotion exposure measure |
| previous_orders | Number of previous orders |
| target | 1 = purchase, 0 = no purchase |

## Method
1. Inspect data quality and class balance.
2. Use an 80/20 **stratified** train/test split (`random_state=42`).
3. Standardize predictors using `StandardScaler`.
4. Train Logistic Regression using `solver='liblinear'`, `max_iter=1000`.
5. Evaluate Accuracy, Precision, Recall, F1-score and ROC-AUC.
6. Analyze the confusion matrix and standardized coefficients.

Scaling is fitted only on the training data through a Pipeline, which avoids test-set leakage.

## Test Results
Accuracy: 0.7300
Precision: 0.7396
Recall: 0.7100
F1-score: 0.7245
ROC-AUC: 0.8004

Confusion Matrix:
[[75, 25], [29, 71]]

## Practical Interpretation
Positive coefficients indicate that larger values of the feature are associated with higher predicted purchase probability, holding other variables constant. Negative coefficients indicate the opposite. Because predictors are standardized, coefficient magnitudes can be compared approximately across features.

See `outputs/coefficient_analysis.csv` for coefficients and odds ratios.

## Limitations
- The dataset contains only six predictors, so important behavioral, device, traffic-source and product-level factors may be missing.
- A single train/test split provides a useful evaluation but not a complete estimate of generalization.
- Logistic Regression assumes a linear relationship between predictors and log-odds.
- The dataset may be synthetic or simplified, so real-world performance could differ.
- A production system should monitor drift and calibrate the probability threshold according to business costs.

## Run
```bash
pip install -r requirements.txt
python src/analysis.py
```

A notebook version is also included.
