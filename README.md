# Credit Card Fraud Detection using Machine Learning

This project is an end-to-end machine learning notebook for detecting fraudulent credit card transactions from an extremely imbalanced dataset. The analysis covers data understanding, exploratory data analysis, feature engineering, preprocessing, class imbalance handling, model training, and model comparison.

## Overview

The notebook uses the publicly available `creditcard.csv` dataset stored in `Dataset/`. It contains 284,807 transactions, of which 492 are fraudulent, making fraud only 0.172% of the data. Because of this imbalance, the project focuses on metrics that are appropriate for rare-event classification, especially Precision-Recall AUC (PR-AUC).

The workflow includes:

1. Business understanding and problem framing
2. Dataset inspection and quality checks
3. Exploratory data analysis
4. Feature engineering
5. Preprocessing and scaling
6. Class imbalance handling with SMOTE
7. Model training and comparison
8. Final model selection and evaluation

## Dataset

The dataset contains the following fields:

- `Time`: Seconds elapsed between the transaction and the first transaction in the dataset.
- `V1` to `V28`: PCA-transformed features.
- `Amount`: Transaction amount.
- `Class`: Target label, where `1` indicates fraud and `0` indicates a legitimate transaction.

Key dataset characteristics:

- Total transactions: 284,807
- Fraudulent transactions: 492
- Fraud rate: 0.172%
- Missing values: none

## Project Goals

- Understand the structure and quality of the transaction data.
- Identify useful patterns that separate fraudulent and legitimate transactions.
- Engineer features that improve model discrimination.
- Address the severe class imbalance in a principled way.
- Compare multiple classifiers using metrics suitable for imbalanced classification.
- Select a final model based primarily on PR-AUC and overall fraud-detection performance.

## Feature Engineering

The notebook creates additional features to improve modeling performance and interpretability, including:

- Hour of transaction
- Log-transformed amount
- High-value transaction indicator
- Night transaction indicator

These features are combined with the original fields to create the final modeling dataset.

## Preprocessing

Preprocessing steps include:

- Data inspection and missing-value analysis
- Duplicate transaction analysis
- Scaling comparison across `StandardScaler`, `MinMaxScaler`, and `RobustScaler`
- Final use of `RobustScaler` for the modeling pipeline
- Train-test split before resampling
- SMOTE to generate synthetic minority-class examples in the training set

## Models Evaluated

The notebook evaluates four classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Model comparison emphasizes:

- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC (PR-AUC)

PR-AUC is treated as the primary metric because the fraud class is the minority class and accuracy alone would be misleading.

## Results Summary

The model comparison shows that Random Forest provides the best overall balance between fraud-detection capability and robustness. It is selected as the final model for detailed evaluation in the notebook.

## Repository Structure

```text
Credit_Card_Fraud_Detection.ipynb
README.md
requirements.txt
Dataset/
	creditcard.csv
```

## Requirements

The project depends on the packages listed in `requirements.txt`, including:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- shap
- jupyter
- notebook
- ipykernel
- scipy

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Notebook

1. Open `Credit_Card_Fraud_Detection.ipynb` in Jupyter or VS Code.
2. Ensure `Dataset/creditcard.csv` is present in the repository.
3. Run the notebook cells from top to bottom.

## Notes

- The project is designed for imbalanced binary classification.
- Metrics and thresholds should be interpreted with the fraud class in mind.
- The notebook is intended as a reproducible analysis and modeling workflow rather than a production deployment package.