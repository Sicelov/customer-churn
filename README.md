# Customer Churn Prediction

## Table of Contents

- [Overview](#overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data Cleaning](#data-cleaning)
- [Data Analysis](#data-analysis)
- [Key Findings](#key-findings)
- [Conclusion](#conclusion)

### Overview

This project aimed to build a machine learning model that predicts whether a telecom customer is likely to churn based on their service usage and account information. The goal was to help the company proactively retain high-risk customers by identifying key drivers of churn and visualizing insights.

![Customer Churn Predictor App](https://github.com/Sicelov/customer-churn/blob/main/Customer-curn.png)

### Data Sources

The dataset was sourced from Kaggle’s Telco Customer Churn Dataset, containing information on 7,043 telecom customers including demographics, service usage, billing, and churn status.

### Tools

- Languages & Libraries: Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn, SHAP, Matplotlib, Seaborn
- ML Lifecycle: MLflow for experiment tracking
- Visualization & Deployment: Streamlit, Docker
- Model Explanation: SHAP (SHapley Additive exPlanations)

### Data Cleaning

- Removed missing and blank entries from columns such as TotalCharges
- Converted string-based numerical values into proper numeric types
- Dropped irrelevant columns like customerID
- Applied one-hot encoding to multi-class categorical variables
- Mapped binary categorical variables to numerical (Yes=1, No=0)

### Data Analysis

- Performed exploratory data analysis to understand churn distribution and feature correlations
- Identified class imbalance with ~26% churn rate
- Applied SMOTE to balance the dataset for model training
- Trained an XGBoost Classifier, tuned hyperparameters, and evaluated model using F1 Score, Recall, and Precision
- Used SHAP to interpret feature importance and generate individual prediction explanations

### Key Findings

- Contract type, tenure, MonthlyCharges, and InternetService were key drivers of churn
- Month-to-month contract customers were more likely to churn
- Long-term customers with lower monthly charges showed higher retention
- SHAP visualizations provided actionable insights for targeting high-risk customer segments

### Conclusion

The deployed churn prediction model helps identify customers likely to leave, enabling proactive retention strategies. With real-time prediction through Streamlit and clear explainability via SHAP, the tool is business-ready and can be integrated into CRM systems. The use of MLflow and Docker also makes the solution reproducible and scalable in cloud environments.


