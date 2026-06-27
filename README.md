# Credit Card Fraud Detection Dashboard

This project is an advanced machine learning solution for detecting fraudulent credit card transactions. It includes data preprocessing, feature engineering, model training, explainable AI with SHAP, risk segmentation, and an interactive Streamlit dashboard.

## Features

* Data preprocessing and feature engineering
* SMOTE for class imbalance handling
* Model training using LightGBM, XGBoost, and Isolation Forest
* Threshold optimization
* SHAP explainability
* Risk segmentation and fraud pattern analysis
* Interactive Streamlit dashboard

## Dashboard Pages

### Overview

* Total transactions
* Total fraud count
* Detection rate
* Average fraud amount

### Transaction Explorer

* Searchable and filterable transactions
* Live risk score by TransactionID

### SHAP Explainer

* Feature contribution analysis
* Plain-English explanations

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* XGBoost
* SHAP
* Streamlit
* Plotly

## Project Structure

```text
dashboard/
├── app.py
├── dashboard_data.csv
├── model.pkl
├── scaler.pkl
├── label_encoders.pkl
└── threshold.pkl

analysis.ipynb
requirements.txt
README.md
charts/
model_comparison.png
shap_summary.png
```

## 🚀 Live Dashboard

👉 **[Open Live Dashboard](https://fraud-detection-dashboard-b6k4fv3folec2ucahzuoapp.streamlit.app/)**

## Author

Kamal Jetwani

