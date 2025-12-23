# 🏦 AI Credit Risk Analyzer

## 🚀 Live Demo
**[Click here to view the App](YOUR_STREAMLIT_LINK_HERE)**

## 📌 Overview
This project is an end-to-end Machine Learning application that predicts loan default probability. It helps loan officers make data-driven decisions by analyzing applicant info and bureau history.

## 🛠️ Tech Stack
* **Python** (Pandas, NumPy)
* **Machine Learning:** LightGBM, Scikit-Learn
* **Model Interpretability:** SHAP
* **Web App:** Streamlit
* **Deployment:** Streamlit Community Cloud

## 📊 Key Features
* **Real-time Risk Scoring:** Calculates default probability instantly.
* **Explainable AI:** Uses SHAP values to explain *why* a loan was rejected.
* **"Bureau Boost":** Incorporates aggregated credit history data for higher accuracy.

## 📂 Project Structure
* `app.py`: The main Streamlit application.
* `notebooks/`: Jupyter notebooks for EDA, Feature Engineering, and Training.
* `models/`: Trained LightGBM models saved as .joblib files.