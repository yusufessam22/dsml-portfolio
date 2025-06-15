# yusuf_dsml-portfolio
Yusuf' data science and machine learning projects

Project 1: Forecasting Univariate Daily Rainfall for an Undisclosed Station in Malaysia (2025 - ongoing)

# 🌧️ Rainfall Forecasting with Machine Learning (Sunway University Project)

This project presents an end-to-end machine learning pipeline for **daily rainfall forecasting** in Malaysia. Designed to handle a small and imbalanced time series dataset, the project combines real-world engineering challenges with modern ML techniques. It is part of an ongoing research initiative at Sunway University.

> 🚧 **Status**: Work in Progress (Actively Developed)

---

## ✅ What’s Been Done So Far

### 📦 Project Setup
- Reproducible environment with fixed random seeds.
- Automated package installation and modular structure for scalability.

### 📊 Data Exploration & Preprocessing
- Inspected, interpolated, and cleaned raw daily rainfall data.
- Explored statistical characteristics, trends, and seasonality.
- Engineered domain-specific features:
  - Lagged rainfall
  - Accumulated rainfall
  - Rainfall intensity
  - Rainfall spell tracking
  - Seasonality encodings

### 🤖 Forecasting Models Implemented
- **XGBoost**, **LightGBM**, and **CatBoost** with:
  - Baseline performance runs
  - Feature selection
  - Merged train-validation sets
- **Feedforward Neural Network (FNN)** using TensorFlow
- Groundwork laid for **LSTM** and **Transformer** architectures
- Hyperparameter tuning with **Optuna**
- Model explainability with **SHAP**

### 🧪 Experiment Management
- Structured experiments across multiple configurations.
- “Workspace” section for model testing, ablation, and sandboxing.

---

## 🧠 Technical Stack

`Python`, `Pandas`, `NumPy`, `Seaborn`, `Matplotlib`, `XGBoost`, `LightGBM`, `CatBoost`, `Optuna`, `TensorFlow`, `SHAP`, `Statsmodels`, `Scikit-learn`, `PyTorch`, `Graphviz`

---

## 🔜 Next Steps

- Complete training and evaluation of **LSTM** and **Transformer** models.
- Integrate external APIs for **live data ingestion**.
- Deploy model using **FastAPI** or **AWS Lambda** as a REST service.
- Build a simple **web dashboard or Power BI report** for rainfall visualization.

---

## 🎯 Why This Project Matters

This project simulates a production-ready **end-to-end ML pipeline** tailored to environmental forecasting. It demonstrates skills across:
- Data engineering & feature design
- ML modeling & tuning
- Interpretability & validation
- Scalable deployment planning

Perfectly suited for roles in **ML Engineering**, **Data Science**, or **AI for Environmental Intelligence**.

---



Project 2: Modeling Competitive Performance in Monte Carlo Tree Search Variants (2024)

Project 3: Investigating photovoltaic solar power output forecasting using machine learning algorithms (2022)

Project 4: Predicting streamflow in Peninsular Malaysia using support vector machine and deep learning algorithms (2022)

Project 5: Predicting suspended sediment load in Peninsular Malaysia using support vector machine and deep learning algorithms (2022)
