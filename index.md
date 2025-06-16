---
title: "Yusuf's Data Science & Machine Learning Portfolio"
layout: default
---

---
## 👋 Introduction

Welcome, and thank you for stopping by my data science and machine learning portfolio! 

I'm Yusuf, a part-time analytics postgraduate student at Georgia Tech, former engineer at Arup, and an aspiring industry-based data science and machine learning professional. This space brings together projects I've worked on through academic research, collaborations with universities, and self-guided learning.

These projects reflect my experience working with real-world data, where I've applied data science and machine learning techniques like statistical analysis, regression modelling, and time series forecasting to uncover useful insights. While much of my work has been shaped by research-driven challenges, I've also taken the initiative to explore ideas independently and build out complete workflows from data preparation to model evaluation.

My background so far has been focused mostly on academic and research settings, but the skills I've developed are relevant to many areas where data plays a central role. I'm always looking to expand into new problem spaces and continue learning through hands-on work.

---

## 🔍 What You'll Find Here

- A set of machine learning and analytics projects, with a focus on time series forecasting and regression modeling  
- Projects built through both collaborative research and independent study  
- Clear, well-structured summaries that explain the thinking behind each project  
- Code and documentation that aim to be understandable for both technical and non-technical readers

Thanks again for visiting, and I hope you find something here that interests you. Whether you're a hiring manager, data team lead, or just curious, I'm glad you're here!

---

## 🗂️ Project Index

Click a project to jump to its section:

- [💧 Rain-Net: Daily Rainfall Forecasting (2025 - ongoing)](#rain-net-daily-rainfall-forecasting-2025---ongoing)
- [♟️ VersusAI: Monte Carlo Tree Search Variant Performance Prediction (2024)](#versusai-monte-carlo-tree-search-variant-performance-prediction-2024)
- [🔆 SolarCast: Photovoltaic Solar Power Prediction (2022)](#solarcast-photovoltaic-solar-power-prediction-2022)
- [🚣 FlowTrack: River Streamflow Forecasting (2022)](#flowtrack-river-streamflow-forecasting-2022)
- [⛰️ SediSense: Suspended Sediment Load Forecasting (2022)](#sedisense-suspended-sediment-load-forecasting-2022)

---

## Rain-Net: Daily Rainfall Forecasting (2025 - ongoing)

### 🔍 Overview
Rain-Net is an ongoing research collaboration with Sunway University focused on forecasting daily rainfall using machine learning. Due to confidentiality, only selected aspects of the project can be shared.

The data used comes from an undisclosed rainfall station in Malaysia. It is a univariate time series containing just one feature: daily rainfall values. The dataset is relatively small, with only a few thousand rows. This poses a major challenge, as reliable forecasting typically requires rich historical data. However, this is representative of many real-world locations where rainfall monitoring is needed but data is limited.

The goal of this project is to design a machine learning framework capable of producing useful and actionable rainfall forecasts, even under such constraints.

### 📊 Data & Features
To extract more predictive signal from the univariate dataset, an extensive set of engineered features was developed:

- Seasonality features (cyclical encodings):  
  Month, day of year, and week of year as sine and cosine transformations

- Lagged rainfall features:  
  Daily rainfall and rainfall intensity over the past 7 days

- Moving averages and accumulation:  
  7, 14, and 30-day moving averages and rainfall sums

- Variability indicators:  
  3, 7, and 30-day rolling standard deviations

- Rainfall change rate:  
  1-day, 3-day, and 7-day rate of change in rainfall

- Binary indicators and rainfall spell features:  
  Rainfall presence, heavy rain, extreme rain, wet spells, dry spells, and intensity-based spell durations

### 📊 Exploratory Data Analysis (EDA)
Exploratory data analysis was conducted to understand the characteristics of the dataset:

1. Statistical profiling:
   - Summary statistics and histograms to show distribution and skewness (most values near zero, with occasional extremes over 300–400mm)
   - Boxplots and violin plots to identify outliers and variability across time windows

2. Trend and seasonality decomposition:
   - Seasonal decomposition was used to separate trend, seasonality, and residual components

3. Temporal correlation:
   - ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots were generated to analyze how previous days’ rainfall impacts current day values

These analyses informed feature design and confirmed the presence of seasonality, autocorrelation, and non-stationarity in the data.

### 🧠 Methods & Models
A wide range of forecasting models were explored, with a focus on performance and interpretability given the small dataset:

- Gradient boosting models:
  - XGBoost, LightGBM, CatBoost, and AdaBoost

- Deep learning models:
  - Feedforward neural network (FNN)
  - LSTM network
  - Transformer-based network (currently in progress)

Gradient boosting models delivered the best performance. Deep learning approaches were less effective, likely due to limited data volume and sparsity in rainfall occurrence.

SHAP (SHapley Additive Explanations) was used for model interpretability, allowing analysis of feature importance and helping guide the reduction of noisy or redundant features.

Hyperparameter tuning was conducted using Optuna with Bayesian optimization. This method was chosen for its efficiency in narrowing down the best settings in fewer trials, especially once the hyperparameter space was defined.

Tweedie regression loss was implemented for tree-based models to better model the zero-inflated nature of rainfall data. The Tweedie distribution is particularly suitable when many observations are zero, and the rest are positive continuous values. To handle negative predictions, all model outputs below zero were clipped to zero.

### 📈 Results & Evaluation
The dataset was split using a train-validation-test approach:
- 70% training, 15% validation, 15% testing
- A one-month buffer was added between splits to avoid temporal leakage

Evaluation metrics included:

- MAE (Mean Absolute Error): measures average forecast error magnitude
- RMSE (Root Mean Square Error): penalizes larger errors more than MAE
- NSE (Nash-Sutcliffe Efficiency): compares model performance to the mean; scores above 0.5 are generally acceptable in hydrological contexts

Best results on the test set so far:
- MAE: 8.633 mm
- RMSE: 14.908 mm
- NSE: 0.133

While the NSE score remains low, this is expected given the data limitations and the sensitivity of NSE to extreme value forecasts. The primary aim moving forward is to improve zero-rainfall day classification and extreme event accuracy to raise the NSE closer to the 0.5 benchmark.

### 🛠️ Tools & Libraries
Python, Pandas, NumPy, Matplotlib, Seaborn, XGBoost, LightGBM, CatBoost, TensorFlow, Optuna, SHAP, Scikit-learn, PyTorch

### 💡 Key Takeaways
This project demonstrates how a sparse, univariate time series dataset can be made more useful through extensive feature engineering and thoughtful modeling. Gradient boosting models provided the most reliable forecasts, while SHAP helped highlight which features mattered most to predictions.

Despite challenges in forecasting extremes and dry days, the model shows promising performance on average rainfall prediction. The study lays the foundation for practical forecasting in data-scarce settings.

### 🔄 Ongoing Work
Work continues in several areas:
- Log-transforming rainfall data before prediction to stabilize variance and reduce the impact of extreme values. This can help models better capture small-scale variability without being skewed by outliers.
- Introducing a binary classification step before regression. The classifier will first predict whether rain will occur, followed by a regressor to estimate the amount. This two-stage approach can improve model performance on zero-rainfall days and reduce unnecessary noise in the regression stage.

---

## VersusAI: Monte Carlo Tree Search Variant Performance Prediction (2024)

### 🔍 Overview
*Coming soon...*

### 📊 Data & Features
*Coming soon...*

### 🧠 Methods & Models
*Coming soon...*

### 📈 Results & Evaluation
*Coming soon...*

### 🛠️ Tools & Libraries
*Coming soon...*

### 💡 Key Takeaways
*Coming soon...*

---

## SolarCast: Photovoltaic Solar Power Prediction (2022)

### 🔍 Overview
*Coming soon...*

### 📊 Data & Features
*Coming soon...*

### 🧠 Methods & Models
*Coming soon...*

### 📈 Results & Evaluation
*Coming soon...*

### 🛠️ Tools & Libraries
*Coming soon...*

### 💡 Key Takeaways
*Coming soon...*

---

## FlowTrack: River Streamflow Forecasting (2022)

### 🔍 Overview
*Coming soon...*

### 📊 Data & Features
*Coming soon...*

### 🧠 Methods & Models
*Coming soon...*

### 📈 Results & Evaluation
*Coming soon...*

### 🛠️ Tools & Libraries
*Coming soon...*

### 💡 Key Takeaways
*Coming soon...*

---

## SediSense: Suspended Sediment Load Forecasting (2022)

### 🔍 Overview
*Coming soon...*

### 📊 Data & Features
*Coming soon...*

### 🧠 Methods & Models
*Coming soon...*

### 📈 Results & Evaluation
*Coming soon...*

### 🛠️ Tools & Libraries
*Coming soon...*

### 💡 Key Takeaways
*Coming soon...*
