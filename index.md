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
Rain-Net is an ongoing research collaboration with Sunway University that focuses on forecasting daily rainfall using machine learning. Due to confidentiality, only selected details are shared here.

The core challenge in this project is the **scarcity of data**: the dataset comes from an **undisclosed rainfall station in Malaysia** and contains only one feature — **daily rainfall amounts** over a few thousand rows. This makes it a **univariate time series forecasting problem**, and a particularly difficult one, as many days record near-zero rainfall, while some days spike past **300mm to over 400mm**.

Despite these limitations, the motivation is clear: **many regions in the world require reliable rainfall forecasting systems but lack sufficient data**. The goal of this study is to develop a machine learning framework that can still produce **useful forecasts** under such constraints, contributing toward scalable solutions in data-sparse environments.

### 📊 Data & Features
To overcome the limitations of a single-variable dataset, a **heavily feature-engineered approach** was taken. These engineered features enable models to extract more information from temporal patterns, rainfall intensity, and cyclicality:

- **Seasonality features (cyclical encoding):**
  - Month, day of year, and week of year (sine/cosine pairs)
- **Lag features:**
  - Rainfall amounts and intensities for the past 7 days
- **Moving averages and accumulated rainfall:**
  - 7, 14, and 30-day averages and sums
- **Variability features:**
  - 3, 7, and 30-day rolling standard deviation
- **Change rate features:**
  - Daily and multi-day rainfall change rates
- **Binary and spell indicators:**
  - Rain/no rain, heavy rain, as well as spells (e.g. dry, wet, extreme)

This level of feature engineering gives models more temporal context, simulating the kind of signal depth you'd expect in multivariate datasets.

### 🧠 Methods & Models
The project explored a range of forecasting models, beginning with **gradient boosting algorithms**, which are known to perform well on small datasets:

- **Models used:**
  - XGBoost, LightGBM, CatBoost, AdaBoost
  - Feedforward Neural Network (FNN)
  - Long Short-Term Memory (LSTM)
  - Transformer-based neural network (in progress)

Gradient boosting models produced the best performance overall. Deep learning models were also explored but struggled to generalize due to limited data.

Hyperparameter tuning was done using **Optuna** with **Bayesian Optimization**, which helps identify optimal settings faster by using previous results to guide the next trials. This is more efficient than grid or random search, especially when the search space is well-defined.

**Tweedie regression** was implemented as the loss function in some models. Tweedie is particularly effective for **zero-inflated data with positive continuous targets**, like rainfall, because it balances between Poisson and Gamma distributions.

To prevent negative rainfall predictions, model outputs below zero were clipped to zero.

### 📈 Results & Evaluation
The dataset was split using a **train-validation-test** approach:
- 70% training, 15% validation, 15% testing
- A **buffer of one month** was maintained between each set to prevent data leakage

**Evaluation metrics used:**
- **MAE (Mean Absolute Error):** Easy to interpret, measures average magnitude of errors
- **RMSE (Root Mean Square Error):** Penalizes larger errors more heavily
- **NSE (Nash-Sutcliffe Efficiency):** Measures how well the model predicts compared to the mean — a value above 0.5 is typically considered acceptable in hydrology

**Best test performance achieved so far:**
- MAE: **8.633 mm**
- RMSE: **14.908 mm**
- NSE: **0.133**

While the NSE is still low, this is expected in a problem with limited data and high imbalance. The next milestone is to improve this to **above 0.5**, indicating practical forecasting accuracy.

### 🛠️ Tools & Libraries
`Python`, `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `XGBoost`, `LightGBM`, `CatBoost`, `TensorFlow`, `Optuna`, `SHAP`, `Scikit-learn`, `PyTorch`

### 💡 Key Takeaways
- **Feature engineering** proved essential in transforming a sparse, univariate dataset into something more informative for machine learning.
- **Gradient boosting models** outperformed deep learning methods due to the data size.
- **SHAP (SHapley Additive Explanations)** was used to understand feature importance. This helped identify which features contributed most to prediction accuracy and guided decisions to drop less useful or noisy features.
- Despite low NSE scores, the model performed reasonably well on MAE and RMSE — suggesting that it's **closely approximating rainfall on average**, but still struggles with extremes and dry-day accuracy.

### 🔄 Ongoing Work
Current experiments include:
- **Log transforming rainfall values** before prediction to stabilize variance and reduce the impact of extreme values. This helps models focus on relative changes in rainfall, rather than being overwhelmed by a few large spikes.
- **Cascaded modeling approach**: introducing a **binary classifier** to predict whether rainfall will occur at all (rain vs. no rain), followed by a regressor to predict the actual amount for predicted rainy days. This setup can improve zero-day forecasting and reduce noise in the regression stage.

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
