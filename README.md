# 🌍 World Happiness Prediction Project

TripleTen Code Pudding for September 2025

A data science project to **analyze global happiness trends** and **predict Life Ladder scores (0–10)** from social, economic, and health indicators.  
This project demonstrates hands-on skills in **EDA, feature engineering, regression modeling, hyperparameter tuning, and interpretability**.

---

## 🚀 Project Overview

The aim of this project is to:

- Load and preprocess the **World Happiness Report 2005-2024** dataset
- Explore relationships between economic, social, and emotional indicators
- Train regression models (**Linear Regression, XGBoost**) to predict happiness
- Use **Optuna** for hyperparameter tuning
- Evaluate model performance with **MAE, RMSE, and R²**
- Provide a helper function to predict happiness for new country–year inputs
- Visualize global trends with maps and charts

---

## ❓ Why We Chose This Dataset

We chose the **World Happiness Report dataset** because:

- ✅ **Recent data** – includes happiness scores up to **2024**
- 🌍 **Global coverage** – ~150+ countries for broad comparisons
- 📊 **Rich indicators** – beyond the **Life Ladder** (target), it contains:
  - Log GDP per capita (economic prosperity)
  - Social support (community strength)
  - Healthy life expectancy at birth (health outcomes)
  - Freedom to make life choices
  - Generosity
  - Perceptions of corruption
  - Positive and Negative affect (emotional well-being)

Together, these indicators make it ideal for exploring global trends and building predictive models to understand what drives happiness worldwide.

---

## 📦 Features

- Cleaned & preprocessed dataset (150+ countries, 2005–2024)
- Train/test splits with **time-aware separation**
- Models:
  - **Linear Regression** (baseline, interpretable)
  - **XGBoost** (tree-based regressor with Optuna tuning)
- Visualizations:
  - Choropleth world map of happiness
  - Correlation heatmaps & trends
- **Prediction function** with SHAP-based explanation of top drivers
- Metrics: **MAE, RMSE, R²**

---

## 🧠 Technologies

- Python 3
- pandas, NumPy
- scikit-learn, XGBoost, Optuna
- Matplotlib, Seaborn
- SHAP (interpretability)
- Streamlit ( demo UI)

---

## 📂 Project Structure

```
WORLD_HAPPINESS_PREDICTION/
├── data/
│   └── World-happiness-report-updated_2024.csv
├── notebooks/
│   ├── Sohini shap.ipynb
│   └── ken_optuna.ipynb
├── app.py   # optional demo
├── requirements.txt
└── README.md
```
## 📊 Dataset

- **Source:** World Happiness Report (compiled indicators, 2005–2024)
- **Rows:** ~2,300 (country–year observations)
- **Target:**
  - `Life Ladder` — Happiness score (0–10)

**Features used:**
- Log GDP per capita
- Social support
- Healthy life expectancy at birth
- Freedom to make life choices
- Generosity
- Perceptions of corruption
- Positive affect
- Negative affect
- year, Country name

---


## 📈 Workflow

### 🔹 Data Preparation
- Handle missing values (drop or impute by year/region)
- Scale numerical features with **StandardScaler**
- Train/test split with **time-aware separation**

### 🔹 Exploratory Data Analysis (EDA)
- Correlation heatmaps
- Trends over time (2005–2024)
- Choropleth maps of happiness by country

### 🔹 Modeling
- **Linear Regression** (baseline)
- **XGBoost** (nonlinear, feature interactions)
- **Optuna** hyperparameter tuning

### 🔹 Evaluation
- Metrics: **MAE, RMSE, R²**
- Compare against **baseline predictor**
- Residual analysis across regions and years

### 🔹 Prediction Function
- **Input:** Python dict or DataFrame with features
- **Output:** Predicted Life Ladder + top drivers (via SHAP)

---

## 📊 Example Results

| Model              | MAE   |
|--------------------|-------|
| Linear Regression  | ~0.39 | 
| XGBoost (Optuna)   | ~0.32 | 

**Insights:**
- Social support and GDP per capita are top positive drivers
- Perceptions of corruption strongly suppress happiness
- Emotional factors (positive/negative affect) add extra explanatory power

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/world-happiness-project.git
cd world-happiness-project
```
2. Install dependencies
```
pip install -r requirements.txt
```
4. Run Jupyter Notebook
```
jupyter notebook notebooks/World_Happiness.ipynb
```
6. Run Streamlit App
```
streamlit run app/streamlit_app.py
```
## 📝 Notes

Happiness is influenced by many non-quantified factors (e.g., governance, culture, conflicts).

Predictions are based only on available indicators.

## Future improvements:

- Add inequality indices, governance measures

- Incorporate regional embeddings

- Time-series forecasting with lag features

## 👥 Project Team

This project was developed by:

- Priti Sagar

- Ken Klabnik

- Sohini Tomar

- Ryan Roberts

