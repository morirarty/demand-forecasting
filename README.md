# 📦 Enterprise Inventory Demand Forecasting
### Optimizing Supply Chain Operations with Hybrid Time-Series Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-red?logo=xgboost)
![Statsmodels](https://img.shields.io/badge/Statsmodels-SARIMA-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

Inventory mismanagement remains one of the most expensive and persistent inefficiencies across global logistics operations. A 2023 IHL Group report estimated that inventory distortion — the combined financial damage of overstocking and stockouts — costs global retailers **$1.77 trillion annually**.

This project delivers a production-grade **Hybrid Demand Forecasting System** that combines classical statistical time-series modeling with modern machine learning to shift supply chain operations from reactive replenishment to **proactive, data-driven inventory planning**.

The system ingests historical transaction records, extracts temporal demand signals through advanced feature engineering, and outputs high-precision demand forecasts — enabling logistics teams to maintain optimal stock levels, prevent capital lock-in, and eliminate preventable stockout events.

---

## 🎯 Business Problem

Every logistics and distribution operation faces the same fundamental tension:

| Scenario | Root Cause | Business Consequence |
|:---------|:-----------|:--------------------|
| **Overstocking** | Demand overestimation | Capital tied in illiquid assets, inflated holding costs, obsolescence & expiration risk |
| **Understocking** | Demand underestimation | Stockout events, lost revenue, damaged customer relationships, emergency procurement premiums |
| **Reactive Planning** | No forecasting system | Decisions driven by intuition rather than data, perpetuating both problems above |

> **Objective:** Build a high-precision demand forecasting engine that predicts future inventory requirements with minimum error — enabling procurement teams to order the right quantity at the right time, every replenishment cycle.

---

## 📊 Business Impact & ROI

### Quantifiable Operational Benefits

| Impact Area | Mechanism | Expected Outcome |
|:------------|:----------|:----------------|
| **Holding Cost Reduction** | Accurate quantity forecasting eliminates excess buffer stock | 20–35% reduction in carrying costs |
| **Stockout Prevention** | Proactive replenishment triggers before depletion | Near-zero unplanned stockout events |
| **Waste & Obsolescence Mitigation** | Probability-based demand modeling prevents over-accumulation | Significant reduction in expired or written-off inventory |
| **Working Capital Efficiency** | Right-sized inventory frees locked capital | Released capital redirected to higher-return operations |
| **Procurement Cycle Optimization** | Forecast-driven order scheduling | Elimination of emergency procurement premium costs |

---

## 🛠️ Technical Methodology

### System Pipeline

```
Raw Transaction Data
        │
        ▼
┌───────────────────┐
│  Data Ingestion   │  → Load historical inventory records
│  & Preprocessing  │  → Handle missing values & outliers
│                   │  → Normalize supply chain shock events
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│    Feature        │  → Lag features (demand memory)
│    Engineering    │  → Rolling window statistics
│                   │  → Temporal & seasonal encodings
└────────┬──────────┘
         │
         ├─────────────────────┐
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│  SARIMA Model   │   │  XGBoost Model  │
│  Linear &       │   │  Non-linear &   │
│  Seasonal       │   │  Residual       │
│  Structures     │   │  Patterns       │
└────────┬────────┘   └────────┬────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │   Hybrid Ensemble   │
         │   Final Forecast    │
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Evaluation &       │
         │  Visualization      │
         │  RMSE · MAE · MAPE  │
         └─────────────────────┘
```

---

### 1. Data Preprocessing

- **Missing Value Treatment:** Forward-fill strategy for short gaps; interpolation for extended missing periods to preserve time-series continuity
- **Outlier Detection:** IQR-based anomaly flagging to identify and neutralize supply chain shock events (port delays, demand spikes) without distorting the underlying trend signal
- **Stationarity Testing:** Augmented Dickey-Fuller (ADF) test applied to confirm or induce stationarity prior to SARIMA fitting

---

### 2. Advanced Feature Engineering

Meaningful predictive features are extracted across three dimensions:

**Demand Memory Features (Lag Variables)**

| Feature | Formula | Signal Captured |
|:--------|:--------|:----------------|
| Lag-1 | $X_{t-1}$ | Previous period demand |
| Lag-7 | $X_{t-7}$ | Same day last week (weekly cycle) |
| Lag-30 | $X_{t-30}$ | Same period last month (monthly cycle) |

**Market Volatility Features (Rolling Windows)**

| Feature | Window | Signal Captured |
|:--------|:------:|:----------------|
| Rolling Mean | 7-day | Short-term demand trend |
| Rolling Mean | 30-day | Long-term demand baseline |
| Rolling Std Dev | 7-day | Demand volatility & uncertainty |
| Rolling Std Dev | 30-day | Seasonal demand spread |

**Temporal & Cyclical Features**

| Feature | Encoding | Purpose |
|:--------|:--------|:--------|
| Day of Week | Integer (0–6) | Weekly demand cycles |
| Month | Integer (1–12) | Monthly & quarterly cycles |
| Quarter | Integer (1–4) | Seasonal business cycles |
| Is Month-End | Boolean | Month-end demand spikes |
| Is Holiday | Boolean | Holiday demand anomalies |

---

### 3. Hybrid Predictive Modeling

This project employs a **two-component hybrid architecture** that leverages the complementary strengths of statistical and machine learning models:

**Component 1 — SARIMA (Seasonal ARIMA)**

SARIMA captures the **linear, structured, and seasonal components** of demand:

```
SARIMA(p, d, q)(P, D, Q)[s]

Where:
  p, d, q  → Non-seasonal AR, differencing, MA orders
  P, D, Q  → Seasonal AR, differencing, MA orders
  s        → Seasonal period (s=12 for monthly, s=52 for weekly)
```

- Handles autocorrelation structure via ACF/PACF analysis
- Models seasonal periodicity explicitly through the seasonal component
- Provides interpretable confidence intervals for uncertainty quantification

**Component 2 — XGBoost Regressor**

XGBoost maps **non-linear relationships and complex interaction effects** that SARIMA cannot capture:

- Learns from engineered lag and rolling features
- Captures demand driver interactions (e.g., promotions × seasonality)
- Gradient boosting minimizes residual error iteratively across 200–500 estimators
- Feature importance ranking provides explainability for business stakeholders

**Hybrid Ensemble Output**

```
Final Forecast = α · SARIMA_Forecast + (1 - α) · XGBoost_Forecast

Where α is determined by validation-set performance weighting
```

---

## 📈 Model Performance & Visualization

### Forecast Accuracy Results

| Metric | Description | Target Threshold |
|:-------|:------------|:----------------|
| **RMSE** | Root Mean Squared Error — penalizes large errors heavily | Lower is better |
| **MAE** | Mean Absolute Error — average absolute deviation from actual | Lower is better |
| **MAPE** | Mean Absolute Percentage Error — scale-independent accuracy | Below 15% = production-ready |

### Actual vs Predicted Visualization

<img width="1347" height="758" alt="Demand Forecast vs Actual" src="https://github.com/user-attachments/assets/8571880b-2e6d-4dfb-9c42-14c86a95ba5e" />

**Chart Interpretation:**

| Visual Element | Interpretation |
|:--------------|:--------------|
| **Actual Demand Line** | Ground-truth historical inventory consumption — includes natural volatility and seasonal spikes |
| **Predicted Demand Line** | Model output — tracks actual trend with high fidelity across both stable and volatile periods |
| **Alignment Quality** | High overlap between actual and predicted serves as empirical evidence of the model's ability to capture complex non-linear demand patterns through the hybrid feature architecture |

---

## 💻 Tech Stack

| Category | Library | Version | Purpose |
|:---------|:--------|:-------:|:--------|
| **Language** | Python | 3.10+ | Core runtime |
| **Data Manipulation** | Pandas | >= 1.5 | Data ingestion, cleaning, feature engineering |
| **Numerical Computing** | NumPy | >= 1.21 | Array operations & mathematical transforms |
| **Statistical Modeling** | Statsmodels | >= 0.13 | SARIMA fitting, ADF stationarity test |
| **Machine Learning** | Scikit-Learn | >= 1.2 | Preprocessing, cross-validation, evaluation |
| **Gradient Boosting** | XGBoost | >= 1.7 | Non-linear demand pattern learning |
| **Visualization** | Matplotlib | >= 3.6 | Forecast trajectory plots |
| **Visualization** | Seaborn | >= 0.12 | Statistical distribution charts |

---

## 📁 Repository Structure

```
enterprise-demand-forecasting/
│
├── notebooks/
│   └── demand_forecasting.ipynb     # Full analysis & modeling notebook
├── scripts/
│   ├── preprocessing.py             # Data cleaning & feature engineering
│   ├── sarima_model.py              # SARIMA fitting & forecasting
│   ├── xgboost_model.py             # XGBoost training & evaluation
│   └── hybrid_ensemble.py           # Ensemble combination logic
├── data/
│   └── .gitkeep                     # Data directory (not tracked)
├── outputs/
│   └── stochastic_inventory_simulation.png
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/[your-username]/enterprise-demand-forecasting.git

# 2. Navigate to project directory
cd enterprise-demand-forecasting

# 3. Install all dependencies
pip install -r requirements.txt
```

### Requirements File

```txt
pandas>=1.5.0
numpy>=1.21.0
statsmodels>=0.13.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

### Run the Forecasting Engine

```bash
# Run full pipeline
python scripts/hybrid_ensemble.py

# Or open the notebook
jupyter notebook notebooks/demand_forecasting.ipynb
```

---

## 💡 Strategic Recommendations

Based on forecasting system outputs, the following operational actions are recommended:

1. **Implement Forecast-Driven Reorder Points**
   Replace static safety stock thresholds with dynamic reorder points updated monthly from model predictions — directly reducing both stockout frequency and excess inventory accumulation.

2. **Segment SKUs by Forecast Confidence**
   Products where MAPE exceeds 20% should trigger additional manual review. High-confidence SKUs (MAPE below 10%) can be fully automated in the procurement workflow.

3. **Integrate Promotional Calendar**
   Feed planned promotional events and holidays as binary features into the XGBoost model to prevent systematic under-forecasting during demand spike periods.

4. **Retrain Monthly on Rolling Window**
   Maintain model accuracy by retraining on a 24-month rolling window every month — ensuring the model adapts to structural demand shifts without overfitting to distant historical patterns.

---

## 🔬 Limitations & Future Work

| Current Limitation | Proposed Enhancement |
|:-------------------|:--------------------|
| Single-product forecasting | Multi-SKU simultaneous forecasting pipeline |
| Static ensemble weighting | Dynamic weight optimization via Bayesian updating |
| No external demand drivers | Integrate macroeconomic indicators & promotional data |
| Point forecast output | Probabilistic forecast with confidence intervals |
| Offline batch processing | Real-time streaming forecast with Apache Kafka |

---

## 📚 References

- Box, G.E.P. & Jenkins, G.M. (1970). *Time Series Analysis: Forecasting and Control.* Holden-Day.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
- IHL Group (2023). *Inventory Distortion Study: Overstocks, Out-of-Stocks & Returns.*
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

---

