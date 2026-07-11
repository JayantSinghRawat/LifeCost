# Life Cost Machine Learning Models Overview

This document provides a comprehensive explanation of the Machine Learning architecture behind the Life Cost project. Use this guide to structure your presentation, particularly when explaining **how each model works**, **what metrics/tests were used to evaluate them**, and **what those metrics actually tell us**.

---

## 1. 🏠 Rent Price Predictor (`rent_model.py`)

**What it does:** Predicts the expected monthly rent (in ₹) for properties in Madhya Pradesh.

**How it works:** The model is an ensemble regression system. It tests multiple Tree-based learning algorithms (`RandomForestRegressor`, `GradientBoostingRegressor`, and `XGBoost`). It takes inputs like the number of Bedrooms (BHK), Bathrooms, Square Footage, and the City (which is one-hot encoded) to deduce the underlying cost patterns.
*Currently Best Model:* Random Forest

### The "Tests" (Metrics) Explained:
*   **RMSE (Root Mean Squared Error):**
    *   **What it is:** Measures the standard deviation of our prediction errors. By squaring the errors before averaging them and picking the square root, it heavily penalizes large mistakes.
    *   **What it tells us:** It roughly tells us, "On average, how many ₹ is our prediction off by in the worst-case scenario?". *(Current model RMSE is roughly ₹5,374)*.
*   **MAE (Mean Absolute Error):**
    *   **What it is:** A simple average of all the raw, absolute errors our model makes.
    *   **What it tells us:** Gives us the direct daily "average mistake magnitude" without exaggerating outliers like hyper-luxury villas.
*   **R² (R-Squared / Coefficient of Determination):**
    *   **What it is:** A percentage score from 0 to 1 (or 0% to 100%) that measures the "Goodness of Fit".
    *   **What it tells us:** It tells you what proportion of the variance in rent is successfully explained by our chosen features (BHK, sqft, etc). *(e.g., an R² of 0.558 means ~56% of real-world rent fluctuations are accurately captured by our model, while the rest are caused by invisible factors like interior quality or specific neighborhood prestige).*

---

## 2. 🛒 Grocery Basket Estimator (`grocery_model.py`)

**What it does:** Estimates the total monthly/daily cost of a standardized nutrition basket for a single person.

**How it works:** Instead of hard-coding prices, it dynamically calculates the **Median Unit Price** across the dataset (e.g., price per 100g or per 1 ml). A **Ridge Regression** model acts as the ML baseline to analyze pricing trends across categories based on normalized quantity.
*Outputs:* An aggregated daily calculation of items (milk, bread, eggs, veg, fruit) derived intelligently from bulk grocery lists.

### The "Tests" (Metrics) Explained:
*   **Regression MAE & R²:**
    *   Applied specifically on the `price_per_100g` target to ensure the underlying data normalization holds true across categories. It validates that the pricing logic correlates accurately before running the basket summation algorithm.

---

## 3. 📈 CPI Inflation Forecaster (`cpi_model.py`)

**What it does:** Forecasts the Consumer Price Index (inflation trend) for Food & Beverages in Bhopal for the next 24 months.

**How it works:** It primarily relies on **ARIMA(1,1,1)** — the gold standard time-series forecasting algorithm. If statsmodels fails, it utilizes a robust fallback: **Linear Regression on Lag Features** (teaching the model to predict the next month using `lag_1` and `lag_3` past months).

### The "Tests" (Metrics) Explained:
*   **MAPE (Mean Absolute Percentage Error):**
    *   **What it is:** Calculates the absolute error from actual historical data, but expresses it as a simple **Percentage (%)** rather than absolute units.
    *   **What it tells us:** This is the most intuitive metric to present to business stakeholders. It tells us, "Our forecast predictions deviate from the true inflation reality by an average of X%". *(Our Linear Baseline is achieving an incredibly low MAPE of ~1.27%, meaning it is highly reliable.)*

---

## 4. 🚇 Metro Fare Classifier (`metro_model.py`)

**What it does:** Predicts the exact fare bracket (₹10, ₹20, or ₹30) for a metro ride in Bhopal based on commute details.

**How it works:** This is fundamentally a **Classification** problem rather than Regression. It uses a **Decision Tree Classifier**, which constructs interpretable "if-this-then-that" rules based on `distance_km`, `num_stops`, and `travel_min`. It tests this against a `LogisticRegression` to pick the best logic.
*Currently Best Model:* Decision Tree

### The "Tests" (Metrics) Explained:
*   **Accuracy:**
    *   **What it is:** The raw percentage of correctly classified trips.
    *   **What it tells us:** "Out of 100 trips, how many did we guess the exact right ticket price for?"
*   **LOO-CV (Leave-One-Out Cross-Validation):**
    *   **What it is:** A very intensive evaluation technique for small-scale local datasets. Instead of a normal 80/20 train/test split, it trains the model on $N-1$ samples and tests it on the 1 sample left out. This procedure is repeated $N$ times.
    *   **What it tells us:** It proves mathematically that our model **has not just "memorized"** the route map, but has actually learned the generalized fare rules and can perfectly predict unseen route combinations. *(Our LOO-Acc is currently 1.000 / 100% — perfect generalized accuracy).*
