# 🌟 Life Cost: AI-Powered Cost of Living Analytics for Madhya Pradesh

Welcome to **Life Cost**, an intelligent, AI-driven platform that provides deep insights into the cost of living across Madhya Pradesh (with a strong focus on Bhopal). This document is designed to **teach you everything about the project**, including its architecture, the embedded Machine Learning algorithms, and how the data is processed.

## 🚀 What is this project?
Life Cost is a full-stack platform (FastAPI backend + HTML/JS/CSS frontend) that answers crucial questions for anyone planning to move or stay in Madhya Pradesh:
- What will be my rent?
- How much will groceries cost?
- What are the metro fares?
- How fast is inflation rising (CPI)?
- Which locality is best for my salary and lifestyle?

It uses **5 core Machine Learning and Data models**, served through a clean REST API.

---

## 🧠 The Machine Learning Algorithms: A Deep Dive

This project isn't just a simple database lookup. It uses predictive modeling. Here is a detailed breakdown of each model used, how it works, and *why* it was chosen.

### 1. 🏠 Rent Price Predictor
**Algorithm:** `RandomForestRegressor` (Ensemble Learning) with `XGBoost` & `GradientBoosting` evaluations.
**File:** `ml/rent_model.py`

*   **What it does**: Predicts the exact monthly rent in ₹ for a residential property based on its features.
*   **The Features (Inputs)**: Size in `sqft`, number of `bhk`, number of `bathrooms`, and the `city` (One-Hot Encoded).
*   **How it works**:
    *   **Decision Trees**: A Decision Tree splits data by asking a series of True/False questions (e.g., "Is `bhk` > 2?", "Is `city` == BHOPAL?").
    *   **Random Forest**: Instead of relying on one tree (which can overfit), it builds a "forest" of usually 300 different Decision Trees, each trained on a random subset of data. The final rent prediction is the average (mean) of all 300 trees. This makes the model highly robust and accurate.
*   **Why this algorithm?** Real estate prices are non-linear (e.g., the jump from 1BHK to 2BHK is different from 3BHK to 4BHK). Random Forest captures these hidden non-linear patterns effortlessly without requiring complex feature engineering.

### 2. 📈 CPI Inflation Trend Forecaster
**Algorithm:** `ARIMA(1, 1, 1)` (Time Series) with a fallback to `LinearRegression` on lag features.
**File:** `ml/cpi_model.py`

*   **What it does**: Forecasts the Consumer Price Index (CPI) for the next 6 months to predict inflation trends in Bhopal.
*   **How it works (ARIMA)**:
    *   **ARIMA** stands for AutoRegressive Integrated Moving Average.
    *   **AR (AutoRegressive)**: Uses the past values to predict the future (e.g., CPI of last month influences this month).
    *   **I (Integrated)**: Differencing the data to make the trend stable (subtracting today's value from yesterday's).
    *   **MA (Moving Average)**: Uses past forecast errors in a regression-like model.
*   **How it works (Linear Fallback)**: If ARIMA fails, the system switches to a `Linear Regression` model using "Lag features." It creates features like `lag_1` (last month's CPI) and `lag_3` (CPI 3 months ago) and fits a straight line: `y = mx + c` to predict the next point.

### 3. 🚇 Metro Fare Classifier
**Algorithm:** `DecisionTreeClassifier`
**File:** `ml/metro_model.py`

*   **What it does**: Classifies a Bhopal Metro route into fare tiers (₹10, ₹20, or ₹30).
*   **The Features**: `distance_km`, `num_stops`, `travel_min`.
*   **How it works**:
    *   Since the fare system usually follows logical rules set by the government (e.g., base fare for up to 3 stops), a `DecisionTreeClassifier` is perfect.
    *   It mathematically discovers the exact rules. For example, it might learn: "IF `num_stops` <= 3 THEN Fare is ₹10. ELSE IF `travel_min` <= 11 THEN Fare is ₹20. ELSE Fare is ₹30."
*   **Why this algorithm?** It is 100% interpretable. We can literally print out the `IF/ELSE` tree it generates and understand exactly how it makes decisions.

### 4. 🗺️ Locality Recommender
**Algorithm:** Weighted Composite Scoring System with Haversine Distance
**File:** `ml/locality_recommender.py`

*   **What it does**: Recommends the top 5 areas to live in Bhopal based on your salary, workplace, and lifestyle.
*   **How it works**: It scores every locality dynamically out of 100% using a custom mathematical formula:
    *   **Budget Fit (40% Weight)**: It calculates `abs(median_rent - ideal_rent)` where `ideal_rent` is exactly 25% of your `monthly_salary`.
    *   **Commute Score (30% Weight)**: Uses the **Haversine Formula**, which is a trigonometric calculation that finds the shortest distance between two points on a sphere (the Earth), using Latitude and Longitude. It calculates distance from the locality to your workplace.
    *   **Listing Availability (15%)**: More listings mean it's easier to find a house.
    *   **Lifestyle Match (15%)**: Checks if the locality matches your tags (e.g., Premium, Family, Value, Affordable).
*   **Result**: It ranks all combinations and returns the top 5 localized recommendations.

### 5. 🛒 Grocery Model
**Algorithm:** Aggregation & Statistical Averaging
**File:** `ml/grocery_model.py`

*   **What it does**: Computes the daily and monthly grocery cost.
*   **How it works**: Uses raw data scraped from Blinkit (quick commerce). It takes a standard basket (milk, bread, eggs, veggies, fruits) and computes averages while factoring in delivery premiums.

### 6. 📊 Overall Cost-of-Living Composite Index
**Algorithm:** Weighted Normalization Benchmark
**File:** `ml/composite_index.py` & `api/main.py`

*   **How it works**: Combines Rent (50%), Groceries (30%), Transport (5%), and CPI adjustment (15%). It normalizes all current values by dividing them by a **Baseline (January 2021 Bhopal)**. If the composite index is 115, it means the Cost of Living is 15% higher than in Jan 2021.

---

## 🛠️ Tech Stack & Architecture

The application has been upgraded with a secondary, high-performance Express/NodeJS rendering backend alongside the predictive Python ML backend:

### 1. ⚙️ Express & EJS Rendering Web App
*   **Web Framework:** `Express.js` (Node.js) serving server-rendered dynamic templates.
*   **Template Engine:** `EJS` with `ejs-mate` layout layout system for modular blocks.
*   **Input Validation:** `Joi` schema validation protecting all search and input fields.
*   **Database Integration:** `mysql2/promise` with automatic lazy-loading mock fallback. If MySQL is unreachable, it seamlessly parses, normalizes, and filters the raw scraped JSON data files dynamically.

### 2. 🤖 Python ML & Predictive API
*   **Predictive Backend:** `FastAPI` (Python) serving REST endpoints for models.
*   **Machine Learning:** `scikit-learn` (Trees & Regression), `statsmodels` (ARIMA), `numpy`, `pandas`.
*   **Data Persistence:** Trained models serialized via `.joblib` binary formats.

### 3. 🌐 Frontend & Devops
*   **Assets & Visuals:** `Chart.js` for inflation charts, `Leaflet.js` for interactive geographical maps.
*   **Containerization:** Docker multi-container setups using `Dockerfile.api`, `Dockerfile.frontend`, and `docker-compose.yml`.

---

## 💻 How to Run the Project Locally

### Option 1: Express + EJS Web App (Node.js & MySQL)
1. **Install Dependencies:**
   ```bash
   npm install
   ```
2. **Setup Environment Variables:** Create a `.env` file in the root directory:
   ```env
   PORT=3000
   DB_HOST=127.0.0.1
   DB_USER=root
   DB_PASSWORD=yourpassword
   DB_DATABASE=lifecost
   ```
3. **Seed the Database:** Extract, parse, and import all the scraped listings (Rent, Metro fares, Blinkit groceries) into MySQL:
   ```bash
   npm run seed
   ```
   *Note: If no MySQL server is running, the application will automatically fall back to serving mock queries loaded from local JSON data files.*
4. **Start the App:**
   ```bash
   npm start
   ```
   Navigate to `http://localhost:3000` to interact with the Dashboard, Rent Search, Metro Route Fare Classifier, and Locality Recommendation Matcher.

### Option 2: Python FastAPI & ML Services
1. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the ML Models:** Generate the serialised model files:
   ```bash
   python ml/train_all.py
   ```
3. **Start the FastAPI Server:**
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
4. **Open Frontend Client:**
   Open `frontend/index.html` in your browser.

### Option 3: Full Docker Orchestration
Build and spin up the multi-container configuration:
```bash
docker-compose up --build
```
*   Frontend: `http://localhost:80`
*   API Docs: `http://localhost:8000/docs`

---

## 📚 Final Recap: Your Learning Takeaways
*   **Ensemble Models (`RandomForest`)** are excellent for tabular data with hidden nonlinear relationships (like Rent).
*   **Time Series (`ARIMA`)** shines when predicting future trends based purely on historical sequences (like Inflation).
*   **Decision Trees** are unbeatable when you need to extract readable rules from a dataset (like Metro Fares).
*   **Database Fallbacks** allow robust frontend operations by seamlessly switching from database schemas to client-side json processing logic when connections drop.

Happy Coding and Exploring Madhya Pradesh! 🌟

