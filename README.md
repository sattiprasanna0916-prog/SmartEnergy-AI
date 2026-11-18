# SmartEnergy AI

### Intelligent Household Energy Consumption Prediction and Analysis Using Machine Learning

## Project Overview
SmartEnergy AI is an intelligent energy analytics and forecasting web application designed to help households track, analyze, optimize, and forecast their electricity consumption. Powered by machine learning and interactive statistical tools, the platform provides actionable insights, appliance-level calculators, what-if simulators, and monthly budget targets.

## Core Features

1. **User Authentication & Isolation**
   - Secure account registration and login powered by bcrypt password hashing.
   - User-specific data persistence and session isolation.

2. **Household Energy Dashboard**
   - Live metrics: Total energy consumed, daily average usage, peak consumption date, and estimated historical bill.
   - Interactive monthly and 30-day daily consumption trends styled in a dark emerald green theme.

3. **Consumption Analysis**
   - Daily consumption distribution histograms.
   - Weekday demand pattern breakdown (Monday through Sunday).
   - Descriptive statistical summary table.

4. **Appliance-wise Energy Calculator (New)**
   - Calculate daily and monthly energy consumption (kWh) and costs for household appliances.
   - Support for custom appliance entries or popular presets (Air Conditioner, Refrigerator, Ceiling Fans, Water Geyser, LED Lighting, etc.).
   - Interactive Plotly horizontal bar chart and donut chart breakdown.
   - Detailed breakdown table with downloadable CSV report.

5. **Monthly Consumption Comparison (New)**
   - Month-over-month consumption and electricity bill comparison.
   - Automatic percentage increase/decrease calculations.
   - Identification of highest and lowest consumption months.
   - Interactive dual-axis trend chart (Energy kWh vs Estimated Bill ₹).
   - Direct side-by-side comparative analysis between any two selected months.
   - Robust handling of missing and partial months.

6. **Unusual Consumption Detection (Enhanced)**
   - Statistical anomaly detection using Tukey's Interquartile Range (IQR) method: $\text{Threshold} = Q_3 + 1.5 \times IQR$.
   - Interactive timeline highlighting abnormal consumption dates with distinct markers and threshold line.
   - Tabular view of high-demand dates with excess energy over average and ratio to mean.
   - Clear warning notices and practical energy-saving recommendations.
   - Scientifically documented non-alarmist approach (statistical demand spikes are not labeled as equipment faults).

7. **Monthly Energy Saving Goal (New)**
   - Establish monthly consumption targets in kWh for any selected month.
   - Real-time tracking of target achievement and remaining energy units.
   - Interactive Plotly gauge visualizer with color-coded safety zones.
   - Estimation of potential bill savings.
   - Persistent MySQL storage (`energy_goals` table) associated with the authenticated user.

8. **What-if Energy Simulator (New)**
   - Simulate the direct impact of altering appliance daily operating hours on monthly consumption and bills.
   - Compare original measured baseline against simulated scenarios.
   - Side-by-side metric comparison and Plotly grouped comparison chart.
   - Clear visual distinction between actual baseline data and simulated projections.

9. **Improved Energy Prediction (Enhanced)**
   - Flexible forecast horizons: 7-day (weekly) and 30-day (monthly) options.
   - Powered by the trained Random Forest regression model.
   - Daily and total predicted consumption with estimated bill calculation.
   - Interactive Plotly line chart with recent historical context continuity and mean reference line.
   - Downloadable forecast CSV.

10. **Machine Learning Model Performance**
    - Evaluation metrics: MAE, RMSE, and $R^2$ Score.
    - Benchmarking Random Forest against Linear Regression with RMSE comparison chart.

## Technology Stack
- **Python 3.12**
- **Streamlit** (Web application framework)
- **Scikit-learn** (Machine learning modeling & evaluation)
- **Pandas & NumPy** (Data processing and aggregation)
- **Plotly** (Interactive data visualization)
- **MySQL** (User authentication and goal persistence)
- **Bcrypt** (Secure credential hashing)
- **Joblib** (Model serialization)
- **Python-dotenv** (Environment configuration)

## Installation

1. Activate your virtual environment:
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your database settings in `.env` (copied from `.env.example` if needed):
   ```env
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_USER=root
   MYSQL_PASSWORD=YourPassword
   MYSQL_DATABASE=smartenergy_db
   DEFAULT_TARIFF=6.0
   ```

## Run Application

Start the Streamlit application:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## Run Automated Tests

To run the automated test suite covering all features:
```bash
python -m unittest tests/test_features.py -v
```

## Project Developer
Satti Prasanna Lakshmi
