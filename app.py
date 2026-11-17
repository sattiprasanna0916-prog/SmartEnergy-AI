
import json
import streamlit as st
import pandas as pd
import plotly.express as px

from config import DEFAULT_TARIFF, METRICS_PATH
from utils.database import initialize_database
from utils.auth import register_user, login_user
from utils.data_processor import load_dataset, prepare_daily_data
from utils.prediction import (
    load_best_model,
    predict_future,
    calculate_electricity_cost,
)
from utils.recommendations import generate_recommendations


st.set_page_config(
    page_title="SmartEnergy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# GLOBAL STYLING
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f5f6fb;
    color: #25243a;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #29245f;
}

p, li {
    color: #39384c;
}

section[data-testid="stSidebar"] {
    background: #302b70;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5e3f0;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(40,35,90,.05);
}

div[data-testid="stMetric"] label {
    color: #55536b !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #302b70 !important;
    font-weight: 700 !important;
}

div.stButton > button {
    border-radius: 11px;
    background: #6c5ce7;
    color: white;
    border: 0;
    padding: .65rem 1rem;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #5748ca;
    color: white;
}

div[data-testid="stPlotlyChart"],
div[data-testid="stDataFrame"] {
    background: white;
    border-radius: 16px;
    padding: 8px;
}

/* Login and registration label readability */
div[data-testid="stTextInput"] label,
div[data-testid="stTextInput"] label p {
    color: #24233b !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

div[data-testid="stTextInput"] input {
    color: #22223a !important;
    background: #ffffff !important;
    border: 1px solid #b9b6cc !important;
    border-radius: 9px !important;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #77758a !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #393650 !important;
    font-weight: 700 !important;
}

/* Select slider and other controls */
div[data-testid="stSelectSlider"] label,
div[data-testid="stSelectSlider"] label p {
    color: #29245f !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

div[data-testid="stSelectSlider"] [data-testid="stMarkdownContainer"] p {
    color: #29245f !important;
    font-weight: 600 !important;
}

div[data-testid="stSelectSlider"] div[role="slider"] {
    border-color: #6c5ce7 !important;
}

/* Compact statistics table */
.stats-wrapper {
    max-width: 520px;
    margin: 0 auto;
}

/* Information cards */
.info-card {
    background: white;
    border: 1px solid #e8e6f1;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}

.info-card h4 {
    color: #302b70;
    margin-bottom: 8px;
}

.info-card p {
    color: #55536b;
    margin-bottom: 0;
}

.auth-card {
    background: white;
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #e5e3f0;
    box-shadow: 0 8px 25px rgba(40,35,90,.05);
}

.insight-card {
    background: white;
    border-left: 5px solid #6c5ce7;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.insight-card p {
    margin: 0;
    color: #39384c;
}

.small-muted {
    color: #66647a;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data(show_spinner="Loading household electricity data...")
def get_data():
    raw = load_dataset()
    daily = prepare_daily_data(raw)

    if daily.empty:
        raise ValueError("The dataset does not contain valid daily consumption records.")

    return raw, daily


# --------------------------------------------------
# LOGIN AND REGISTRATION
# --------------------------------------------------

def login_screen():

    st.markdown(
        "<h1 style='color:#302b70'>⚡ SmartEnergy AI</h1>"
        "<p style='color:#55536b;font-size:17px'>"
        "Intelligent Household Energy Consumption Prediction and Analysis"
        "</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])

    with left:

        st.markdown('<div class="auth-card">', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Login", "Create Account"])

        with tab1:

            st.subheader("Welcome Back")

            email = st.text_input(
                "Email Address",
                placeholder="Enter your email address",
                key="login_email",
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password",
            )

            if st.button("Login", width='stretch'):

                try:
                    if not email.strip() or not password:
                        st.error("Please enter your email and password.")
                    else:
                        ok, user, message = login_user(email, password)

                        if ok:
                            st.session_state.user = user
                            st.session_state.logged_in = True
                            st.rerun()
                        else:
                            st.error(message)

                except Exception as exc:
                    st.error(f"Login failed: {exc}")

        with tab2:

            st.subheader("Create Your Account")

            name = st.text_input(
                "Full Name",
                placeholder="Enter your full name",
                key="reg_name",
            )

            email2 = st.text_input(
                "Email Address",
                placeholder="Enter your email address",
                key="reg_email",
            )

            pass2 = st.text_input(
                "Password",
                type="password",
                placeholder="Minimum 8 characters",
                key="reg_password",
            )

            if st.button("Create Account", use_container_width=True):

                try:
                    ok, message = register_user(name, email2, pass2)

                    if ok:
                        st.success(message)
                    else:
                        st.error(message)

                except Exception as exc:
                    st.error(f"Registration failed: {exc}")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:

        st.image(
            "https://images.unsplash.com/photo-1558002038-1055907df827?w=1000",
            width='stretch',
        )

        st.markdown("### Understand Your Home's Energy")

        st.write(
            "Analyze historical electricity consumption, "
            "estimate electricity costs, understand usage patterns "
            "and forecast future household energy consumption."
        )


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

try:
    initialize_database()

except Exception as exc:
    st.error(
        "Cannot connect to MySQL. Check your .env settings "
        "and make sure MySQL is running."
    )
    st.code(str(exc))
    st.stop()


# --------------------------------------------------
# SESSION MANAGEMENT
# --------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_screen()
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

user = st.session_state.user

with st.sidebar:

    st.markdown("## ⚡ SmartEnergy AI")

    st.caption(f"Welcome, {user['name']}")

    page = st.radio(
        "MENU",
        [
            "Dashboard",
            "Consumption Analysis",
            "Energy Prediction",
            "Energy Insights",
            "Model Performance",
            "About",
        ],
    )

    tariff = st.number_input(
        "Electricity tariff (₹/kWh)",
        min_value=0.1,
        value=float(DEFAULT_TARIFF),
        step=0.5,
    )

    st.divider()

    if st.button("Logout", width='stretch'):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

try:
    raw_data, daily_data = get_data()

except Exception as exc:
    st.error(f"Dataset loading error: {exc}")
    st.stop()


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

try:
    model = load_best_model()
    model_error = None

except Exception as exc:
    model = None
    model_error = str(exc)


# --------------------------------------------------
# COMMON CALCULATIONS
# --------------------------------------------------

daily_data = daily_data.sort_values("date").copy()

monthly = (
    daily_data
    .set_index("date")["energy_kwh"]
    .resample("ME")
    .sum()
    .reset_index()
)

monthly["month"] = monthly["date"].dt.strftime("%b %Y")

avg = float(daily_data["energy_kwh"].mean())
total = float(daily_data["energy_kwh"].sum())
peak = float(daily_data["energy_kwh"].max())

peak_date = daily_data.loc[
    daily_data["energy_kwh"].idxmax(), "date"
]


# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

def metric_row():

    a, b, c, d = st.columns(4)

    a.metric(
        "Total Consumption",
        f"{total:,.2f} kWh",
    )

    b.metric(
        "Average Daily Usage",
        f"{avg:.2f} kWh",
    )

    c.metric(
        "Peak Daily Usage",
        f"{peak:.2f} kWh",
    )

    d.metric(
        "Estimated Historical Bill",
        f"₹{calculate_electricity_cost(total, tariff):,.2f}",
    )

    descriptions = st.columns(4)

    descriptions[0].caption(
        "Total energy consumed across the available dataset."
    )

    descriptions[1].caption(
        "Average energy consumed per recorded day."
    )

    descriptions[2].caption(
        f"Highest recorded daily usage. Date: {peak_date:%d %b %Y}."
    )

    descriptions[3].caption(
        "Estimated cost using the selected electricity tariff."
    )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.title("Good Energy Starts With Good Insights")

    st.write(
        "Your household electricity consumption overview."
    )

    metric_row()

    st.markdown("### Energy Consumption Trends")

    chart_col1, chart_col2 = st.columns([1.3, 1])

    with chart_col1:

        if not monthly.empty:

            fig = px.bar(
                monthly.tail(12),
                x="month",
                y="energy_kwh",
                title="Monthly Energy Consumption",
                color_discrete_sequence=["#8a78f8"],
                labels={
                    "month": "Month",
                    "energy_kwh": "Energy (kWh)",
                },
            )

            fig.update_layout(
                template="plotly_white",
                margin=dict(l=10, r=10, t=55, b=10),
                xaxis_title="",
                yaxis_title="Energy (kWh)",
            )

            st.plotly_chart(
                fig,
                width='stretch',
                key="dashboard_monthly_chart",
            )

    with chart_col2:

        recent = daily_data.tail(30)

        fig = px.line(
            recent,
            x="date",
            y="energy_kwh",
            title="Daily Usage — Latest 30 Records",
            markers=True,
            color_discrete_sequence=["#7968ed"],
            labels={
                "date": "Date",
                "energy_kwh": "Energy (kWh)",
            },
        )

        fig.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_title="Date",
            yaxis_title="Energy (kWh)",
        )

        st.plotly_chart(
            fig,
            width='stretch',
            key="dashboard_daily_chart",
        )

    st.markdown("### Home Energy Snapshot")

    x, y = st.columns(2)

    x.metric(
        "Average Daily Electricity Cost",
        f"₹{avg * tariff:,.2f}",
    )

    y.metric(
        "Estimated 10% Daily Savings",
        f"₹{avg * tariff * 0.10:,.2f}",
    )

    st.caption(
        "Charts and metrics are calculated from the loaded historical "
        "household electricity dataset. Changing the source dataset "
        "changes these values."
    )


# --------------------------------------------------
# CONSUMPTION ANALYSIS
# --------------------------------------------------

elif page == "Consumption Analysis":

    st.title("Consumption Analytics")

    st.write(
        "Explore household electricity usage patterns and daily consumption statistics."
    )

    c1, c2 = st.columns(2)

    with c1:

        fig = px.histogram(
            daily_data,
            x="energy_kwh",
            nbins=25,
            title="Daily Consumption Distribution",
            color_discrete_sequence=["#8a78f8"],
            labels={"energy_kwh": "Energy (kWh)"},
        )

        fig.update_layout(template="plotly_white")

        st.plotly_chart(
            fig,
            width='stretch',
        )

    with c2:

        weekday = daily_data.copy()

        weekday["day_name"] = weekday["date"].dt.day_name()

        weekday = (
            weekday.groupby("day_name", as_index=False)["energy_kwh"]
            .mean()
        )

        order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        weekday["day_name"] = pd.Categorical(
            weekday["day_name"],
            categories=order,
            ordered=True,
        )

        weekday = weekday.sort_values("day_name")

        fig = px.bar(
            weekday,
            x="day_name",
            y="energy_kwh",
            title="Average Usage by Weekday",
            color_discrete_sequence=["#8a78f8"],
            labels={
                "day_name": "Day",
                "energy_kwh": "Average Energy (kWh)",
            },
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="",
            yaxis_title="Average Energy (kWh)",
        )

        st.plotly_chart(
            fig,
            width='stretch',
        )

    st.subheader("Consumption Statistics")

    stats = (
        daily_data["energy_kwh"]
        .describe()
        .rename({
            "count": "Number of days",
            "mean": "Average consumption",
            "std": "Standard deviation",
            "min": "Minimum consumption",
            "25%": "25th percentile",
            "50%": "Median consumption",
            "75%": "75th percentile",
            "max": "Maximum consumption",
        })
        .round(2)
    )

    stats_df = stats.to_frame("Energy (kWh)")

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.dataframe(
            stats_df,
            width='stretch',
            height=310,
        )

    st.caption(
        "Statistics are calculated from daily household energy consumption."
    )


# --------------------------------------------------
# ENERGY PREDICTION
# --------------------------------------------------

elif page == "Energy Prediction":

    st.title("Future Energy Consumption Prediction")

    st.write(
        "Select a forecast period to estimate future household electricity consumption."
    )

    if model is None:

        st.warning(
            "Trained model file not found or could not be loaded."
        )

        st.caption(
            "Expected location: models/best_model.pkl"
        )

        if model_error:
            with st.expander("Model loading details"):
                st.code(model_error)

    else:

        st.markdown("### Select Forecast Period")

        days = st.select_slider(
            "Forecast Period (Days)",
            options=[7, 14, 30],
            value=7,
            help="Choose how many future days you want the model to predict.",
        )

        st.caption(
            f"The model will generate predictions for the next {days} days."
        )

        if st.button(
            "Generate Forecast",
            width='stretch',
        ):

            try:

                future = predict_future(
                    model,
                    daily_data["date"].max(),
                    days,
                    tariff,
                )

                total_pred = float(
                    future["predicted_energy_kwh"].sum()
                )

                estimated_bill = total_pred * tariff

                a, b, c = st.columns(3)

                a.metric(
                    "Predicted Consumption",
                    f"{total_pred:.2f} kWh",
                )

                b.metric(
                    "Estimated Electricity Bill",
                    f"₹{estimated_bill:,.2f}",
                )

                c.metric(
                    "Forecast Period",
                    f"{days} Days",
                )

                fig = px.area(
                    future,
                    x="date",
                    y="predicted_energy_kwh",
                    title="Future Energy Consumption Forecast",
                    markers=True,
                    color_discrete_sequence=["#7968ed"],
                    labels={
                        "date": "Date",
                        "predicted_energy_kwh": "Predicted Energy (kWh)",
                    },
                )

                fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Date",
                    yaxis_title="Predicted Energy (kWh)",
                )

                st.plotly_chart(
                    fig,
                    width='stretch',
                )

                st.subheader("Daily Forecast Details")

                st.dataframe(
                    future.round(3),
                    width='stretch',
                )

                st.download_button(
                    "Download Forecast CSV",
                    future.to_csv(index=False).encode("utf-8"),
                    "smartenergy_forecast.csv",
                    "text/csv",
                )

            except Exception as exc:
                st.error(f"Forecast generation failed: {exc}")


# --------------------------------------------------
# ENERGY INSIGHTS
# --------------------------------------------------

elif page == "Energy Insights":

    st.title("Personalized Energy Insights")

    st.write(
        "Recommendations generated using the actual household consumption records."
    )

    insights = generate_recommendations(
        daily_data,
        tariff,
    )

    a, b, c = st.columns(3)

    a.metric(
        "Average Daily Usage",
        f"{insights['average_daily_usage']:.2f} kWh",
    )

    b.metric(
        "Maximum Daily Usage",
        f"{insights['maximum_daily_usage']:.2f} kWh",
    )

    c.metric(
        "Potential Daily Savings",
        f"₹{insights['estimated_daily_savings']:.2f}",
    )

    st.caption(
        "Potential savings are illustrative estimates, not guaranteed bill reductions."
    )

    st.markdown("### Your Usage Analysis")

    st.info(insights["usage_summary"])

    st.markdown("### Highest Consumption Days")

    high_days = insights["high_usage_days"]

    if high_days:

        high_df = pd.DataFrame(high_days)

        high_df["date"] = pd.to_datetime(
            high_df["date"]
        ).dt.strftime("%d %b %Y")

        st.dataframe(
            high_df,
             width="stretch",
        )

    else:

        st.write("No high-consumption days were identified.")

    st.markdown("### Recommended Actions")

    for index, tip in enumerate(
        insights["recommendations"],
        start=1,
    ):

        st.markdown(
            f"""
            <div class="insight-card">
                <p><b>{index}. {tip}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("Machine Learning Model Performance")

    if METRICS_PATH.exists():

        try:

            metrics = json.loads(
                METRICS_PATH.read_text(encoding="utf-8")
            )

            results = pd.DataFrame(
                metrics["results"]
            ).T

            st.subheader("Model Evaluation Results")

            st.dataframe(
                results.round(4),
                width='stretch',
            )

            best_model = metrics.get(
                "best_model",
                "Not specified",
            )

            st.success(
                f"Best performing model: {best_model}"
            )

            if "RMSE" in results.columns:

                chart = (
                    results.reset_index()
                    .rename(columns={"index": "Model"})
                )

                fig = px.bar(
                    chart,
                    x="Model",
                    y="RMSE",
                    title="RMSE Comparison — Lower is Better",
                    color_discrete_sequence=["#7968ed"],
                )

                fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Model",
                    yaxis_title="RMSE",
                )

                st.plotly_chart(
                    fig,
                    width='stretch',
                )

        except Exception as exc:

            st.error(
                f"Unable to read model metrics: {exc}"
            )

    else:

        st.warning(
            "Model metrics file not found. "
            "Expected location: models/model_metrics.json"
        )


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

elif page == "About":

    st.title("About SmartEnergy AI")

    st.write(
        "SmartEnergy AI is an intelligent household electricity "
        "consumption analysis and prediction application. "
        "It uses historical energy consumption data and machine "
        "learning techniques to analyze usage patterns, estimate "
        "electricity costs and forecast future energy requirements."
    )

    st.divider()

    st.subheader("Project Objective")

    st.markdown("""
    - Analyze historical household electricity consumption.
    - Identify daily and monthly electricity usage patterns.
    - Estimate electricity costs using a configurable tariff.
    - Predict future household energy consumption using machine learning.
    - Compare machine learning models using regression evaluation metrics.
    - Generate energy-saving recommendations based on consumption patterns.
    """)

    st.divider()

    st.subheader("Dataset Information")

    st.markdown("""
    **Dataset:** Individual Household Electric Power Consumption

    **Source:** UCI Machine Learning Repository

    **Data characteristics:**
    - Contains household electricity consumption measurements.
    - Includes date and time information.
    - Contains active power, reactive power, voltage and current-related measurements.
    - Includes three sub-metering measurements.
    - Minute-level records are aggregated into daily energy consumption for analysis.
    """)

    st.divider()

    st.subheader("Machine Learning Model Details")

    st.markdown("""
    **Problem Type:** Regression

    **Models used in the project:**
    - Random Forest Regressor
    - Linear Regression

    **Prediction target:** Daily household energy consumption (kWh).

    **Input features used by the prediction pipeline:**
    - Day
    - Month
    - Day of week
    - Day of year

    **Evaluation metrics:**
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R-squared (R²)
    """)

    st.divider()

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Plotly
    - Joblib
    - MySQL
    """)

    st.divider()

    st.subheader("Project Limitations")

    st.markdown("""
    - Predictions depend on the quality and coverage of historical data.
    - The current model predicts household-level daily consumption.
    - Actual appliance-level monitoring is not available.
    - Real-time IoT integration is not implemented.
    - Electricity bill estimates depend on the tariff entered by the user.
    - Forecasts are estimates and may differ from actual future consumption.
    """)

    st.divider()

    st.subheader("Application Features")

    st.markdown("""
    - User registration and login
    - Household consumption dashboard
    - Daily and monthly consumption analysis
    - Future energy consumption prediction
    - Electricity bill estimation
    - Data-driven energy insights
    - Machine learning model performance evaluation
    """)

    st.info(
        "SmartEnergy AI is an academic machine learning project "
        "for household electricity consumption analysis and prediction."
    )
