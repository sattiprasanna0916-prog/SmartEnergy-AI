import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from config import DEFAULT_TARIFF, METRICS_PATH
from utils.database import (
    initialize_database,
    save_user_goal,
    get_user_goal,
    get_user_goals,
)
from utils.auth import register_user, login_user
from utils.data_processor import (
    load_dataset,
    prepare_daily_data,
    prepare_monthly_data,
    detect_iqr_outliers,
)
from utils.prediction import (
    load_best_model,
    predict_future,
    calculate_electricity_cost,
)
from utils.recommendations import generate_recommendations
from utils.calculator import (
    APPLIANCE_PRESETS,
    calculate_single_appliance,
    calculate_appliances_summary,
    simulate_appliance_change,
)


st.set_page_config(
    page_title="SmartEnergy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# GLOBAL STYLING (DARK EMERALD GREEN THEME)
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f4fbf7;
    color: #1f2937;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1450px;
}

h1, h2, h3, h4 {
    color: #064e3b !important;
    font-weight: 700 !important;
}

p, li {
    color: #374151;
}

section[data-testid="stSidebar"] {
    background: #064e3b;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 6px 12px;
    margin-bottom: 4px;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.16);
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d1fae5;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(6, 78, 59, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(6, 78, 59, 0.10);
}

div[data-testid="stMetric"] label {
    color: #047857 !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #064e3b !important;
    font-weight: 700 !important;
}

div.stButton > button {
    border-radius: 10px;
    background: #059669;
    color: white;
    border: 0;
    padding: 0.65rem 1.2rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: #047857;
    color: white;
    box-shadow: 0 4px 12px rgba(5, 150, 105, 0.25);
}

div[data-testid="stPlotlyChart"],
div[data-testid="stDataFrame"] {
    background: white;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    padding: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
}

/* Form inputs */
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #064e3b !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    color: #1f2937 !important;
    background: #ffffff !important;
    border: 1px solid #a7f3d0 !important;
    border-radius: 9px !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #064e3b !important;
    font-weight: 700 !important;
}

.auth-card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #d1fae5;
    box-shadow: 0 8px 30px rgba(6, 78, 59, 0.07);
}

.insight-card {
    background: white;
    border-left: 5px solid #059669;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}

.insight-card p {
    margin: 0;
    color: #1f2937;
}

.info-card {
    background: white;
    border: 1px solid #d1fae5;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: 0 2px 10px rgba(6, 78, 59, 0.04);
}

.scenario-badge-actual {
    display: inline-block;
    background: #ecfdf5;
    color: #064e3b;
    border: 1px solid #a7f3d0;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.scenario-badge-simulated {
    display: inline-block;
    background: #f0fdf4;
    color: #059669;
    border: 1px solid #86efac;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.warning-callout {
    background: #fffbeb;
    border-left: 5px solid #f59e0b;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
    color: #92400e;
    font-size: 14px;
}

.success-callout {
    background: #ecfdf5;
    border-left: 5px solid #10b981;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
    color: #065f46;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Plotly theme emerald color palette
EMERALD_COLORS = ["#059669", "#10b981", "#34d399", "#6ee7b7", "#047857", "#065f46", "#a7f3d0"]


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
        "<h1 style='color:#064e3b'>⚡ SmartEnergy AI</h1>"
        "<p style='color:#047857;font-size:17px'>"
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

            if st.button("Login", width="stretch"):
                try:
                    if not email.strip() or not password:
                        st.error("Please enter your email and password.")
                    else:
                        ok, user_rec, message = login_user(email, password)
                        if ok:
                            st.session_state.user = user_rec
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

            if st.button("Create Account", width="stretch"):
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
            width="stretch",
        )
        st.markdown("### Understand Your Home's Energy")
        st.write(
            "Analyze historical electricity consumption, estimate electricity costs, "
            "track energy saving goals, calculate appliance loads, simulate usage scenarios, "
            "and forecast future household energy consumption using machine learning."
        )


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

try:
    initialize_database()
except Exception as exc:
    st.error(
        "Cannot connect to MySQL. Check your .env settings and make sure MySQL is running."
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

user = st.session_state.user


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ SmartEnergy AI")
    st.caption(f"Logged in as: **{user['name']}**")

    page = st.radio(
        "MENU",
        [
            "Dashboard",
            "Consumption Analysis",
            "Monthly Comparison",
            "Appliance Energy Calculator",
            "Energy Simulator",
            "Energy Prediction",
            "Energy Insights",
            "Energy Saving Goal",
            "Model Performance",
            "About",
        ],
    )

    tariff = st.number_input(
        "Electricity tariff (₹/kWh)",
        min_value=0.1,
        value=float(DEFAULT_TARIFF),
        step=0.5,
        help="Configurable tariff used to calculate electricity costs across all pages.",
    )

    st.divider()

    if st.button("Logout", width="stretch"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()


# --------------------------------------------------
# LOAD DATASET & TRAINED MODEL
# --------------------------------------------------

try:
    raw_data, daily_data = get_data()
except Exception as exc:
    st.error(f"Dataset loading error: {exc}")
    st.stop()

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
monthly_data = prepare_monthly_data(daily_data, tariff)

avg = float(daily_data["energy_kwh"].mean())
total = float(daily_data["energy_kwh"].sum())
peak = float(daily_data["energy_kwh"].max())
peak_date = daily_data.loc[daily_data["energy_kwh"].idxmax(), "date"]


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":
    st.title("Good Energy Starts With Good Insights")
    st.write("Your household electricity consumption overview and live telemetry snapshot.")

    a, b, c, d = st.columns(4)
    a.metric("Total Consumption", f"{total:,.2f} kWh")
    b.metric("Average Daily Usage", f"{avg:.2f} kWh")
    c.metric("Peak Daily Usage", f"{peak:.2f} kWh")
    d.metric("Estimated Historical Bill", f"₹{calculate_electricity_cost(total, tariff):,.2f}")

    desc = st.columns(4)
    desc[0].caption("Total energy consumed across the dataset.")
    desc[1].caption("Average energy consumed per recorded day.")
    desc[2].caption(f"Peak recorded day: {peak_date:%d %b %Y}.")
    desc[3].caption(f"Estimated bill at ₹{tariff:.2f}/kWh.")

    st.markdown("### Energy Consumption Trends")
    chart_col1, chart_col2 = st.columns([1.3, 1])

    with chart_col1:
        if not monthly_data.empty:
            recent_months = monthly_data.tail(12)
            fig = px.bar(
                recent_months,
                x="month_str",
                y="energy_kwh",
                title="Monthly Energy Consumption (Recent 12 Months)",
                color_discrete_sequence=["#059669"],
                labels={"month_str": "Month", "energy_kwh": "Energy (kWh)"},
            )
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=10, r=10, t=55, b=10),
                xaxis_title="",
                yaxis_title="Energy (kWh)",
            )
            st.plotly_chart(fig, width="stretch", key="dashboard_monthly_chart")

    with chart_col2:
        recent = daily_data.tail(30)
        fig = px.line(
            recent,
            x="date",
            y="energy_kwh",
            title="Daily Usage — Latest 30 Records",
            markers=True,
            color_discrete_sequence=["#10b981"],
            labels={"date": "Date", "energy_kwh": "Energy (kWh)"},
        )
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_title="Date",
            yaxis_title="Energy (kWh)",
        )
        st.plotly_chart(fig, width="stretch", key="dashboard_daily_chart")

    st.markdown("### Home Energy Snapshot")
    x, y = st.columns(2)
    x.metric("Average Daily Electricity Cost", f"₹{avg * tariff:,.2f}")
    y.metric("Estimated 10% Daily Savings", f"₹{avg * tariff * 0.10:,.2f}")
    st.caption("Metrics and charts are dynamically synchronized with the loaded dataset and active electricity tariff.")


# --------------------------------------------------
# CONSUMPTION ANALYSIS
# --------------------------------------------------

elif page == "Consumption Analysis":
    st.title("Consumption Analytics")
    st.write("Explore household electricity usage patterns, distributions, and weekday trends.")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            daily_data,
            x="energy_kwh",
            nbins=25,
            title="Daily Consumption Distribution",
            color_discrete_sequence=["#059669"],
            labels={"energy_kwh": "Energy (kWh)"},
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="stretch")

    with c2:
        weekday = daily_data.copy()
        weekday["day_name"] = weekday["date"].dt.day_name()
        weekday = (
            weekday.groupby("day_name", as_index=False)["energy_kwh"]
            .mean()
        )
        order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        weekday["day_name"] = pd.Categorical(
            weekday["day_name"], categories=order, ordered=True
        )
        weekday = weekday.sort_values("day_name")

        fig = px.bar(
            weekday,
            x="day_name",
            y="energy_kwh",
            title="Average Usage by Weekday",
            color_discrete_sequence=["#10b981"],
            labels={"day_name": "Day", "energy_kwh": "Average Energy (kWh)"},
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="",
            yaxis_title="Average Energy (kWh)",
        )
        st.plotly_chart(fig, width="stretch")

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
        st.dataframe(stats_df, width="stretch", height=320)
    st.caption("Descriptive statistics calculated across all recorded daily consumption data.")


# --------------------------------------------------
# FEATURE 1: APPLIANCE-WISE ENERGY CALCULATOR
# --------------------------------------------------

elif page == "Appliance Energy Calculator":
    st.title("Appliance Energy Calculator")
    st.write(
        "Estimate energy consumption and electricity costs for individual home appliances. "
        "Customize wattage, quantity, and daily operating hours to see their individual and total impact."
    )

    # Initialize session state for user's appliance list
    if "appliance_list" not in st.session_state:
        # Default realistic household preset
        st.session_state.appliance_list = [
            {
                "name": "Air Conditioner (1.5 Ton)",
                "category": "Cooling / Heating",
                "power_watts": 1500,
                "count": 1,
                "daily_hours": 6.0,
                "days_per_month": 30,
            },
            {
                "name": "Refrigerator (Frost Free)",
                "category": "Kitchen",
                "power_watts": 180,
                "count": 1,
                "daily_hours": 24.0,
                "days_per_month": 30,
            },
            {
                "name": "Ceiling Fan",
                "category": "Comfort",
                "power_watts": 75,
                "count": 3,
                "daily_hours": 12.0,
                "days_per_month": 30,
            },
            {
                "name": "LED Television (43-inch)",
                "category": "Entertainment",
                "power_watts": 100,
                "count": 1,
                "daily_hours": 4.0,
                "days_per_month": 30,
            },
            {
                "name": "Water Geyser",
                "category": "Bathroom",
                "power_watts": 2000,
                "count": 1,
                "daily_hours": 1.5,
                "days_per_month": 30,
            },
            {
                "name": "LED Light Bulbs",
                "category": "Lighting",
                "power_watts": 14,
                "count": 6,
                "daily_hours": 6.0,
                "days_per_month": 30,
            },
        ]

    # Quick Add or Custom Appliance Form
    with st.expander("➕ Add an Appliance to Your Household", expanded=False):
        form_col1, form_col2 = st.columns([1, 1])

        preset_names = ["-- Select from Presets --"] + [p["name"] for p in APPLIANCE_PRESETS]

        with form_col1:
            selected_preset = st.selectbox(
                "Choose from Popular Appliance Presets (Optional)",
                options=preset_names,
                key="calc_preset_select",
            )

            # Auto-fill defaults if preset chosen
            default_name = ""
            default_watts = 100.0
            default_count = 1
            default_hours = 4.0
            default_days = 30
            default_cat = "General"

            if selected_preset != "-- Select from Presets --":
                preset_data = next((p for p in APPLIANCE_PRESETS if p["name"] == selected_preset), None)
                if preset_data:
                    default_name = preset_data["name"]
                    default_watts = float(preset_data["power_watts"])
                    default_count = int(preset_data["default_count"])
                    default_hours = float(preset_data["default_hours"])
                    default_days = int(preset_data["default_days"])
                    default_cat = preset_data.get("category", "General")

            app_name = st.text_input(
                "Appliance Name",
                value=default_name if default_name else "Microwave Oven",
                placeholder="e.g. Inverter AC, Laptop, Induction Cooktop",
            )
            app_category = st.selectbox(
                "Category",
                ["Kitchen", "Cooling / Heating", "Lighting", "Entertainment", "Bathroom", "Laundry", "Work & Study", "General"],
                index=0,
            )

        with form_col2:
            app_watts = st.number_input(
                "Power Rating (Watts)",
                min_value=1.0,
                max_value=15000.0,
                value=default_watts,
                step=10.0,
                help="Appliance power specification in Watts (W).",
            )
            app_count = st.number_input(
                "Number of Units",
                min_value=1,
                max_value=50,
                value=default_count,
                step=1,
            )
            app_hours = st.slider(
                "Daily Usage (Hours/Day)",
                min_value=0.1,
                max_value=24.0,
                value=default_hours,
                step=0.5,
            )
            app_days = st.slider(
                "Days of Usage per Month",
                min_value=1,
                max_value=31,
                value=default_days,
                step=1,
            )

        if st.button("Add Appliance", width="stretch"):
            if not app_name.strip():
                st.error("Please provide an appliance name.")
            else:
                st.session_state.appliance_list.append({
                    "name": app_name.strip(),
                    "category": app_category,
                    "power_watts": float(app_watts),
                    "count": int(app_count),
                    "daily_hours": float(app_hours),
                    "days_per_month": int(app_days),
                })
                st.success(f"Added **{app_name}** to your appliances list!")
                st.rerun()

    # Calculations summary
    summary = calculate_appliances_summary(st.session_state.appliance_list, tariff)

    # Top Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Daily Energy", f"{summary['total_daily_kwh']:.2f} kWh")
    m2.metric("Total Monthly Energy", f"{summary['total_monthly_kwh']:.2f} kWh")
    m3.metric("Estimated Monthly Cost", f"₹{summary['total_monthly_cost']:,.2f}")
    m4.metric("Largest Energy Consumer", f"{summary['highest_appliance']}")

    desc_col = st.columns(4)
    desc_col[0].caption("Combined daily electricity requirement.")
    desc_col[1].caption("Projected monthly energy requirement.")
    desc_col[2].caption(f"Calculated using active tariff (₹{tariff:.2f}/kWh).")
    desc_col[3].caption(f"Consumes {summary['highest_kwh']:.2f} kWh/month.")

    # Charts
    if not summary["df"].empty:
        c_left, c_right = st.columns([1.3, 1])

        with c_left:
            fig_bar = px.bar(
                summary["df"].sort_values("Monthly Energy (kWh)", ascending=True),
                x="Monthly Energy (kWh)",
                y="Appliance",
                orientation="h",
                title="Appliance-Wise Monthly Consumption (kWh)",
                color="Monthly Energy (kWh)",
                color_continuous_scale=["#a7f3d0", "#10b981", "#059669", "#064e3b"],
                text="Monthly Energy (kWh)",
            )
            fig_bar.update_layout(
                template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=10),
                coloraxis_showscale=False,
                xaxis_title="Monthly Energy (kWh)",
                yaxis_title="",
            )
            fig_bar.update_traces(texttemplate="%{text:.1f} kWh", textposition="outside")
            st.plotly_chart(fig_bar, width="stretch")

        with c_right:
            fig_pie = px.pie(
                summary["df"],
                names="Appliance",
                values="Monthly Energy (kWh)",
                title="Share of Monthly Energy (%)",
                color_discrete_sequence=EMERALD_COLORS,
                hole=0.45,
            )
            fig_pie.update_layout(
                template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_pie, width="stretch")

        st.subheader("Detailed Appliance Energy Breakdown")
        st.dataframe(
            summary["df"],
            width="stretch",
            hide_index=True,
        )

        b_col1, b_col2 = st.columns([1, 1])
        with b_col1:
            st.download_button(
                "📥 Download Appliance Energy Breakdown (CSV)",
                summary["df"].to_csv(index=False).encode("utf-8"),
                "appliance_energy_breakdown.csv",
                "text/csv",
            )
        with b_col2:
            if st.button("Reset Appliances to Default", width="stretch"):
                st.session_state.pop("appliance_list", None)
                st.rerun()
    else:
        st.info("No appliances added yet. Click above to add appliances.")


# --------------------------------------------------
# FEATURE 2: MONTHLY CONSUMPTION COMPARISON
# --------------------------------------------------

elif page == "Monthly Comparison":
    st.title("Monthly Consumption Comparison")
    st.write(
        "Analyze monthly electricity consumption, compare estimated monthly bills, "
        "track month-over-month percentage changes, and identify peak consumption periods."
    )

    if monthly_data.empty:
        st.warning("Monthly data could not be computed from the loaded dataset.")
    else:
        # Highest and lowest consumption months
        highest_row = monthly_data.loc[monthly_data["energy_kwh"].idxmax()]
        lowest_row = monthly_data.loc[monthly_data["energy_kwh"].idxmin()]
        avg_monthly_kwh = float(monthly_data["energy_kwh"].mean())
        avg_monthly_bill = avg_monthly_kwh * tariff

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Highest Consumption Month", f"{highest_row['month_str']}", f"{highest_row['energy_kwh']:,.1f} kWh")
        k2.metric("Lowest Consumption Month", f"{lowest_row['month_str']}", f"{lowest_row['energy_kwh']:,.1f} kWh")
        k3.metric("Average Monthly Energy", f"{avg_monthly_kwh:,.1f} kWh")
        k4.metric("Average Monthly Bill", f"₹{avg_monthly_bill:,.2f}")

        st.markdown("### Interactive Monthly Trend Analysis")

        # Trend chart with consumption bars and bill line
        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Bar(
                x=monthly_data["month_str"],
                y=monthly_data["energy_kwh"],
                name="Monthly Energy (kWh)",
                marker_color="#059669",
                hovertemplate="<b>%{x}</b><br>Energy: %{y:.1f} kWh<extra></extra>",
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=monthly_data["month_str"],
                y=monthly_data["estimated_bill"],
                name="Estimated Bill (₹)",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color="#f59e0b", width=2.5),
                marker=dict(size=6),
                hovertemplate="<b>%{x}</b><br>Estimated Bill: ₹%{y:,.2f}<extra></extra>",
            )
        )
        fig_trend.update_layout(
            title="Monthly Energy Consumption & Estimated Electricity Bills",
            template="plotly_white",
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(title="", tickangle=-45),
            yaxis=dict(title="Energy (kWh)", side="left"),
            yaxis2=dict(
                title="Estimated Bill (₹)",
                side="right",
                overlaying="y",
                showgrid=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_trend, width="stretch")

        st.markdown("### Compare Two Specific Months")
        st.write("Select any two months from the historical record to evaluate side-by-side consumption differences.")

        month_options = list(monthly_data["month_str"].unique())
        col_m1, col_m2 = st.columns(2)

        default_idx_a = max(0, len(month_options) - 2)
        default_idx_b = max(0, len(month_options) - 1)

        with col_m1:
            month_a_name = st.selectbox("Baseline Month (A)", options=month_options, index=default_idx_a)
        with col_m2:
            month_b_name = st.selectbox("Comparison Month (B)", options=month_options, index=default_idx_b)

        row_a = monthly_data[monthly_data["month_str"] == month_a_name].iloc[0]
        row_b = monthly_data[monthly_data["month_str"] == month_b_name].iloc[0]

        diff_kwh = row_b["energy_kwh"] - row_a["energy_kwh"]
        diff_pct = (diff_kwh / row_a["energy_kwh"] * 100.0) if row_a["energy_kwh"] > 0 else 0.0
        diff_bill = row_b["estimated_bill"] - row_a["estimated_bill"]

        cmp1, cmp2, cmp3 = st.columns(3)
        cmp1.metric(
            f"Month A: {month_a_name}",
            f"{row_a['energy_kwh']:,.1f} kWh",
            f"₹{row_a['estimated_bill']:,.2f}",
        )
        cmp2.metric(
            f"Month B: {month_b_name}",
            f"{row_b['energy_kwh']:,.1f} kWh",
            f"₹{row_b['estimated_bill']:,.2f}",
        )
        delta_color = "inverse" if diff_kwh > 0 else "normal"
        cmp3.metric(
            "Change (B vs A)",
            f"{diff_pct:+.1f}%",
            f"{diff_kwh:+.1f} kWh (₹{diff_bill:+,.2f})",
            delta_color=delta_color,
        )

        # Comparative bar chart
        cmp_df = pd.DataFrame([
            {"Month": month_a_name, "Energy (kWh)": row_a["energy_kwh"], "Bill (₹)": row_a["estimated_bill"]},
            {"Month": month_b_name, "Energy (kWh)": row_b["energy_kwh"], "Bill (₹)": row_b["estimated_bill"]},
        ])
        fig_cmp = px.bar(
            cmp_df,
            x="Month",
            y="Energy (kWh)",
            text="Energy (kWh)",
            title=f"Direct Comparison: {month_a_name} vs {month_b_name}",
            color="Month",
            color_discrete_sequence=["#059669", "#10b981"],
        )
        fig_cmp.update_layout(template="plotly_white", showlegend=False)
        fig_cmp.update_traces(texttemplate="%{text:,.1f} kWh", textposition="outside")
        st.plotly_chart(fig_cmp, width="stretch")

        st.markdown("### Complete Monthly Consumption History")
        display_monthly = monthly_data[[
            "month_str", "days_recorded", "energy_kwh", "avg_daily_kwh",
            "estimated_bill", "mom_pct_change"
        ]].copy()
        display_monthly.columns = [
            "Month", "Days Recorded", "Total Energy (kWh)", "Daily Average (kWh)",
            "Estimated Bill (₹)", "Month-over-Month Change (%)"
        ]
        display_monthly = display_monthly.round(2)

        st.dataframe(display_monthly, width="stretch", hide_index=True)

        st.download_button(
            "📥 Download Monthly Comparison Data (CSV)",
            display_monthly.to_csv(index=False).encode("utf-8"),
            "monthly_consumption_history.csv",
            "text/csv",
        )


# --------------------------------------------------
# FEATURE 3: UNUSUAL CONSUMPTION DETECTION (ENHANCED ENERGY INSIGHTS)
# --------------------------------------------------

elif page == "Energy Insights":
    st.title("Personalized Energy Insights & Anomaly Detection")
    st.write(
        "Analyze usage distributions, detect statistical consumption anomalies using the "
        "Interquartile Range (IQR) method, and discover practical energy-saving opportunities."
    )

    insights = generate_recommendations(daily_data, tariff)
    stats, outliers_df = detect_iqr_outliers(daily_data)

    # Top Metric row
    a, b, c, d = st.columns(4)
    a.metric("Average Daily Usage", f"{insights['average_daily_usage']:.2f} kWh")
    b.metric("IQR Upper Threshold", f"{insights['threshold']:.2f} kWh")
    b_help = f"Q3 ({insights['q3']:.2f}) + 1.5 × IQR ({insights['iqr']:.2f})"
    c.metric("Abnormal Days Detected", f"{insights['outlier_count']} Days")
    d.metric("Potential Daily Savings", f"₹{insights['estimated_daily_savings']:.2f}")

    # Warning alert if abnormal days detected
    if insights["warning_message"]:
        st.markdown(
            f"""
            <div class="warning-callout">
                <b>⚠️ High Consumption Notice:</b> {insights['warning_message']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Statistical Outlier Timeline (IQR Method)")
    st.caption(
        f"Days exceeding the threshold of **{insights['threshold']:.2f} kWh** are highlighted. "
        "Statistical spikes correspond to heavy usage days (HVAC operation, multiple appliances, or gatherings) "
        "rather than equipment failure."
    )

    # Plotly interactive chart showing daily consumption with outlier markers
    fig_anom = go.Figure()
    # Normal days line
    fig_anom.add_trace(
        go.Scatter(
            x=daily_data["date"],
            y=daily_data["energy_kwh"],
            mode="lines",
            name="Daily Consumption (kWh)",
            line=dict(color="#059669", width=1.5),
            hovertemplate="Date: %{x}<br>Energy: %{y:.2f} kWh<extra></extra>",
        )
    )
    # Threshold horizontal line
    fig_anom.add_hline(
        y=insights["threshold"],
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"IQR Threshold ({insights['threshold']:.2f} kWh)",
        annotation_position="top left",
    )
    # Outlier points
    if not outliers_df.empty:
        fig_anom.add_trace(
            go.Scatter(
                x=outliers_df["date"],
                y=outliers_df["energy_kwh"],
                mode="markers",
                name="Abnormal High Usage",
                marker=dict(color="#ef4444", size=7, symbol="diamond"),
                hovertemplate="<b>High Consumption Spike</b><br>Date: %{x}<br>Energy: %{y:.2f} kWh<extra></extra>",
            )
        )

    fig_anom.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_anom, width="stretch")

    st.markdown("### Unusually High Consumption Days")
    if not outliers_df.empty:
        table_outliers = outliers_df[["date", "energy_kwh", "excess_kwh", "ratio_to_mean"]].copy()
        table_outliers["date"] = pd.to_datetime(table_outliers["date"]).dt.strftime("%d %b %Y")
        table_outliers["estimated_cost"] = (table_outliers["energy_kwh"] * tariff).round(2)
        table_outliers.columns = [
            "Date", "Consumption (kWh)", "Excess Over Average (kWh)",
            "Ratio to Average (x)", "Estimated Cost (₹)"
        ]
        st.dataframe(table_outliers, width="stretch", hide_index=True)
    else:
        st.success("No abnormal high-consumption days detected in the analyzed period.")

    st.markdown("### Practical Energy-Saving Recommendations")
    for index, tip in enumerate(insights["recommendations"], start=1):
        st.markdown(
            f"""
            <div class="insight-card">
                <p><b>{index}. {tip}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if insights.get("disclaimer"):
        st.caption(f"ℹ️ {insights['disclaimer']}")


# --------------------------------------------------
# FEATURE 4: MONTHLY ENERGY SAVING GOAL
# --------------------------------------------------

elif page == "Energy Saving Goal":
    st.title("Monthly Energy Saving Goal")
    st.write(
        "Set personal monthly energy consumption targets, track your achievement, "
        "monitor remaining units, and estimate potential electricity bill savings."
    )

    # Select target month (default: latest recorded month)
    available_months = list(monthly_data["month_str"].unique())
    selected_goal_month = st.selectbox(
        "Select Goal Month",
        options=available_months,
        index=len(available_months) - 1,
        help="Select the month you want to establish or evaluate a goal for.",
    )

    # Get actual consumption for selected month
    target_month_row = monthly_data[monthly_data["month_str"] == selected_goal_month].iloc[0]
    actual_kwh = float(target_month_row["energy_kwh"])
    actual_bill = float(target_month_row["estimated_bill"])

    # Retrieve existing user goal from MySQL (with session state fallback)
    user_id = user.get("id", 1)
    db_goal = None
    try:
        db_goal = get_user_goal(user_id, selected_goal_month)
    except Exception:
        db_goal = None

    # Default suggested target: 10% below actual or 700 kWh
    default_target = float(db_goal["target_kwh"]) if db_goal else round(actual_kwh * 0.90, 1)

    g_col1, g_col2 = st.columns([1, 1])

    with g_col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Set Consumption Target")
        target_kwh_input = st.number_input(
            f"Monthly Target for {selected_goal_month} (kWh)",
            min_value=10.0,
            max_value=10000.0,
            value=float(default_target),
            step=10.0,
            help="Set your monthly electricity limit in kWh.",
        )

        st.caption("Quick Adjustment Suggestions:")
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        if btn_c1.button("5% Reduction", width="stretch"):
            target_kwh_input = round(actual_kwh * 0.95, 1)
        if btn_c2.button("10% Reduction", width="stretch"):
            target_kwh_input = round(actual_kwh * 0.90, 1)
        if btn_c3.button("15% Reduction", width="stretch"):
            target_kwh_input = round(actual_kwh * 0.85, 1)

        if st.button("Save Goal", width="stretch"):
            try:
                ok, msg = save_user_goal(user_id, selected_goal_month, target_kwh_input)
                if ok:
                    st.success(f"Goal for **{selected_goal_month}** set to **{target_kwh_input:.1f} kWh**!")
                    st.rerun()
                else:
                    st.warning(f"Note: {msg}")
            except Exception as e:
                st.error(f"Error saving goal: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    with g_col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Current Goal Summary")
        target_bill = target_kwh_input * tariff
        potential_cost_savings = actual_bill - target_bill
        remaining_units = target_kwh_input - actual_kwh

        st.write(f"**Target Energy:** `{target_kwh_input:.1f} kWh` (Target Bill: `₹{target_bill:,.2f}`)")
        st.write(f"**Actual Consumption:** `{actual_kwh:.1f} kWh` (Actual Bill: `₹{actual_bill:,.2f}`)")

        if remaining_units >= 0:
            st.markdown(
                f"""
                <div class="success-callout">
                    <b>🎯 Target On Track:</b> You have <b>{remaining_units:.1f} kWh</b> remaining under your budget!
                    Potential monthly cost reduction: <b>₹{max(0.0, potential_cost_savings):,.2f}</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="warning-callout">
                    <b>⚠️ Target Exceeded:</b> Consumption exceeds target by <b>{abs(remaining_units):.1f} kWh</b>
                    (Excess cost: <b>₹{abs(potential_cost_savings):,.2f}</b>).
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Goal Progress Gauge Chart
    st.markdown("### Target Achievement Progress")
    progress_pct = min(150.0, (actual_kwh / target_kwh_input * 100.0)) if target_kwh_input > 0 else 0.0

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=actual_kwh,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Consumption Progress vs Target ({selected_goal_month})"},
            delta={"reference": target_kwh_input, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
            gauge={
                "axis": {"range": [None, max(target_kwh_input * 1.3, actual_kwh * 1.1)]},
                "bar": {"color": "#064e3b" if actual_kwh <= target_kwh_input else "#ef4444"},
                "steps": [
                    {"range": [0, target_kwh_input * 0.8], "color": "#d1fae5"},
                    {"range": [target_kwh_input * 0.8, target_kwh_input], "color": "#fef3c7"},
                    {"range": [target_kwh_input, max(target_kwh_input * 1.3, actual_kwh * 1.1)], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#b91c1c", "width": 4},
                    "thickness": 0.75,
                    "value": target_kwh_input,
                },
            },
        )
    )
    fig_gauge.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, width="stretch")

    # Past Goals History from MySQL
    st.markdown("### Your Energy Saving Goals History")
    user_goals_history = []
    try:
        user_goals_history = get_user_goals(user_id)
    except Exception:
        user_goals_history = []

    if user_goals_history:
        history_df = pd.DataFrame(user_goals_history)
        history_df = history_df[["month_year", "target_kwh", "created_at"]].copy()
        history_df.columns = ["Month", "Target (kWh)", "Last Updated"]
        st.dataframe(history_df, width="stretch", hide_index=True)
    else:
        st.info("No saved goals found in the database. Set your first goal above!")


# --------------------------------------------------
# FEATURE 5: WHAT-IF ENERGY SIMULATOR
# --------------------------------------------------

elif page == "Energy Simulator":
    st.title("What-if Energy Simulator")
    st.write(
        "Simulate the direct impact of altering appliance daily usage hours on your monthly "
        "energy consumption and electricity bills. Explore realistic conservation scenarios "
        "to discover actionable cost reductions."
    )

    st.markdown(
        """
        <span class="scenario-badge-actual">ACTUAL MEASURED BASELINE</span>
        &nbsp;vs&nbsp;
        <span class="scenario-badge-simulated">SIMULATED SCENARIO</span>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # Select baseline month or use overall historical average
    sim_mode = st.radio(
        "Select Baseline Reference",
        ["Historical Monthly Average", "Specific Recorded Month"],
        horizontal=True,
    )

    if sim_mode == "Historical Monthly Average":
        baseline_kwh = float(monthly_data["energy_kwh"].mean()) if not monthly_data.empty else 750.0
        baseline_label = "Historical Average Month"
    else:
        selected_sim_month = st.selectbox(
            "Select Baseline Month",
            options=list(monthly_data["month_str"].unique()),
            index=len(monthly_data) - 1,
        )
        baseline_row = monthly_data[monthly_data["month_str"] == selected_sim_month].iloc[0]
        baseline_kwh = float(baseline_row["energy_kwh"])
        baseline_label = f"Actual {selected_sim_month}"

    st.subheader("Modify Daily Appliance Operating Hours")
    st.write("Adjust the sliders below to simulate reducing or extending usage for major household loads:")

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.markdown("**❄️ Air Conditioner (1500W)**")
        ac_orig = st.slider("AC Baseline Usage (Hours/Day)", 0.0, 24.0, 8.0, 0.5, key="ac_orig")
        ac_sim = st.slider("AC Simulated Usage (Hours/Day)", 0.0, 24.0, 5.0, 0.5, key="ac_sim")

        st.markdown("**🚿 Water Geyser / Heater (2000W)**")
        geyser_orig = st.slider("Geyser Baseline Usage (Hours/Day)", 0.0, 8.0, 2.0, 0.5, key="gey_orig")
        geyser_sim = st.slider("Geyser Simulated Usage (Hours/Day)", 0.0, 8.0, 1.0, 0.5, key="gey_sim")

    with sim_col2:
        st.markdown("**🌀 Ceiling Fans (3 Units @ 75W = 225W)**")
        fan_orig = st.slider("Fans Baseline Usage (Hours/Day)", 0.0, 24.0, 12.0, 0.5, key="fan_orig")
        fan_sim = st.slider("Fans Simulated Usage (Hours/Day)", 0.0, 24.0, 10.0, 0.5, key="fan_sim")

        st.markdown("**💡 Lighting (8 LED Bulbs @ 14W = 112W)**")
        light_orig = st.slider("Lighting Baseline Usage (Hours/Day)", 0.0, 24.0, 8.0, 0.5, key="light_orig")
        light_sim = st.slider("Lighting Simulated Usage (Hours/Day)", 0.0, 24.0, 6.0, 0.5, key="light_sim")

    # Run simulation
    simulation_mods = [
        {"name": "Air Conditioner (1.5 Ton)", "power_watts": 1500, "count": 1, "original_hours": ac_orig, "simulated_hours": ac_sim, "days_per_month": 30},
        {"name": "Water Geyser", "power_watts": 2000, "count": 1, "original_hours": geyser_orig, "simulated_hours": geyser_sim, "days_per_month": 30},
        {"name": "Ceiling Fans", "power_watts": 75, "count": 3, "original_hours": fan_orig, "simulated_hours": fan_sim, "days_per_month": 30},
        {"name": "LED Lighting", "power_watts": 14, "count": 8, "original_hours": light_orig, "simulated_hours": light_sim, "days_per_month": 30},
    ]

    sim_result = simulate_appliance_change(baseline_kwh, simulation_mods, tariff)

    st.markdown("### Simulation Results & Comparison")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Original Baseline Energy", f"{sim_result['baseline_monthly_kwh']:,.1f} kWh")
    s2.metric("Simulated Monthly Energy", f"{sim_result['simulated_monthly_kwh']:,.1f} kWh", f"{sim_result['pct_change']:+.1f}%")
    s3.metric("Original Monthly Bill", f"₹{sim_result['original_bill']:,.2f}")
    delta_color = "normal" if sim_result["is_savings"] else "inverse"
    s4.metric(
        "Simulated Monthly Bill",
        f"₹{sim_result['simulated_bill']:,.2f}",
        f"₹{sim_result['bill_savings']:+,.2f}",
        delta_color=delta_color,
    )

    if sim_result["is_savings"]:
        st.markdown(
            f"""
            <div class="success-callout">
                <b>🎉 Potential Savings Identified:</b> By applying these adjustments, you can save
                <b>{abs(sim_result['kwh_delta']):.1f} kWh/month</b>, reducing your monthly electricity bill by
                <b>₹{sim_result['bill_savings']:,.2f}</b> ({abs(sim_result['pct_change']):.1f}% reduction).
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif sim_result["bill_savings"] < 0:
        st.markdown(
            f"""
            <div class="warning-callout">
                <b>⚠️ Increased Consumption Scenario:</b> These adjustments increase monthly demand by
                <b>{abs(sim_result['kwh_delta']):.1f} kWh/month</b>, raising your bill by
                <b>₹{abs(sim_result['bill_savings']):,.2f}</b> (+{abs(sim_result['pct_change']):.1f}%).
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Plotly Comparison Bar Chart
    comp_chart_df = pd.DataFrame([
        {"Scenario": "Original Baseline", "Energy (kWh)": sim_result["baseline_monthly_kwh"], "Bill (₹)": sim_result["original_bill"]},
        {"Scenario": "Simulated Scenario", "Energy (kWh)": sim_result["simulated_monthly_kwh"], "Bill (₹)": sim_result["simulated_bill"]},
    ])
    fig_sim_bar = px.bar(
        comp_chart_df,
        x="Scenario",
        y="Energy (kWh)",
        text="Energy (kWh)",
        title="Baseline vs Simulated Monthly Consumption",
        color="Scenario",
        color_discrete_sequence=["#064e3b", "#10b981"],
    )
    fig_sim_bar.update_layout(template="plotly_white", showlegend=False)
    fig_sim_bar.update_traces(texttemplate="%{text:,.1f} kWh", textposition="outside")
    st.plotly_chart(fig_sim_bar, width="stretch")

    st.subheader("Detailed Impact by Appliance")
    st.dataframe(sim_result["detailed_df"], width="stretch", hide_index=True)


# --------------------------------------------------
# FEATURE 6: IMPROVE ENERGY PREDICTION
# --------------------------------------------------

elif page == "Energy Prediction":
    st.title("Future Energy Consumption Prediction")
    st.write(
        "Forecast future household electricity demand using the trained Random Forest regression model. "
        "Select your forecast horizon to view predicted daily consumption, total energy demand, and estimated electricity costs."
    )

    if model is None:
        st.warning("Trained model file not found or could not be loaded.")
        st.caption("Expected location: models/best_model.pkl")
        if model_error:
            with st.expander("Model loading details"):
                st.code(model_error)
    else:
        st.markdown("### Select Forecast Horizon")
        p_col1, p_col2 = st.columns([1, 1])

        with p_col1:
            forecast_period = st.radio(
                "Forecast Period Option",
                options=[7, 14, 30],
                format_func=lambda x: f"{x} Days ({'1 Week' if x == 7 else ('2 Weeks' if x == 14 else '1 Month')})",
                horizontal=True,
                index=0,
                help="Choose between 7-day short-term or 30-day monthly forecasting horizons.",
            )

        with p_col2:
            show_history = st.checkbox(
                "Show Recent Historical Context (Last 14 Days)",
                value=True,
                help="Plots the 14 days immediately preceding the forecast for contextual continuity.",
            )

        if st.button("Generate Forecast", width="stretch"):
            try:
                last_dataset_date = daily_data["date"].max()
                future = predict_future(
                    model=model,
                    last_date=last_dataset_date,
                    days=forecast_period,
                    tariff=tariff,
                )

                total_pred_kwh = float(future["predicted_energy_kwh"].sum())
                avg_daily_pred = float(future["predicted_energy_kwh"].mean())
                total_pred_bill = total_pred_kwh * tariff
                peak_pred_row = future.loc[future["predicted_energy_kwh"].idxmax()]

                # Prediction Metric Cards
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Total Forecasted Energy", f"{total_pred_kwh:,.2f} kWh")
                f2.metric("Estimated Electricity Bill", f"₹{total_pred_bill:,.2f}")
                f3.metric("Daily Average Forecast", f"{avg_daily_pred:.2f} kWh")
                f4.metric(
                    "Peak Predicted Day",
                    f"{peak_pred_row['date']:%d %b %Y}",
                    f"{peak_pred_row['predicted_energy_kwh']:.2f} kWh",
                )

                desc_f = st.columns(4)
                desc_f[0].caption(f"Projected over next {forecast_period} days.")
                desc_f[1].caption(f"Calculated at ₹{tariff:.2f}/kWh.")
                desc_f[2].caption("Average anticipated daily load.")
                desc_f[3].caption(f"{peak_pred_row['day_name']} peak load.")

                st.markdown("### Interactive Forecast Visualization")

                # Build interactive Plotly Line Chart
                fig_forecast = go.Figure()

                if show_history:
                    recent_hist = daily_data.tail(14).copy()
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=recent_hist["date"],
                            y=recent_hist["energy_kwh"],
                            mode="lines+markers",
                            name="Actual Historical",
                            line=dict(color="#64748b", width=2),
                            marker=dict(size=6),
                            hovertemplate="<b>Actual Historical</b><br>Date: %{x}<br>Energy: %{y:.2f} kWh<extra></extra>",
                        )
                    )

                fig_forecast.add_trace(
                    go.Scatter(
                        x=future["date"],
                        y=future["predicted_energy_kwh"],
                        mode="lines+markers",
                        name="Model Prediction (RF)",
                        line=dict(color="#059669", width=2.5),
                        marker=dict(size=7, color="#047857"),
                        hovertemplate="<b>Forecasted Demand</b><br>Date: %{x}<br>Energy: %{y:.2f} kWh<extra></extra>",
                    )
                )

                # Mean prediction line
                fig_forecast.add_hline(
                    y=avg_daily_pred,
                    line_dash="dot",
                    line_color="#10b981",
                    annotation_text=f"Mean Forecast ({avg_daily_pred:.2f} kWh)",
                    annotation_position="bottom right",
                )

                # Dividing line for forecast start
                if show_history:
                    fig_forecast.add_vline(
                        x=last_dataset_date,
                        line_dash="dash",
                        line_color="#94a3b8",
                        annotation_text="Forecast Horizon Start",
                        annotation_position="top left",
                    )

                fig_forecast.update_layout(
                    title=f"Future {forecast_period}-Day Household Energy Demand Forecast",
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=50, b=10),
                    xaxis_title="Date",
                    yaxis_title="Energy Consumption (kWh)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_forecast, width="stretch")

                st.subheader("Daily Forecast Breakdown Table")
                display_future = future[[
                    "date", "day_name", "predicted_energy_kwh", "estimated_cost",
                    "cumulative_energy_kwh", "cumulative_cost"
                ]].copy()
                display_future["date"] = display_future["date"].dt.strftime("%d %b %Y")
                display_future.columns = [
                    "Date", "Day of Week", "Predicted Energy (kWh)", "Estimated Daily Cost (₹)",
                    "Cumulative Energy (kWh)", "Cumulative Cost (₹)"
                ]
                display_future = display_future.round(2)

                st.dataframe(display_future, width="stretch", hide_index=True)

                st.download_button(
                    "📥 Download Forecast CSV",
                    display_future.to_csv(index=False).encode("utf-8"),
                    f"smartenergy_{forecast_period}day_forecast.csv",
                    "text/csv",
                )

            except Exception as exc:
                st.error(f"Forecast generation failed: {exc}")


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":
    st.title("Machine Learning Model Performance")
    st.write("Comparative evaluation metrics for regression algorithms trained on the household energy dataset.")

    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            results = pd.DataFrame(metrics["results"]).T

            st.subheader("Model Evaluation Results")
            st.dataframe(results.round(4), width="stretch")

            best_model = metrics.get("best_model", "Not specified")
            st.success(f"🏆 Best performing model: **{best_model}**")

            if "RMSE" in results.columns:
                chart = results.reset_index().rename(columns={"index": "Model"})
                fig = px.bar(
                    chart,
                    x="Model",
                    y="RMSE",
                    title="Root Mean Squared Error (RMSE) Comparison — Lower is Better",
                    color="Model",
                    color_discrete_sequence=["#059669", "#10b981"],
                    text="RMSE",
                )
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Model",
                    yaxis_title="RMSE",
                    showlegend=False,
                )
                fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                st.plotly_chart(fig, width="stretch")

        except Exception as exc:
            st.error(f"Unable to read model metrics: {exc}")
    else:
        st.warning("Model metrics file not found. Expected location: models/model_metrics.json")


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

elif page == "About":
    st.title("About SmartEnergy AI")
    st.write(
        "SmartEnergy AI is an intelligent household electricity consumption analysis and prediction platform. "
        "It leverages machine learning and statistical analytics to help households understand historical "
        "consumption, simulate efficiency changes, detect consumption anomalies, calculate appliance-level loads, "
        "and accurately forecast future power demand."
    )

    st.divider()

    st.subheader("Project Objectives")
    st.markdown("""
    - Analyze historical household electricity consumption.
    - Identify daily, weekly, and monthly electricity usage patterns.
    - Calculate appliance-level consumption and costs with interactive tools.
    - Compare historical monthly consumption and detect seasonal variations.
    - Set and track user-specific monthly energy conservation goals in MySQL.
    - Simulate what-if scenarios for appliance usage adjustments.
    - Detect statistical consumption anomalies using the IQR method.
    - Forecast future energy requirements using a trained Random Forest Regressor.
    - Maintain strict user isolation and transparent electricity bill calculations.
    """)

    st.divider()

    st.subheader("Dataset Information")
    st.markdown("""
    **Dataset:** Individual Household Electric Power Consumption
    
    **Source:** UCI Machine Learning Repository
    
    **Characteristics:**
    - High-frequency electrical measurements sampled over a multi-year timeframe.
    - Measurements include Global Active Power (kW), Global Reactive Power (kVAR), Voltage (V), and Sub-metering circuits (Wh).
    - Minute-level observations are aggregated into total daily energy consumption (kWh) for analytics and modeling.
    """)

    st.divider()

    st.subheader("Machine Learning Pipeline")
    st.markdown("""
    **Problem Formulation:** Supervised Regression
    
    **Evaluated Models:**
    - Random Forest Regressor (Selected Best Model)
    - Linear Regression
    
    **Prediction Target:** Daily household energy consumption (kWh)
    
    **Engineered Features:**
    - `day` (Day of Month)
    - `month` (Calendar Month)
    - `day_of_week` (Day of Week index, 0=Monday)
    - `day_of_year` (Day of Year, 1-366)
    
    **Evaluation Metrics:**
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R-squared ($R^2$) Score
    """)

    st.divider()

    st.subheader("Application Modules")
    st.markdown("""
    1. **Dashboard:** Key metrics, historical consumption trends, and cost snapshots.
    2. **Consumption Analysis:** Histograms, weekday demand analysis, and descriptive statistics.
    3. **Monthly Comparison:** Month-over-month percentage changes, peak month detection, and side-by-side comparison.
    4. **Appliance Energy Calculator:** Wattage, unit, and operating hour calculations with Plotly breakdowns.
    5. **Energy Simulator:** What-if scenario testing contrasting actual baselines with simulated modifications.
    6. **Energy Prediction:** 7-day and 30-day forecast horizons powered by Random Forest regression.
    7. **Energy Insights:** Statistical anomaly detection via Tukey's IQR fences with practical saving tips.
    8. **Energy Saving Goal:** User-authenticated target tracking, MySQL persistence, and gauge visualizer.
    9. **Model Performance:** Model evaluation metrics and RMSE benchmarking.
    """)

    st.info("SmartEnergy AI – Intelligent Household Energy Consumption Prediction and Analysis.")
