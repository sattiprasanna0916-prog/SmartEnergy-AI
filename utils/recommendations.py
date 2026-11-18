"""
Recommendations and Consumption Insights Module.
Analyzes household energy usage patterns, identifies statistical spikes
using the IQR method, and provides data-driven, non-alarmist recommendations.
"""

from typing import Dict, Any, List
import pandas as pd


def generate_recommendations(daily_energy: pd.DataFrame, tariff: float = 6.0) -> Dict[str, Any]:
    """
    Generate comprehensive energy insights, IQR-based outlier detection,
    and actionable energy-saving recommendations.
    """
    if daily_energy is None or daily_energy.empty:
        return {
            "average_daily_usage": 0.0,
            "maximum_daily_usage": 0.0,
            "median_daily_usage": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "threshold": 0.0,
            "estimated_daily_savings": 0.0,
            "usage_summary": "No consumption data is available.",
            "high_usage_days": [],
            "warning_message": None,
            "outlier_count": 0,
            "total_days": 0,
            "recommendations": [
                "Load valid household electricity consumption data."
            ],
            "disclaimer": "",
        }

    data = daily_energy.copy()
    data["energy_kwh"] = pd.to_numeric(
        data["energy_kwh"],
        errors="coerce",
    )
    data = data.dropna(subset=["energy_kwh"])
    data = data[data["energy_kwh"] >= 0]

    if data.empty:
        return {
            "average_daily_usage": 0.0,
            "maximum_daily_usage": 0.0,
            "median_daily_usage": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "threshold": 0.0,
            "estimated_daily_savings": 0.0,
            "usage_summary": "No valid consumption values are available.",
            "high_usage_days": [],
            "warning_message": None,
            "outlier_count": 0,
            "total_days": 0,
            "recommendations": [
                "Check the dataset for valid energy consumption values."
            ],
            "disclaimer": "",
        }

    avg = float(data["energy_kwh"].mean())
    peak = float(data["energy_kwh"].max())
    median = float(data["energy_kwh"].median())
    minimum = float(data["energy_kwh"].min())
    std = float(data["energy_kwh"].std())
    tariff = float(tariff)

    # Statistical IQR Outlier Detection (Tukey's Fences)
    q1 = float(data["energy_kwh"].quantile(0.25))
    q3 = float(data["energy_kwh"].quantile(0.75))
    iqr = q3 - q1
    iqr_threshold = q3 + (1.5 * iqr)

    # Detect high-consumption outlier days
    outlier_df = data[data["energy_kwh"] > iqr_threshold].sort_values(
        "energy_kwh",
        ascending=False,
    )
    outlier_count = len(outlier_df)

    high_usage_days: List[Dict[str, Any]] = []
    for _, row in outlier_df.iterrows():
        kwh_val = float(row["energy_kwh"])
        high_usage_days.append({
            "date": row["date"],
            "energy_kwh": round(kwh_val, 2),
            "excess_over_avg": round(kwh_val - avg, 2),
            "ratio_to_avg": round(kwh_val / avg if avg > 0 else 1.0, 2),
            "estimated_cost": round(kwh_val * tariff, 2),
        })

    # Estimate a realistic 10% daily savings potential
    estimated_savings = avg * 0.10 * tariff

    # Construct contextual warning messages
    warning_message = None
    if outlier_count > 0:
        warning_message = (
            f"Detected {outlier_count} day(s) with unusually high energy consumption exceeding "
            f"the statistical threshold of {iqr_threshold:.2f} kWh (Q3 + 1.5×IQR). "
            f"These peaks reflect periods of heavy electrical demand (such as multiple simultaneous "
            f"appliances, extreme weather HVAC usage, or special gatherings)."
        )

    # Practical recommendations
    tips: List[str] = []

    if avg > 30:
        tips.append(
            "Your average daily electricity consumption is very high (> 30 kWh/day). "
            "Consider setting the air conditioning / heating thermostat to 24°C–25°C, "
            "as every 1°C adjustment can save approximately 6% of HVAC energy."
        )
    elif avg > 15:
        tips.append(
            "Your average daily electricity consumption is moderately high (15–30 kWh/day). "
            "Review high-load appliances like water geysers and air conditioners. "
            "Setting timers or limiting operation can substantially reduce monthly bills."
        )
    else:
        tips.append(
            "Your average daily electricity usage is within an efficient range (< 15 kWh/day). "
            "Maintaining regular appliance maintenance and good habits will sustain this efficiency."
        )

    if peak > avg * 1.5:
        tips.append(
            f"Peak recorded daily usage reached {peak:.2f} kWh, which is "
            f"{(peak / avg if avg > 0 else 0):.1f}x higher than your average of {avg:.2f} kWh. "
            "Staggering heavy appliance runtimes (e.g. running the washing machine, water heater, "
            "and iron at separate times) helps avoid sudden demand surges."
        )

    if outlier_count > 5:
        tips.append(
            f"There are {outlier_count} statistical outlier days in your historical record. "
            "Correlate these dates with seasonal weather extremes or household activities "
            "to plan proactive conservation strategies during similar future periods."
        )

    if std > 8:
        tips.append(
            "Significant day-to-day consumption variance observed. Establishing consistent "
            "daily routines for heating/cooling and lighting provides predictable utility costs."
        )

    tips.append(
        "Replace remaining conventional bulbs with high-efficiency 9W–12W LED lighting to reduce "
        "continuous lighting loads by up to 80%."
    )
    tips.append(
        "Eliminate 'vampire power' by using smart power strips or turning off entertainment systems, "
        "chargers, and kitchen gadgets at the wall switch when not in active use."
    )
    tips.append(
        "Schedule regular maintenance for HVAC systems and clean appliance filters every 4–6 weeks "
        "to ensure optimal operating efficiency."
    )

    usage_summary = (
        f"Your household average daily consumption is {avg:.2f} kWh, with a median of {median:.2f} kWh. "
        f"The maximum recorded daily consumption is {peak:.2f} kWh. "
        f"Using the IQR method (Q1={q1:.2f} kWh, Q3={q3:.2f} kWh, IQR={iqr:.2f} kWh), "
        f"the statistical high-consumption threshold is {iqr_threshold:.2f} kWh, "
        f"identifying {outlier_count} unusually high-demand day(s)."
    )

    disclaimer = (
        "Statistical Note: Unusual consumption detection flags statistical spikes based on historical "
        "distribution patterns. These variations indicate intense usage periods and are not to be "
        "interpreted as confirmed appliance breakdowns or electrical circuit faults."
    )

    return {
        "average_daily_usage": round(avg, 2),
        "maximum_daily_usage": round(peak, 2),
        "median_daily_usage": round(median, 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "threshold": round(iqr_threshold, 2),
        "estimated_daily_savings": round(estimated_savings, 2),
        "usage_summary": usage_summary,
        "high_usage_days": high_usage_days,
        "warning_message": warning_message,
        "outlier_count": outlier_count,
        "total_days": len(data),
        "recommendations": tips,
        "disclaimer": disclaimer,
    }
