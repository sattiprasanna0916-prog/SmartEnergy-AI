import pandas as pd
def generate_recommendations(daily_energy, tariff=6.0):
    if daily_energy is None or daily_energy.empty:
        return {
            "average_daily_usage": 0,
            "maximum_daily_usage": 0,
            "estimated_daily_savings": 0,
            "usage_summary": "No consumption data is available.",
            "high_usage_days": [],
            "recommendations": [
                "Load valid household electricity consumption data."
            ],
        }
    data = daily_energy.copy()
    data["energy_kwh"] = pd.to_numeric(
        data["energy_kwh"],
        errors="coerce",
    )
    data = data.dropna(
        subset=["energy_kwh"]
    )

    data = data[data["energy_kwh"] >= 0]

    if data.empty:
        return {
            "average_daily_usage": 0,
            "maximum_daily_usage": 0,
            "estimated_daily_savings": 0,
            "usage_summary": "No valid consumption values are available.",
            "high_usage_days": [],
            "recommendations": [
                "Check the dataset for valid energy consumption values."
            ],
        }

    avg = float(data["energy_kwh"].mean())

    peak = float(data["energy_kwh"].max())

    median = float(data["energy_kwh"].median())

    minimum = float(data["energy_kwh"].min())

    std = float(data["energy_kwh"].std())

    tariff = float(tariff)

    # Identify unusually high usage days using the 75th percentile.
    threshold = float(
        data["energy_kwh"].quantile(0.75)
    )

    high_usage = data[
        data["energy_kwh"] > threshold
    ].sort_values(
        "energy_kwh",
        ascending=False,
    ).head(5)

    high_usage_days = []

    for _, row in high_usage.iterrows():

        high_usage_days.append({
            "date": row["date"],
            "energy_kwh": round(
                float(row["energy_kwh"]),
                2,
            ),
        })

    # Estimate a possible 10% reduction.
    estimated_savings = avg * 0.10 * tariff

    tips = []

    if avg > 15:

        tips.append(
            "Your average daily electricity usage is high. "
            "Review the operating time of high-power appliances."
        )

    elif avg > 8:

        tips.append(
            "Your average daily electricity usage is moderate. "
            "Try reducing unnecessary appliance operating time."
        )

    else:

        tips.append(
            "Your average daily electricity usage is relatively low. "
            "Continue following energy-efficient habits."
        )

    if peak > avg * 1.5:

        tips.append(
            f"Your highest daily usage is {peak:.2f} kWh, "
            f"which is considerably higher than your average of {avg:.2f} kWh. "
            "Review activities and appliance usage on the peak-consumption date."
        )

    if threshold > avg:

        tips.append(
            f"Several high-usage days exceed approximately "
            f"{threshold:.2f} kWh. Compare those dates with your normal usage."
        )

    if std > 0:

        tips.append(
            "Your consumption varies across days. "
            "Maintaining a consistent appliance usage schedule "
            "may help control sudden increases."
        )

    if minimum < avg * 0.5:

        tips.append(
            "Some days have much lower consumption than your average. "
            "Compare those days to identify energy-saving habits."
        )

    tips.append(
        "Switch off lights, fans and appliances when they are not required."
    )

    tips.append(
        "Use LED lighting and energy-efficient appliances where practical."
    )

    tips.append(
        "Avoid leaving electronic devices in standby mode for long periods."
    )

    usage_summary = (
        f"Your average daily consumption is {avg:.2f} kWh. "
        f"The maximum recorded daily consumption is {peak:.2f} kWh, "
        f"and the median daily consumption is {median:.2f} kWh."
    )

    return {
        "average_daily_usage": round(avg, 2),

        "maximum_daily_usage": round(peak, 2),

        "estimated_daily_savings": round(
            estimated_savings,
            2,
        ),

        "usage_summary": usage_summary,

        "high_usage_days": high_usage_days,

        "recommendations": tips,
    }
