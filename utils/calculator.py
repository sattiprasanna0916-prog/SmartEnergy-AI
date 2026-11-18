"""
Appliance Energy Calculator and What-If Simulator Utility Module.
Handles energy consumption calculations, standard appliance presets,
and scenario simulation for household electricity management.
"""

from typing import List, Dict, Any
import pandas as pd


# Standard household appliance presets with realistic ratings (Watts) and typical daily usage (hours)
APPLIANCE_PRESETS = [
    {
        "name": "Air Conditioner (1.5 Ton Split)",
        "category": "Cooling / Heating",
        "power_watts": 1500,
        "default_hours": 8.0,
        "default_count": 1,
        "default_days": 30,
        "icon": "❄️",
    },
    {
        "name": "Refrigerator (Frost Free, 260L)",
        "category": "Kitchen",
        "power_watts": 200,
        "default_hours": 24.0,  # Continuous run / duty cycle
        "default_count": 1,
        "default_days": 30,
        "icon": "🧊",
    },
    {
        "name": "Ceiling Fan",
        "category": "Comfort",
        "power_watts": 75,
        "default_hours": 12.0,
        "default_count": 3,
        "default_days": 30,
        "icon": "🌀",
    },
    {
        "name": "LED Television (43-inch)",
        "category": "Entertainment",
        "power_watts": 100,
        "default_hours": 5.0,
        "default_count": 1,
        "default_days": 30,
        "icon": "📺",
    },
    {
        "name": "Water Geyser / Water Heater",
        "category": "Bathroom",
        "power_watts": 2000,
        "default_hours": 1.5,
        "default_count": 1,
        "default_days": 30,
        "icon": "🚿",
    },
    {
        "name": "Washing Machine (Front/Top Load)",
        "category": "Laundry",
        "power_watts": 500,
        "default_hours": 1.0,
        "default_count": 1,
        "default_days": 30,
        "icon": "🧺",
    },
    {
        "name": "Microwave Oven",
        "category": "Kitchen",
        "power_watts": 1200,
        "default_hours": 0.5,
        "default_count": 1,
        "default_days": 30,
        "icon": "🍲",
    },
    {
        "name": "Desktop PC / Workstation",
        "category": "Work & Study",
        "power_watts": 250,
        "default_hours": 6.0,
        "default_count": 1,
        "default_days": 30,
        "icon": "💻",
    },
    {
        "name": "LED Light Bulbs",
        "category": "Lighting",
        "power_watts": 15,
        "default_hours": 6.0,
        "default_count": 6,
        "default_days": 30,
        "icon": "💡",
    },
    {
        "name": "Induction Cooktop",
        "category": "Kitchen",
        "power_watts": 1800,
        "default_hours": 1.5,
        "default_count": 1,
        "default_days": 30,
        "icon": "🍳",
    },
    {
        "name": "Electric Iron",
        "category": "Laundry",
        "power_watts": 1000,
        "default_hours": 0.5,
        "default_count": 1,
        "default_days": 15,
        "icon": "👔",
    },
    {
        "name": "Water Purifier (RO + UV)",
        "category": "Kitchen",
        "power_watts": 60,
        "default_hours": 4.0,
        "default_count": 1,
        "default_days": 30,
        "icon": "💧",
    },
]


def calculate_single_appliance(
    power_watts: float,
    count: int,
    daily_hours: float,
    days_per_month: int = 30,
    tariff: float = 6.0,
) -> Dict[str, float]:
    """
    Calculate daily and monthly energy consumption and cost for a single appliance.

    Formula:
    Daily Energy (kWh) = (Power in Watts * Count * Daily Usage Hours) / 1000
    Monthly Energy (kWh) = Daily Energy (kWh) * Days of usage per month
    Monthly Cost (INR) = Monthly Energy (kWh) * Tariff (INR/kWh)
    """
    power_watts = max(0.0, float(power_watts))
    count = max(0, int(count))
    daily_hours = max(0.0, min(24.0, float(daily_hours)))
    days_per_month = max(0, min(31, int(days_per_month)))
    tariff = max(0.0, float(tariff))

    daily_kwh = (power_watts * count * daily_hours) / 1000.0
    monthly_kwh = daily_kwh * days_per_month
    monthly_cost = monthly_kwh * tariff

    return {
        "power_watts": round(power_watts, 1),
        "count": count,
        "daily_hours": round(daily_hours, 2),
        "days_per_month": days_per_month,
        "daily_kwh": round(daily_kwh, 3),
        "monthly_kwh": round(monthly_kwh, 2),
        "monthly_cost": round(monthly_cost, 2),
    }


def calculate_appliances_summary(
    appliances: List[Dict[str, Any]],
    tariff: float = 6.0,
) -> Dict[str, Any]:
    """
    Compute summary metrics, dataframe, and percentage share across all configured appliances.
    """
    if not appliances:
        return {
            "total_daily_kwh": 0.0,
            "total_monthly_kwh": 0.0,
            "total_monthly_cost": 0.0,
            "highest_appliance": "None",
            "highest_kwh": 0.0,
            "df": pd.DataFrame(),
        }

    rows = []
    total_daily = 0.0
    total_monthly = 0.0
    total_cost = 0.0

    for item in appliances:
        calc = calculate_single_appliance(
            power_watts=item.get("power_watts", 0),
            count=item.get("count", 1),
            daily_hours=item.get("daily_hours", 0),
            days_per_month=item.get("days_per_month", 30),
            tariff=tariff,
        )
        total_daily += calc["daily_kwh"]
        total_monthly += calc["monthly_kwh"]
        total_cost += calc["monthly_cost"]

        rows.append({
            "Appliance": item.get("name", "Unknown"),
            "Category": item.get("category", "General"),
            "Power (Watts)": calc["power_watts"],
            "Units": calc["count"],
            "Daily Hours": calc["daily_hours"],
            "Days / Month": calc["days_per_month"],
            "Daily Energy (kWh)": calc["daily_kwh"],
            "Monthly Energy (kWh)": calc["monthly_kwh"],
            "Monthly Cost (₹)": calc["monthly_cost"],
        })

    df = pd.DataFrame(rows)

    if not df.empty and total_monthly > 0:
        df["Share (%)"] = (df["Monthly Energy (kWh)"] / total_monthly * 100.0).round(1)
        highest_row = df.loc[df["Monthly Energy (kWh)"].idxmax()]
        highest_appliance = highest_row["Appliance"]
        highest_kwh = float(highest_row["Monthly Energy (kWh)"])
    else:
        df["Share (%)"] = 0.0
        highest_appliance = "None"
        highest_kwh = 0.0

    return {
        "total_daily_kwh": round(total_daily, 2),
        "total_monthly_kwh": round(total_monthly, 2),
        "total_monthly_cost": round(total_cost, 2),
        "highest_appliance": highest_appliance,
        "highest_kwh": round(highest_kwh, 2),
        "df": df,
    }


def simulate_appliance_change(
    baseline_monthly_kwh: float,
    appliance_changes: List[Dict[str, Any]],
    tariff: float = 6.0,
) -> Dict[str, Any]:
    """
    Simulate impact of changing appliance usage hours on total monthly consumption and bill.

    appliance_changes format:
    [
        {
            "name": str,
            "power_watts": float,
            "count": int,
            "original_hours": float,
            "simulated_hours": float,
            "days_per_month": int,
        },
        ...
    ]
    """
    baseline_monthly_kwh = max(0.0, float(baseline_monthly_kwh))
    tariff = max(0.0, float(tariff))

    total_kwh_delta = 0.0
    detailed_changes = []

    for item in appliance_changes:
        power = float(item.get("power_watts", 0))
        count = int(item.get("count", 1))
        orig_h = float(item.get("original_hours", 0))
        sim_h = float(item.get("simulated_hours", 0))
        days = int(item.get("days_per_month", 30))

        orig_monthly_kwh = (power * count * orig_h / 1000.0) * days
        sim_monthly_kwh = (power * count * sim_h / 1000.0) * days
        delta_kwh = sim_monthly_kwh - orig_monthly_kwh
        cost_diff = delta_kwh * tariff

        total_kwh_delta += delta_kwh

        detailed_changes.append({
            "Appliance": item.get("name", "Appliance"),
            "Original Hours": orig_h,
            "Simulated Hours": sim_h,
            "Hour Change": round(sim_h - orig_h, 1),
            "Original Monthly (kWh)": round(orig_monthly_kwh, 2),
            "Simulated Monthly (kWh)": round(sim_monthly_kwh, 2),
            "Monthly kWh Impact": round(delta_kwh, 2),
            "Monthly Cost Impact (₹)": round(cost_diff, 2),
        })

    simulated_monthly_kwh = max(0.0, baseline_monthly_kwh + total_kwh_delta)
    original_bill = baseline_monthly_kwh * tariff
    simulated_bill = simulated_monthly_kwh * tariff
    bill_savings = original_bill - simulated_bill  # Positive means savings, negative means increase
    pct_change = (
        ((simulated_monthly_kwh - baseline_monthly_kwh) / baseline_monthly_kwh * 100.0)
        if baseline_monthly_kwh > 0
        else 0.0
    )

    return {
        "baseline_monthly_kwh": round(baseline_monthly_kwh, 2),
        "simulated_monthly_kwh": round(simulated_monthly_kwh, 2),
        "original_bill": round(original_bill, 2),
        "simulated_bill": round(simulated_bill, 2),
        "kwh_delta": round(total_kwh_delta, 2),
        "bill_savings": round(bill_savings, 2),
        "pct_change": round(pct_change, 2),
        "is_savings": bill_savings > 0,
        "detailed_df": pd.DataFrame(detailed_changes),
    }
