import joblib
import pandas as pd
from datetime import timedelta
from config import BEST_MODEL_PATH, DEFAULT_TARIFF

FEATURE_COLUMNS = [
    "day",
    "month",
    "day_of_week",
    "day_of_year",
]


def load_best_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model missing: {BEST_MODEL_PATH}"
        )
    model = joblib.load(BEST_MODEL_PATH)
    return model


def predict_future(
    model,
    last_date,
    days=7,
    tariff=DEFAULT_TARIFF,
):
    """
    Generate future household electricity predictions using the trained Random Forest model.
    Ensures feature columns and preprocessing strictly match the original training pipeline.
    """
    if model is None:
        raise ValueError("A trained model is required.")

    days = int(days)
    if days < 1 or days > 365:
        raise ValueError("Forecast period must be between 1 and 365 days.")

    last_date = pd.Timestamp(last_date)

    dates = [
        last_date + timedelta(days=i)
        for i in range(1, days + 1)
    ]

    future = pd.DataFrame({
        "date": dates
    })

    future["day"] = future["date"].dt.day
    future["month"] = future["date"].dt.month
    try:
        future["day_of_week"] = future["date"].dt.day_of_week
    except AttributeError:
        future["day_of_week"] = future["date"].dt.dayofweek

    try:
        future["day_of_year"] = future["date"].dt.day_of_year
    except AttributeError:
        future["day_of_year"] = future["date"].dt.dayofyear

    # Extract exact features expected by the model
    features = future[FEATURE_COLUMNS]

    predictions = model.predict(features)

    # Ensure non-negative predicted energy
    future["predicted_energy_kwh"] = (
        pd.Series(predictions)
        .clip(lower=0.0)
        .values
    )

    future["estimated_cost"] = (
        future["predicted_energy_kwh"] * float(tariff)
    )

    future["day_name"] = future["date"].dt.day_name()
    future["cumulative_energy_kwh"] = future["predicted_energy_kwh"].cumsum()
    future["cumulative_cost"] = future["estimated_cost"].cumsum()

    return future


def calculate_electricity_cost(
    energy_kwh,
    tariff=DEFAULT_TARIFF,
):
    return round(
        float(energy_kwh) * float(tariff),
        2,
    )
