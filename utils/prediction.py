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

    if model is None:
        raise ValueError("A trained model is required.")

    if days not in [7, 14, 30]:
        raise ValueError("Forecast days must be 7, 14 or 30.")

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

    future["day_of_week"] = (
        future["date"].dt.dayofweek
    )

    future["day_of_year"] = (
        future["date"].dt.dayofyear
    )

    features = future[FEATURE_COLUMNS]

    predictions = model.predict(features)

    future["predicted_energy_kwh"] = (
        pd.Series(predictions)
        .clip(lower=0)
        .values
    )

    future["estimated_cost"] = (
        future["predicted_energy_kwh"] * float(tariff)
    )

    return future


def calculate_electricity_cost(
    energy_kwh,
    tariff=DEFAULT_TARIFF,
):

    return round(
        float(energy_kwh) * float(tariff),
        2,
    )
