import pandas as pd
from config import DATASET_PATH
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. "
            "Place household_power_consumption.txt inside the data folder."
        )
    df = pd.read_csv(
        DATASET_PATH,
        sep=";",
        na_values=["?", ""],
        low_memory=False,
    )
    # Clean column names and remove BOM if present.
    df.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in df.columns
    ]
    required_columns = [
        "Date",
        "Time",
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str)
        + " "
        + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    numeric_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "datetime",
            "Global_active_power",
        ]
    )

    df = df.sort_values("datetime")

    return df.reset_index(drop=True)
def prepare_daily_data(df):
    if df is None or df.empty:
        raise ValueError("Cannot prepare daily data from an empty dataset.")
    frame = df.copy()
    # Global active power is measured in kW.
    # Each source record represents one minute.
    frame["energy_kwh"] = (
        frame["Global_active_power"] / 60.0
    )
    daily = (
        frame.set_index("datetime")["energy_kwh"]
        .resample("D")
        .sum()
        .rename("energy_kwh")
        .reset_index()
    )
    daily = daily.rename(
        columns={"datetime": "date"}
    )

    daily = daily.dropna(
        subset=["energy_kwh"]
    )

    daily["day"] = daily["date"].dt.day

    daily["month"] = daily["date"].dt.month

    daily["day_of_week"] = (
        daily["date"].dt.dayofweek
    )

    daily["day_of_year"] = (
        daily["date"].dt.dayofyear
    )

    return daily.reset_index(drop=True)
