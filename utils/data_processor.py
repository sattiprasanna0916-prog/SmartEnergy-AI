import pandas as pd
from config import DATASET_PATH, DEFAULT_TARIFF


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
    try:
        daily["day_of_week"] = daily["date"].dt.day_of_week
    except AttributeError:
        daily["day_of_week"] = daily["date"].dt.dayofweek

    try:
        daily["day_of_year"] = daily["date"].dt.day_of_year
    except AttributeError:
        daily["day_of_year"] = daily["date"].dt.dayofyear

    return daily.reset_index(drop=True)


def prepare_monthly_data(daily_df: pd.DataFrame, tariff: float = DEFAULT_TARIFF) -> pd.DataFrame:
    """
    Aggregate daily electricity consumption into a monthly summary dataframe.
    Computes total consumption, average daily consumption, estimated bills,
    and month-over-month percentage changes, with robust handling for missing/partial months.
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Resample by Month End
    grouped = df.set_index("date")
    monthly_sum = grouped["energy_kwh"].resample("ME").sum()
    monthly_mean = grouped["energy_kwh"].resample("ME").mean()
    monthly_count = grouped["energy_kwh"].resample("ME").count()

    monthly = pd.DataFrame({
        "date": monthly_sum.index,
        "energy_kwh": monthly_sum.values,
        "avg_daily_kwh": monthly_mean.values,
        "days_recorded": monthly_count.values,
    })

    # Filter out months that had zero recorded days (handle missing periods cleanly)
    monthly = monthly[monthly["days_recorded"] > 0].copy()

    monthly["month_str"] = monthly["date"].dt.strftime("%b %Y")
    monthly["year_month"] = monthly["date"].dt.strftime("%Y-%m")
    monthly["estimated_bill"] = monthly["energy_kwh"] * float(tariff)

    # Calculate month-over-month differences
    monthly["prev_energy_kwh"] = monthly["energy_kwh"].shift(1)
    monthly["mom_kwh_change"] = monthly["energy_kwh"] - monthly["prev_energy_kwh"]

    # Calculate % increase or decrease safely
    def _safe_pct_change(row):
        prev = row["prev_energy_kwh"]
        if pd.isna(prev) or prev <= 0:
            return 0.0
        return ((row["energy_kwh"] - prev) / prev) * 100.0

    monthly["mom_pct_change"] = monthly.apply(_safe_pct_change, axis=1)
    monthly["prev_bill"] = monthly["estimated_bill"].shift(1)
    monthly["mom_bill_change"] = monthly["estimated_bill"] - monthly["prev_bill"]

    return monthly.reset_index(drop=True)


def detect_iqr_outliers(daily_df: pd.DataFrame, multiplier: float = 1.5):
    """
    Detect unusually high-consumption days using the standard statistical IQR (Interquartile Range) method.
    Q1: 25th percentile
    Q3: 75th percentile
    IQR: Q3 - Q1
    Upper Threshold: Q3 + (multiplier * IQR)

    Returns:
    - stats: Dictionary containing summary stats (mean, median, q1, q3, iqr, threshold, count)
    - outliers_df: DataFrame of identified unusual days with deviation metrics
    """
    if daily_df is None or daily_df.empty:
        return {
            "mean": 0.0,
            "median": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "threshold": 0.0,
            "outlier_count": 0,
            "total_days": 0,
        }, pd.DataFrame()

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
    df = df.dropna(subset=["energy_kwh"])
    df = df[df["energy_kwh"] >= 0]

    q1 = float(df["energy_kwh"].quantile(0.25))
    q3 = float(df["energy_kwh"].quantile(0.75))
    iqr = q3 - q1
    upper_threshold = q3 + (multiplier * iqr)
    mean_val = float(df["energy_kwh"].mean())
    median_val = float(df["energy_kwh"].median())

    df["is_abnormal"] = df["energy_kwh"] > upper_threshold

    outliers = df[df["is_abnormal"]].copy()
    outliers["excess_kwh"] = (outliers["energy_kwh"] - mean_val).round(2)
    outliers["threshold_excess"] = (outliers["energy_kwh"] - upper_threshold).round(2)
    outliers["ratio_to_mean"] = (outliers["energy_kwh"] / mean_val).round(2)
    outliers = outliers.sort_values("energy_kwh", ascending=False).reset_index(drop=True)

    stats = {
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "threshold": round(upper_threshold, 2),
        "outlier_count": len(outliers),
        "total_days": len(df),
    }

    return stats, outliers
