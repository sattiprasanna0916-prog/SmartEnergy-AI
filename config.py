import os
from pathlib import Path
from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent
# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")
# Dataset and model paths
DATASET_PATH = (
    BASE_DIR / "data" / "household_power_consumption.txt"
)
MODEL_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
# MySQL configuration
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "Root@123")
MYSQL_DATABASE = os.getenv(
    "MYSQL_DATABASE",
    "smartenergy_db",
)
# Electricity tariff
DEFAULT_TARIFF = float(
    os.getenv("DEFAULT_TARIFF", "6.0")
)
