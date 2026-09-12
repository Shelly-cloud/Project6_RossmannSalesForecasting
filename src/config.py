"""Central configuration: paths, column groups and modelling constants.

Everything that another module might need to know about *where* things live or
*what* a column means belongs here, so no other module hard-codes a path.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = ROOT / "data" / "external"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLRUNS_DIR = ROOT / "mlruns"

TRAIN_CSV = DATA_RAW / "train.csv"
TEST_CSV = DATA_RAW / "test.csv"
STORE_CSV = DATA_RAW / "store.csv"
STORE_STATES_CSV = DATA_RAW / "store_states.csv"
STATE_NAMES_CSV = DATA_RAW / "state_names.csv"
WEATHER_CSV = DATA_RAW / "weather.csv"
GOOGLETREND_CSV = DATA_RAW / "googletrend.csv"

for _d in (DATA_PROCESSED, DATA_EXTERNAL, MODELS_DIR, LOGS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- schema
TARGET = "Sales"
SECONDARY_TARGET = "Customers"  # predicted separately for the dashboard

# Kaggle ships StateHoliday with mixed "0" (str) and 0 (int) values. Forcing
# str at read time stops pandas from silently creating two distinct categories.
RAW_DTYPES = {"StateHoliday": str}

# Columns that must never reach the model: Customers is unknown at forecast
# time, Sales is the target, Date is expanded into features, Id is a row key.
LEAKAGE_COLS = ["Customers", "Sales", "Date", "Id"]

CATEGORICAL_FEATURES = [
    "StoreType",
    "Assortment",
    "StateHoliday",
    "PromoInterval",
    "State",
]

# ---------------------------------------------------------------- modelling
RANDOM_STATE = 42

# The brief asks for a 6-week-ahead forecast, and the Kaggle test set is
# exactly 48 days. Validating on the final 6 weeks of train therefore mirrors
# the real forecast horizon far better than a random split would.
VALIDATION_WEEKS = 6
FORECAST_HORIZON_DAYS = 42

# Sales below this are treated as "store effectively shut" and excluded from
# training: they carry no demand signal and would drag every prediction down.
MIN_TRAINABLE_SALES = 1

MODEL_TIMESTAMP_FMT = "%d-%m-%Y-%H-%M-%S-%f"

MLFLOW_EXPERIMENT = "rossmann-sales-forecasting"
