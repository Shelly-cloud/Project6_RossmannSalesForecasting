"""One-off sanity check of the raw CSVs before any pipeline is written."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import (
    GOOGLETREND_CSV,
    RAW_DTYPES,
    STATE_NAMES_CSV,
    STORE_CSV,
    STORE_STATES_CSV,
    TEST_CSV,
    TRAIN_CSV,
    WEATHER_CSV,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

train = pd.read_csv(TRAIN_CSV, dtype=RAW_DTYPES, parse_dates=["Date"], low_memory=False)
test = pd.read_csv(TEST_CSV, dtype=RAW_DTYPES, parse_dates=["Date"])
store = pd.read_csv(STORE_CSV)

print("=" * 70)
print("TRAIN", train.shape, train.Date.min().date(), "->", train.Date.max().date())
print("TEST ", test.shape, test.Date.min().date(), "->", test.Date.max().date())
print("STORE", store.shape)
print("test horizon days:", (test.Date.max() - test.Date.min()).days + 1)
print("stores train/test:", train.Store.nunique(), test.Store.nunique())
print()
print("train columns:", list(train.columns))
print("test  columns:", list(test.columns))
print("store columns:", list(store.columns))
print()
print("StateHoliday train:", sorted(train.StateHoliday.dropna().unique()))
print("StateHoliday test :", sorted(test.StateHoliday.dropna().unique()))
print()
print("nulls train:", train.isna().sum()[lambda s: s > 0].to_dict())
print("nulls test :", test.isna().sum()[lambda s: s > 0].to_dict())
print("nulls store:", store.isna().sum()[lambda s: s > 0].to_dict())
print()
print("closed rows:", int((train.Open == 0).sum()))
print("zero sales  :", int((train.Sales == 0).sum()))
print("zero sales while open:", int(((train.Sales == 0) & (train.Open == 1)).sum()))
print()
print("--- store.csv head ---")
print(store.head(3))
print()

for name, path in [
    ("store_states", STORE_STATES_CSV),
    ("state_names", STATE_NAMES_CSV),
    ("weather", WEATHER_CSV),
    ("googletrend", GOOGLETREND_CSV),
]:
    df = pd.read_csv(path)
    print(f"--- {name} {df.shape} ---")
    print("cols:", list(df.columns)[:20])
    print(df.head(3).to_string())
    print()
