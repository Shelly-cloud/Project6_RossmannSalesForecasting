"""Cache the cleaned, feature-engineered training frame to data/processed/.

Running the full clean + feature pipeline takes real time; caching its output
also gives DVC a second, genuinely useful artifact to version alongside the
raw Kaggle CSVs (Task 1 deliverable: "multiple data versions in your DVC
store").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import clean_dataset
from src.config import DATA_PROCESSED
from src.data_loader import load_dataset
from src.features import build_holiday_calendar, engineer_features
from src.logger import get_logger

logger = get_logger("export_processed")

train_raw = load_dataset("train")
test_raw = load_dataset("test")
calendar = build_holiday_calendar(train_raw, test_raw)

clean = clean_dataset(train_raw, outlier_strategy="flag", for_training=True)
clean, _ = engineer_features(clean, holiday_calendar=calendar)

out_path = DATA_PROCESSED / "train_clean_engineered.parquet"
clean.to_parquet(out_path, index=False)
logger.info("wrote %s (%d rows, %d cols)", out_path, len(clean), clean.shape[1])
print(f"wrote {out_path} ({len(clean):,} rows, {clean.shape[1]} cols)")
