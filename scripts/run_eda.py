"""Task 1 runner: load -> clean -> engineer -> run every analysis -> write report.

Produces:
  reports/figures/*.png      one chart per analysis
  reports/eda_findings.md    the written insights, in report order
  logs/rossmann.log          the full audit trail (Task 1.2)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window

from src import eda
from src.cleaning import clean_dataset, missing_value_report, outlier_report
from src.config import REPORTS_DIR
from src.data_loader import load_dataset
from src.features import build_holiday_calendar, engineer_features
from src.logger import get_logger

logger = get_logger("run_eda")

logger.info("=" * 70)
logger.info("TASK 1 - EXPLORATION OF CUSTOMER PURCHASING BEHAVIOUR")
logger.info("=" * 70)

train_raw = load_dataset("train")
test_raw = load_dataset("test")

# Pre-cleaning data-quality record, referenced by q14.
logger.info("missing values before cleaning:\n%s", missing_value_report(train_raw).to_string())

clean = clean_dataset(train_raw, outlier_strategy="flag", for_training=True)
calendar = build_holiday_calendar(train_raw, test_raw)
clean, _ = engineer_features(clean, holiday_calendar=calendar)

worst = outlier_report(clean).head(5)
logger.info("top-5 stores by max sales z-score:\n%s", worst.to_string())

findings = eda.run_all(clean, train_raw, test_raw)

# ---- write the findings report -------------------------------------------
out = REPORTS_DIR / "eda_findings.md"
lines = [
    "# Task 1 - Exploration of Customer Purchasing Behaviour",
    "",
    "Rossmann Pharmaceuticals sales forecasting | NextHikes IT Solutions",
    "",
    f"Dataset: {len(train_raw):,} raw training rows across "
    f"{train_raw.Store.nunique():,} stores, "
    f"{train_raw.Date.min().date()} to {train_raw.Date.max().date()}. "
    f"After removing closed and zero-sales days, {len(clean):,} trading rows remain.",
    "",
    "Each section states the question, the evidence table, the chart and the "
    "conclusion drawn. Charts are in `reports/figures/`.",
    "",
    "---",
    "",
]

for i, f in enumerate(findings, 1):
    lines.append(f"## {i}. {f.question}")
    lines.append("")
    if f.figure_path:
        rel = Path(f.figure_path).relative_to(REPORTS_DIR).as_posix()
        lines.append(f"![{f.key}]({rel})")
        lines.append("")
    if f.table is not None:
        lines.append("```")
        lines.append(f.table.to_string())
        lines.append("```")
        lines.append("")
    lines.append(f"**Insight.** {f.insight}")
    lines.append("")
    lines.append("---")
    lines.append("")

out.write_text("\n".join(lines), encoding="utf-8")
logger.info("wrote %s (%d findings)", out, len(findings))

print(f"\nTASK 1 COMPLETE")
print(f"  findings : {len(findings)}")
print(f"  report   : {out}")
print(f"  figures  : {len([f for f in findings if f.figure_path])} PNGs in reports/figures")
for f in findings:
    print(f"    - {f.key}")
