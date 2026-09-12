"""Task 3: Flask web interface for serving sales/customer predictions.

Routes
------
GET  /                 the dashboard: pick a store, upload a date-parameter CSV
POST /predict          run inference, show the chart + a results table
GET  /download/<token> download the last prediction as CSV
GET  /sample-csv       a template CSV with the columns the app expects
GET  /api/models       JSON list of saved model versions (for the MLflow screenshot story)
GET  /api/predict      JSON API equivalent of /predict, for programmatic use

Run with:  python app/app.py   (serves on http://127.0.0.1:5000)
"""
from __future__ import annotations

import base64
import io
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from app.predictor import list_store_ids, model_metadata, predict_for_store
from src import viz
from src.logger import get_logger

logger = get_logger("app")
viz.apply_theme()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload cap

# In-memory store of the last few predictions, keyed by a short token, so the
# chart page and the CSV download can share one computed result without a
# database. Fine for a single-analyst demo deployment; swap for a real store
# (Redis, a temp-file cache) before multi-user production use.
_RESULTS: dict[str, pd.DataFrame] = {}
_MAX_CACHED = 20


def _remember(df: pd.DataFrame) -> str:
    token = uuid.uuid4().hex[:12]
    _RESULTS[token] = df
    if len(_RESULTS) > _MAX_CACHED:
        _RESULTS.pop(next(iter(_RESULTS)))
    return token


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _make_chart(result: pd.DataFrame) -> str:
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(result["Date"], result["PredictedSales"], marker="o", markersize=3,
             color=viz.SERIES[0], label="Predicted sales")
    if "PredictedSalesLower" in result.columns:
        ax1.fill_between(result["Date"], result["PredictedSalesLower"],
                          result["PredictedSalesUpper"], color=viz.SERIES[0], alpha=0.15,
                          label="95% interval")
    viz.finish(ax1, "Forecast: predicted daily sales",
               "Shaded band = Random Forest tree-spread interval, where available",
               "Predicted sales", "Date")
    ax1.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_base64(fig)


def _validate_upload(df: pd.DataFrame) -> list[str]:
    errors = []
    if "Date" not in df.columns:
        errors.append("CSV must include a 'Date' column.")
        return errors
    try:
        pd.to_datetime(df["Date"])
    except (ValueError, TypeError):
        errors.append("The 'Date' column could not be parsed as dates.")
    for optional in ("IsHoliday", "IsWeekend", "IsPromo", "SchoolHoliday"):
        if optional not in df.columns:
            logger.info("upload missing optional column '%s'; defaulting to 0", optional)
    if len(df) == 0:
        errors.append("CSV has no rows.")
    if len(df) > 400:
        errors.append("CSV has more than 400 rows; please upload at most one forecast horizon.")
    return errors


@app.route("/")
def index():
    stores = list_store_ids()
    meta = model_metadata()
    return render_template("index.html", stores=stores, meta=meta)


@app.route("/predict", methods=["POST"])
def predict():
    stores = list_store_ids()
    try:
        store_id = int(request.form["store_id"])
    except (KeyError, ValueError):
        return render_template("index.html", stores=stores, meta=model_metadata(),
                                error="Please choose a valid store."), 400

    upload = request.files.get("csv_file")
    if upload is None or upload.filename == "":
        return render_template("index.html", stores=stores, meta=model_metadata(),
                                error="Please upload a CSV file with a Date column."), 400

    try:
        rows = pd.read_csv(upload)
    except Exception as exc:  # noqa: BLE001
        return render_template("index.html", stores=stores, meta=model_metadata(),
                                error=f"Could not read the CSV: {exc}"), 400

    errors = _validate_upload(rows)
    if errors:
        return render_template("index.html", stores=stores, meta=model_metadata(),
                                error=" ".join(errors)), 400

    try:
        result = predict_for_store(store_id, rows)
    except Exception as exc:  # noqa: BLE001
        logger.exception("prediction failed")
        return render_template("index.html", stores=stores, meta=model_metadata(),
                                error=f"Prediction failed: {exc}"), 500

    token = _remember(result)
    chart_b64 = _make_chart(result)

    table = result.copy()
    table["Date"] = pd.to_datetime(table["Date"]).dt.date
    table = table.round(1)

    total_sales = float(result["PredictedSales"].sum())
    total_customers = (
        int(result["PredictedCustomers"].sum()) if "PredictedCustomers" in result else None
    )

    return render_template(
        "results.html",
        store_id=store_id,
        chart_b64=chart_b64,
        rows=table.to_dict(orient="records"),
        columns=list(table.columns),
        token=token,
        total_sales=total_sales,
        total_customers=total_customers,
        n_days=len(result),
    )


@app.route("/download/<token>")
def download(token: str):
    result = _RESULTS.get(token)
    if result is None:
        return redirect(url_for("index"))
    buf = io.StringIO()
    result.to_csv(buf, index=False)
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(
        mem, mimetype="text/csv", as_attachment=True,
        download_name=f"rossmann_forecast_{token}.csv",
    )


@app.route("/sample-csv")
def sample_csv():
    """A ready-to-edit template covering the brief's 6-week forecast horizon."""
    dates = pd.date_range(start=pd.Timestamp.today().normalize(), periods=42, freq="D")
    template = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "IsHoliday": 0,
            "IsWeekend": (dates.dayofweek >= 5).astype(int),
            "IsPromo": 0,
            "SchoolHoliday": 0,
        }
    )
    buf = io.StringIO()
    template.to_csv(buf, index=False)
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(mem, mimetype="text/csv", as_attachment=True,
                      download_name="rossmann_forecast_template.csv")


@app.route("/api/models")
def api_models():
    from app.predictor import available_models

    return jsonify(available_models())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON equivalent of /predict: {"store_id": int, "rows": [{"Date": ..., ...}]}"""
    payload = request.get_json(force=True, silent=True) or {}
    store_id = payload.get("store_id")
    rows = payload.get("rows")
    if store_id is None or not rows:
        return jsonify({"error": "expected {'store_id': int, 'rows': [...]}"}), 400

    df = pd.DataFrame(rows)
    errors = _validate_upload(df)
    if errors:
        return jsonify({"error": " ".join(errors)}), 400

    try:
        result = predict_for_store(int(store_id), df)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    result = result.copy()
    result["Date"] = pd.to_datetime(result["Date"]).dt.strftime("%Y-%m-%d")
    return jsonify(result.to_dict(orient="records"))


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
