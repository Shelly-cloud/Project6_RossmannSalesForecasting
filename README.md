# Rossmann Store Sales Forecasting

**NextHikes IT Solutions** — Machine Learning Engineering project for Rossmann
Pharmaceuticals: forecast daily sales across 1,115 stores, six weeks ahead, and
serve the prediction through a web dashboard for the finance team.

## Project structure

```
RossmannSalesForecasting/
├── data/
│   ├── raw/              # train.csv, test.csv, store.csv + locality extras (DVC-tracked)
│   ├── processed/        # cached intermediate frames
│   └── external/
├── src/                  # the shared library -- every script and the app import from here
│   ├── config.py         # paths, schema constants, modelling constants
│   ├── logger.py         # Task 1.2 -- project-wide logging
│   ├── data_loader.py     # load + join train/test/store + weather/trend/state
│   ├── cleaning.py       # Task 1 -- missing-value & outlier pipelines
│   ├── features.py       # Task 2.1 -- feature engineering transformers
│   ├── pipeline.py       # Task 2.2 -- the end-to-end sklearn Pipeline
│   ├── metrics.py        # Task 2.3 -- RMSPE loss + evaluation bundle
│   ├── confidence.py     # Task 2.4 -- feature importance + prediction intervals
│   ├── serialize.py      # Task 2.5 -- timestamped model save/load
│   ├── lstm_model.py     # Task 2.6 -- LSTM time-series forecaster
│   ├── mlflow_utils.py   # Task 2.7 -- MLflow tracking setup
│   ├── eda.py            # Task 1 -- one function per exploratory question
│   └── viz.py            # shared, accessibility-checked chart theme
├── scripts/
│   ├── run_eda.py        # Task 1 runner -> reports/eda_findings.md + figures
│   ├── train_ml.py       # Task 2 runner -> RandomForest + LightGBM, MLflow, serialized models
│   ├── train_lstm.py     # Task 2.6 runner -> LSTM, MLflow, serialized model
│   └── smoke_test.py     # fast end-to-end pipeline check on a store subsample
├── app/                  # Task 3 -- Flask web dashboard
│   ├── app.py
│   ├── predictor.py      # loads the latest serialized pipeline, runs inference
│   ├── templates/
│   └── static/
├── notebooks/
│   └── Rossmann_Sales_Forecasting.ipynb   # the required Jupyter notebook deliverable
├── models/                # timestamped .pkl / .keras + .json metadata (Task 2.5)
├── mlruns/                # local MLflow tracking store (SQLite + artifacts)
├── reports/
│   ├── eda_findings.md    # Task 1 write-up
│   └── figures/           # every chart, PNG
├── requirements.txt
├── Procfile               # for gunicorn-based deployment (Render / Railway / etc.)
└── logs/rossmann.log      # full pipeline audit trail
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Data is **not checked into git** — it's pulled from the public fast.ai mirror of the
Kaggle Rossmann Store Sales competition and tracked with DVC:

```bash
dvc pull        # if a remote is configured
# or, to fetch from scratch:
curl -o data/raw/rossmann.tgz http://files.fast.ai/part2/lesson14/rossmann.tgz
tar -xzf data/raw/rossmann.tgz -C data/raw
```

This gives `train.csv`, `test.csv`, `store.csv` (the three Kaggle files) plus three
locality extras used for the "locality" factor the brief calls out: `store_states.csv`,
`weather.csv`, `googletrend.csv`.

## Running the pipeline

```bash
python scripts/run_eda.py      # Task 1: cleaning, feature engineering, 14 analyses
python scripts/train_ml.py     # Task 2.1-2.5, 2.7: RandomForest + LightGBM, MLflow, serialize
python scripts/train_lstm.py   # Task 2.6: LSTM forecaster
```

Each writes to `reports/`, `models/` and `mlruns/`, and logs every step to
`logs/rossmann.log`.

### MLflow dashboard

```bash
mlflow ui --backend-store-uri "sqlite:///mlruns/mlflow.db" --port 5000
```

Open `http://127.0.0.1:5000` to see every run (RandomForest, LightGBM, LSTM) with its
parameters, metrics, feature-importance artefacts and serialized-model path.

### DVC data versioning

```bash
dvc init
dvc add data/raw/train.csv data/raw/test.csv data/raw/store.csv
git add data/raw/*.dvc .gitignore
git commit -m "Track raw Rossmann data with DVC"
```

Re-running `dvc add` after any data change (e.g. a refreshed extract) creates a new
tracked revision — `dvc log` / the `.dvc/cache` directory is what the interim
submission's "multiple data versions" screenshot should show.

## The web dashboard (Task 3)

```bash
python app/app.py
```

Open `http://127.0.0.1:5000`, choose a store, and upload a CSV with a `Date` column
(optionally `IsHoliday`, `IsWeekend`, `IsPromo`, `SchoolHoliday`) — a template is
available from the dashboard's "Download a 6-week template CSV" link. The results page
shows a chart of predicted sales (with a confidence band, when the underlying model is
the Random Forest) plus an estimated customer count, and a CSV download of the full
forecast table.

### Deployment

Heroku's free tier no longer exists; the `Procfile` here targets any gunicorn-compatible
host (Render, Railway, Fly.io):

```bash
web: gunicorn --chdir . --pythonpath . -w 2 -b 0.0.0.0:$PORT app.app:app
```

1. Push this repo to GitHub.
2. On Render (or similar): create a new **Web Service**, point it at the repo, build
   command `pip install -r requirements.txt`, start command from the `Procfile`.
3. Make sure `models/*.pkl` (or a small representative subset) and `mlruns/` are either
   committed or regenerated by a build step — the `.gitignore` here excludes them by
   default since they're large binaries; adjust before deploying if you want the trained
   model to ship with the repo rather than be trained at deploy time.

## Design notes worth knowing before reading the code

- **`Customers` is dropped from every feature set.** It's in `train.csv` but not
  `test.csv` — using it would produce a great validation score and a model that cannot
  be deployed. The dashboard estimates customers separately from each store's own
  learned sales-per-customer ratio.
- **Closed days (`Open == 0`) are excluded from training**, not learned. They always
  have `Sales == 0`; `Sales = 0` is imposed as a deterministic rule at inference instead.
- **Validation is the final 6 weeks of the training window**, not a random split —
  matching the brief's actual forecast horizon and avoiding leakage of each store's own
  future statistics into its past.
- **RMSPE was chosen as the primary loss** specifically because store size varies
  sixfold across the estate (see Q12 in `reports/eda_findings.md`); a scale-sensitive
  loss like RMSE would let the largest stores dominate the objective. See
  `src/metrics.py` for the full defence.

## Attribution

Data: [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle),
mirrored by [fast.ai](http://files.fast.ai/part2/lesson14/rossmann.tgz).
