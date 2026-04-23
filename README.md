# MIA-MLOPS-II — California Housing MLOps Project

End-to-end, container-first MLOps project for the MLOps II course (MIA master's degree). It trains, tracks, serves and consumes a **regression** model for scikit-learn's [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). The target is `MedHouseVal` — median house value per block group, in units of $100,000 — and the feature schema is auto-discovered from the ingested CSV, so the pipeline does not rely on hard-coded column names.

The project implements the containerised level of the assignment described for **ML Models and Something More Inc.**, focused on the baseline stack:

> Apache Airflow · MLflow · PostgreSQL · MinIO · FastAPI · Docker.

A Streamlit frontend is included as the user-facing application and is also packaged to be deployable as a Docker Space on Hugging Face.

---

## Table of contents

1. [Overview & architecture](#overview--architecture)
2. [Directory structure](#directory-structure)
3. [Setup](#setup)
4. [Running the stack with Docker Compose](#running-the-stack-with-docker-compose)
5. [Running the training pipeline](#running-the-training-pipeline)
6. [Using the API](#using-the-api)
7. [Using the Streamlit app](#using-the-streamlit-app)
8. [Deploying the Streamlit app to Hugging Face Spaces](#deploying-the-streamlit-app-to-hugging-face-spaces)
9. [Design decisions](#design-decisions)
10. [Assignment coverage vs. future work](#assignment-coverage-vs-future-work)
11. [Testing & linting](#testing--linting)

---

## Overview & architecture

### Components

| Service | Purpose | Port |
| --- | --- | --- |
| `postgres` | Backend DB for Airflow and MLflow (two logical databases) | 5433 (host) → 5432 (container) |
| `minio` | S3-compatible data lake + MLflow artifact store | 9000 (S3), 9001 (console) |
| `minio-init` | One-shot bucket creation | — |
| `mlflow` | Tracking server (backend: Postgres, artifacts: MinIO) | 5001 (host) → 5000 (container) |
| `airflow-init` | One-shot DB migrations + admin user creation | — |
| `airflow-webserver` | Airflow UI | 8080 |
| `airflow-scheduler` | DAG scheduler (`LocalExecutor`) | — |
| `api` | FastAPI inference service | 8000 |
| `streamlit` | Streamlit frontend calling the API | 8501 |

### End-to-end flow

```mermaid
flowchart LR
    SK["scikit-learn<br/>fetch_california_housing"] --> AF[Airflow DAG]
    AF -->|raw csv| MinIO[("MinIO<br/>s3://data-lake")]
    AF -->|train + log| MLflow[MLflow Tracking]
    MLflow -->|artifacts| MinIO
    MLflow -->|metadata| PG[(Postgres)]
    AF -->|set @champion alias| MLflow
    API[FastAPI] -->|models:/name@champion| MLflow
    UI[Streamlit] -->|POST /predict| API
    HFSpaces["HF Spaces<br/>(Docker)"] -.->|embedded mode| UI
```

### Training pipeline (Airflow DAG `california_housing_training`)

```
ingest → validate → preprocess → train → register
```

* `ingest` calls `sklearn.datasets.fetch_california_housing(as_frame=True)`, writes the dataframe to `data/raw/california_housing.csv` and uploads it to `s3://data-lake/raw/california_housing.csv`. No external credentials are required — the dataset ships with scikit-learn.
* `validate` runs lightweight, schema-agnostic checks (min row count, target present, numeric, non-null, non-constant) and fails fast if the dataset is corrupted.
* `preprocess` builds random train/val/test splits and persists them to `data/processed/` as parquet.
* `train` fits three estimators (`LinearRegression`, `RandomForestRegressor`, `XGBRegressor`) as `Pipeline(preprocessor, estimator)`, each logged as a nested MLflow run with parameters, metrics, the regression diagnostic report, predicted-vs-actual and residual PNGs, and the full sklearn model. 5-fold cross-validation scored with R² is used to compare models robustly.
* `register` picks the run with the highest `cv_score_mean`, registers the model as `california_housing_regressor` and moves the `champion` alias onto the new version. The API picks up new versions on the next reload.

### Inference flow

```
Streamlit form → FastAPI /predict → sklearn Pipeline (preprocessor + regressor) → float prediction
```

Because the preprocessor is embedded inside the logged sklearn `Pipeline`, the exact same feature transformation used at training time is applied at inference time — no risk of skew.

---

## Directory structure

```
MIA-MLOPS-II/
├── README.md
├── .env.example
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements/
│   ├── base.txt
│   ├── api.txt
│   ├── streamlit.txt
│   └── airflow-extra.txt
├── docker/
│   ├── api.Dockerfile
│   ├── streamlit.Dockerfile
│   ├── streamlit-spaces.Dockerfile
│   ├── mlflow.Dockerfile
│   ├── airflow.Dockerfile
│   └── postgres-init.sh
├── conf/
│   └── config.yaml
├── data/
│   ├── raw/      (.gitkeep)
│   ├── interim/  (.gitkeep)
│   ├── processed/(.gitkeep)
│   └── external/ (.gitkeep)
├── models/       (.gitkeep — champion pickle fallback ends up here)
├── notebooks/    (optional EDA)
├── reports/figures/
├── src/housing/
│   ├── __init__.py
│   ├── config.py
│   ├── data/        (ingest.py, validate.py)
│   ├── features/    (preprocess.py)
│   ├── models/      (train.py, evaluate.py, register.py, predict.py)
│   ├── api/         (main.py, schemas.py, model_loader.py)
│   ├── streamlit_app/(app.py, api_client.py)
│   └── utils/       (logging.py, mlflow_utils.py)
├── airflow/
│   ├── dags/housing_training_dag.py
│   ├── plugins/.gitkeep
│   └── logs/
├── scripts/
│   ├── run_training_local.py
│   └── bootstrap_minio.py
└── tests/
    ├── conftest.py
    ├── test_preprocess.py
    ├── test_schemas.py
    └── test_api.py
```

The layout follows [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) while adding production-oriented folders (`docker/`, `airflow/`, `scripts/`).

---

## Setup

### Prerequisites

* Docker Desktop (≥ 6 GB RAM allocated, recommended 8 GB).
* `docker compose` v2.20+.
* macOS / Linux / WSL2.

No Python installation is required on the host: everything runs inside containers. **No external credentials are required**: the dataset ships with scikit-learn.

### Clone and bootstrap env

```bash
git clone <this-repo>
cd MIA-MLOPS-II
cp .env.example .env
```

The defaults for Postgres/MinIO/MLflow work out of the box. Review `.env` if you need to change ports or credentials.

### Inspect the ingested schema

The feature schema of the dataset is not hard-coded anywhere. After bringing the stack up (see below) you can confirm which columns you actually have — plus correlations with the target — with:

```bash
docker compose exec api python -m scripts.inspect_dataset
```

This prints the dataset shape, each column's dtype / uniques / sample values, the target summary statistics, per-feature Pearson correlation with the target, and the numeric/categorical/binary split that the preprocessor will use.

---

## Running the stack with Docker Compose

```bash
docker compose up -d --build

# Wait ~60-90s for all healthchecks to turn green.
docker compose ps
```

Once everything is healthy:

| UI | URL | Credentials |
| --- | --- | --- |
| Airflow | http://localhost:8080 | `airflow` / `airflow` |
| MLflow | http://localhost:5001 | — |
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| FastAPI Swagger | http://localhost:8000/docs | — |
| Streamlit | http://localhost:8501 | — |

### Stopping

```bash
docker compose down           # keep volumes (data, models, mlflow)
docker compose down -v        # nuke everything
```

---

## Running the training pipeline

### Option A — through Airflow (recommended)

1. Open http://localhost:8080.
2. Un-pause the `california_housing_training` DAG.
3. Click **Trigger DAG**.
4. Follow the tasks in the *Grid* view — `ingest`, `validate`, `preprocess`, `train`, `register`.
5. After `train`, visit MLflow at http://localhost:5001 to inspect every run; the parent run is named `training_pipeline` and each child run corresponds to one model family.
6. After `register`, the API automatically picks up the new champion on its next `/admin/reload` call (or on restart).

Alternatively, trigger the DAG from the CLI:

```bash
make train
# equivalent to:
docker compose exec airflow-scheduler airflow dags trigger california_housing_training
```

### Option B — run the pipeline inside the API container (bypass Airflow)

Useful for rapid iteration or grading without starting Airflow:

```bash
make train-local
# equivalent to:
docker compose exec api python -m scripts.run_training_local
```

It ingests, validates, trains, evaluates and registers the champion, plus dumps a `models/model.pkl` pickle used as a fallback by the API.

### Re-loading the API after training

```bash
curl -X POST http://localhost:8000/admin/reload
# or
docker compose restart api
```

---

## Using the API

FastAPI exposes:

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Service info |
| GET | `/health` | Health + model-loaded flag |
| GET | `/model/info` | Model metadata (registry version, feature count, target stats) |
| POST | `/predict` | Single-record prediction |
| POST | `/predict/batch` | Batch prediction (up to 1000 records) |
| POST | `/admin/reload` | Force a model reload from the registry |

### Example request

The API accepts a flexible `{"features": {...}}` envelope. The exact keys must match the columns the model was trained on — fetch them from `GET /model/info`:

```bash
curl -s http://localhost:8000/model/info | jq .feature_names
# -> ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]
```

Then send a prediction request with those exact keys:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "features": {
      "MedInc": 5.3,
      "HouseAge": 28,
      "AveRooms": 6.1,
      "AveBedrms": 1.0,
      "Population": 1200,
      "AveOccup": 2.8,
      "Latitude": 34.2,
      "Longitude": -118.3
    }
  }'
```

Response shape (target is in units of $100,000):

```json
{
  "predicted_value": 2.734
}
```

Validation is enforced at two layers:

1. Pydantic checks the envelope (dict under `features`, batch size, etc.).
2. The API cross-checks the keys against the feature list the model was trained on and returns a clear HTTP 422 with the missing column names if a feature is absent.

---

## Using the Streamlit app

Open http://localhost:8501. The form is **generated dynamically** from the feature metadata that training writes to `models/feature_metadata.json`, so it automatically adapts to whatever columns the dataset has. The UI shows:

* the predicted target value formatted as US dollars (for the default `MedHouseVal` target, predictions are automatically converted `value × $100,000`);
* training-set summary statistics (min / mean / median / max) for context;
* the raw JSON payload sent to the API (expandable).

The app reads `APP_MODE` from the environment:

* `APP_MODE=api` (default, used in docker-compose): calls FastAPI at `API_URL`.
* `APP_MODE=embedded`: loads `models/model.pkl` directly — used on Hugging Face Spaces.

---

## Deploying the Streamlit app to Hugging Face Spaces

The file [`docker/streamlit-spaces.Dockerfile`](docker/streamlit-spaces.Dockerfile) is a self-contained image that:

* runs in `embedded` mode,
* listens on port 7860 (required by Spaces),
* bundles the champion pickle from `models/model.pkl`.

### Steps

1. Train the model at least once (see [training pipeline](#running-the-training-pipeline)) so `models/model.pkl` exists.
2. Create a new **Docker Space** on Hugging Face.
3. Push the project (or a minimal subset: `src/`, `conf/`, `models/`, `requirements/`, `docker/streamlit-spaces.Dockerfile`) to the Space.
4. Rename or symlink `docker/streamlit-spaces.Dockerfile` to `Dockerfile` at the Space root, or configure it explicitly via the Space settings.
5. The Space will build the image and expose the Streamlit UI at `https://<user>-<space>.hf.space`.

---

## Design decisions

### Regression (continuous target)

`MedHouseVal` is a continuous variable, so we model it as regression:

* Primary CV score: **R²** — unitless, comparable across models, and directly interpretable as "fraction of variance explained".
* Test metrics also logged: **MAE** (mean absolute error in target units), **RMSE** (penalises large misses harder), **MAPE** (percentage error).
* Diagnostic artifacts per run: a **predicted-vs-actual** scatter and a **residuals-vs-predicted** plot, both logged to MLflow.

Expected performance out of the box: R² ≈ 0.60 for linear regression and R² ≈ 0.80+ for the tree ensembles (RandomForest, XGBoost), which is typical for this dataset.

### `Pipeline(preprocessor, estimator)` everywhere

The preprocessor is *inside* the sklearn `Pipeline` and therefore part of the logged MLflow artifact. The API loads the whole pipeline and feeds raw feature rows to it, guaranteeing that training and inference use identical transformations.

### `LocalExecutor` instead of Celery

`LocalExecutor` + Postgres covers the assignment's orchestration requirement without adding Redis and workers to the stack. For a production workload at real scale, `CeleryExecutor` would be substituted.

### MinIO as a single S3 endpoint

Both the data lake (`s3://data-lake/raw/...`) and MLflow artifact store (`s3://mlflow-artifacts/...`) live in the same MinIO instance. One infra component, two buckets, minimal complexity.

### Model Registry alias instead of stages

MLflow's `Stages` API is deprecated. We use the modern alias API (`models:/name@champion`) so the API loads `models:/california_housing_regressor@champion` and picks up new versions automatically.

### Graceful degradation in the API

The API starts successfully even if MLflow is not reachable or the model has not been trained yet. `/health` reports `degraded`, and `store.load()` retries MLflow three times before falling back to the local pickle. This avoids the classic "chicken-and-egg" docker-compose boot ordering problem.

### Schema-agnostic preprocessing

The concrete column layout of the dataset is not hard-coded. `features.preprocess.infer_feature_groups` inspects the dataframe at training time and classifies each column as **numeric**, **binary** (values in `{0, 1}`) or **categorical**, then builds the matching `ColumnTransformer`. The result is persisted to `models/feature_metadata.json`, which the FastAPI service uses to validate payloads and the Streamlit app uses to render the input form.

California Housing is entirely numeric (8 features), so the default behaviour is: median imputation + `StandardScaler`. Swap to any other regression dataset and the same code will add one-hot encoding, binary passthrough, etc., as needed.

You can override the auto-inference by listing columns explicitly under `features.numeric` / `features.categorical` / `features.binary` in `conf/config.yaml`. If the target happens to live on a suspicious column (e.g. a derived "value" label), add it to `dataset.drop_columns` to prevent leakage.

---

## Assignment coverage vs. future work

### Implemented ✅

* **Cookiecutter Data Science** project structure.
* **Data pipeline**: ingest (scikit-learn → local CSV → MinIO), lightweight validation (row count, target presence, numeric, non-null, non-constant), preprocess (median imputation + StandardScaler + OneHot in a schema-agnostic `ColumnTransformer`).
* **Experimentation**: LinearRegression, RandomForestRegressor, XGBRegressor — each evaluated with 5-fold CV on R².
* **Experiment tracking**: MLflow — parent + nested runs, parameters, metrics (R²/MAE/RMSE/MAPE), regression report, predicted-vs-actual + residual plots, sklearn artifacts, input examples and signatures.
* **Model selection + artifact management**: best run chosen by `cv_score_mean`, registered, `champion` alias moved atomically.
* **Orchestration**: Airflow DAG with five sequential tasks.
* **Model serving (REST)**: FastAPI with Pydantic schemas, `/health`, `/model/info`, `/predict`, `/predict/batch`, `/admin/reload`, CORS, robust lifespan loader.
* **Frontend**: Streamlit app with API and embedded modes; target-aware formatting (dollars for `MedHouseVal`).
* **Containers**: Docker Compose with PostgreSQL, MinIO, MLflow, Airflow (LocalExecutor), FastAPI, Streamlit. Healthchecks on every long-running service.
* **Documentation**: this README, inline docstrings, `.env.example`, Makefile targets.

### Partially implemented ⚠️

* **Security layers**: CORS middleware and schema validation are in place. An API key / JWT auth dependency is sketched but not enforced (easy to wire via `fastapi.security`).

### Left as future work ❌

The assignment statement references these for the broader federated/cloud service scenario. They are out of scope for the containerised level and not required by the rubric:

* **GraphQL / gRPC / Streaming endpoints** — the service currently exposes only REST. Alternatives could be added as additional FastAPI routers (Strawberry for GraphQL, `grpcio` for gRPC, `fastapi-mqtt` or Kafka for streaming).
* **Federated learning** — would require multiple training clients and a secure aggregation server (`Flower`, `TFF`). The current Airflow DAG is the centralised counterpart.
* **Full security layers** — JWT auth, secrets manager, TLS termination, RBAC. Hooks are in place (environment variables, CORS) but the production-grade implementation is not delivered here.
* **Drift monitoring / model performance monitoring** in production — a natural extension is to add an Evidently report task to the DAG and a Prometheus exporter on the API.
* **CI/CD** — no GitHub Actions workflow is shipped. `make lint` and `make test` cover local quality gates.

---

## Testing & linting

### Inside Docker

```bash
make test              # pytest inside the api container
docker compose exec api pytest -v
```

### On the host (optional)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/api.txt pytest
PYTHONPATH=src pytest
```

### Lint

```bash
pip install ruff
ruff check src tests scripts
```

---

## Useful commands summary

```bash
make up          # docker compose up -d --build
make down        # docker compose down
make logs        # tail logs
make train       # trigger Airflow DAG
make train-local # bypass Airflow
make test        # run pytest
make lint        # ruff
```

---

## Dataset credit

California Housing dataset shipped with scikit-learn. Originally from:

> Pace, R. K. and Barry, R., *Sparse spatial auto-regressions*, Statistics & Probability Letters, 1997.

See the scikit-learn [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) for full provenance.

## License

MIT — see [`pyproject.toml`](pyproject.toml).
