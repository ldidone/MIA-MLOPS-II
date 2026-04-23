#!/bin/bash
# Creates the logical databases used by Airflow and MLflow inside the shared
# Postgres instance. Runs once, on first container start (Postgres init hook).
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE airflow;
    CREATE DATABASE mlflow;
EOSQL
