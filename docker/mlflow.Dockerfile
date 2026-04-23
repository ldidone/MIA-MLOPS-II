FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install \
    mlflow==2.13.2 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.131

EXPOSE 5000

HEALTHCHECK --interval=15s --timeout=5s --retries=10 \
    CMD curl -fsS http://localhost:5000/ || exit 1

# Backend URI + artifact root come from env vars set in docker-compose.
CMD ["sh", "-c", "mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT} \
    --serve-artifacts"]
