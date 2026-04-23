FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements /app/requirements
RUN pip install --upgrade pip && pip install -r /app/requirements/api.txt

COPY pyproject.toml /app/
COPY conf /app/conf
COPY src /app/src
COPY scripts /app/scripts
COPY tests /app/tests

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=10 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "housing.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
