# Hugging Face Spaces (sdk: docker) expects this file at the repository root.
# Kept in sync with docker/streamlit-spaces.Dockerfile — edit both when changing the Space image.
#
# Streamlit image for Hugging Face Spaces (sdk: docker).
# Build context must be the repository root so ``COPY conf`` / ``COPY src`` work.
#
# It runs the app in ``embedded`` mode: load ``models/model.pkl`` + feature metadata
# from disk (no MLflow / API). Train locally or in docker-compose first, then commit
# ``models/model.pkl`` and ``models/feature_metadata.json`` (use Git LFS if large).
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    APP_MODE=embedded \
    LOCAL_MODEL_PATH=/app/models/model.pkl

WORKDIR /app

COPY requirements/base.txt requirements/streamlit.txt /app/requirements/
RUN pip install --upgrade pip \
    && pip install -r /app/requirements/streamlit.txt

COPY pyproject.toml /app/
COPY conf /app/conf
COPY src /app/src
COPY models /app/models

EXPOSE 7860

CMD ["streamlit", "run", "src/housing/streamlit_app/app.py", "--server.port=7860"]
