# Variant of the Streamlit image targeting Hugging Face Spaces (Docker SDK).
# It runs the app in `embedded` mode, loading the model directly from the
# bundled joblib file under /app/models.
FROM python:3.11-slim

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
RUN pip install --upgrade pip && pip install -r /app/requirements/streamlit.txt

COPY pyproject.toml /app/
COPY conf /app/conf
COPY src /app/src
COPY models /app/models

EXPOSE 7860

CMD ["streamlit", "run", "src/housing/streamlit_app/app.py", "--server.port=7860"]
