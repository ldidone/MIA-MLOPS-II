.PHONY: help up down logs build train train-local test lint fmt clean

help:
	@echo "Available targets:"
	@echo "  up          - Start the full docker-compose stack"
	@echo "  down        - Stop and remove containers"
	@echo "  build       - Rebuild images"
	@echo "  logs        - Tail logs of all services"
	@echo "  train       - Trigger the Airflow training DAG"
	@echo "  train-local - Run the training pipeline inside the api container"
	@echo "  test        - Run pytest inside the api container"
	@echo "  lint        - Run ruff"
	@echo "  clean       - Remove local caches"

up:
	docker compose up -d --build

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f --tail=200

train:
	docker compose exec airflow-scheduler airflow dags trigger california_housing_training

train-local:
	docker compose exec api python -m scripts.run_training_local

test:
	docker compose exec api sh -c "pip install -q -r /app/requirements/dev.txt && pytest"

lint:
	ruff check src tests scripts

fmt:
	ruff check --fix src tests scripts

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
