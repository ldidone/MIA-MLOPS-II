.PHONY: help up down logs build train train-local test lint fmt clean \
	hf-space-create hf-space-remote hf-space-push hf-space-build

# Hugging Face Docker Space slug (repo id = <your_hf_user>/<HF_SPACE>).
# Override when creating / pushing, e.g. `make hf-space-create HF_SPACE=my-demo`.
HF_SPACE ?= california-housing-mia-mlops-ii

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
	@echo "  hf-space-create - Create an empty Docker Space on the Hub (needs hf auth)"
	@echo "  hf-space-remote - Print git remote add command for that Space"
	@echo "  hf-space-push   - Push current branch to Space remote (branch: hf)"
	@echo "  hf-space-build  - Local docker build (same as HF Space image)"

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

# --- Hugging Face Spaces (Docker) -------------------------------------------
# Requires: https://hf.co/docs/huggingface_hub/guides/cli  (`hf auth login`).
# After create, add the remote once, then push whenever you change the repo:
#   make hf-space-create HF_SPACE=my-space
#   make hf-space-remote HF_SPACE=my-space   # copy the printed line
#   git push hf main   # or: make hf-space-push

hf-space-create:
	hf repos create $(HF_SPACE) --type space --space-sdk docker --exist-ok --public

hf-space-remote:
	@u=$$(hf auth whoami --format json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('user') or d.get('name') or '')"); \
	if [ -z "$$u" ]; then echo "Run: hf auth login   (could not read username)"; exit 1; fi; \
	echo "git remote add hf https://huggingface.co/spaces/$$u/$(HF_SPACE)"; \
	echo "# If remote 'hf' already exists: git remote set-url hf https://huggingface.co/spaces/$$u/$(HF_SPACE)"

hf-space-push:
	@test "$$(git remote | grep -x hf)" = "hf" || (echo "Add remote first: make hf-space-remote" && exit 1)
	git push -u hf HEAD:main

hf-space-build:
	docker build -t hf-california-housing-space .
