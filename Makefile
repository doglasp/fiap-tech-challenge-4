.PHONY: install install-dev run test lint up down logs load-test

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest

lint:
	ruff check app tests scripts

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

load-test:
	python scripts/load_test.py --prices-file artifacts/AAPL_clean.csv
