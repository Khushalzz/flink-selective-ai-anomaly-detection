.PHONY: help install test lint demo benchmark docker-up docker-down clean

PYTHON ?= python

help:
	@echo "BDT: Bounded Decision Tiering Streaming Pipeline"
	@echo "Available commands:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make test         - Run pytest test suite"
	@echo "  make lint         - Run flake8 code linter"
	@echo "  make demo         - Run interactive streaming simulation demo"
	@echo "  make benchmark    - Run 6-system evaluation benchmark"
	@echo "  make sweep        - Run streaming escalation sweep"
	@echo "  make docker-up    - Start Docker infrastructure containers"
	@echo "  make docker-down  - Stop Docker infrastructure containers"
	@echo "  make flink-build  - Package shaded Apache Flink Java JAR"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest flake8

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=scratch,venv,.venv,archive*

demo:
	$(PYTHON) demo_streaming_pipeline.py --samples 5000

benchmark:
	$(PYTHON) experiments/evaluate_systems.py
	$(PYTHON) experiments/plot_benchmark.py

sweep:
	$(PYTHON) experiments/streaming_benchmark_sweep.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down

flink-build:
	cd flink-job && mvn clean package -DskipTests
