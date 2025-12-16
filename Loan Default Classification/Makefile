.PHONY: help install test lint format clean run-notebooks docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean temporary files"
	@echo "  make run-notebooks - Start Jupyter notebook server"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	pylint src/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage

run-notebooks:
	jupyter notebook notebooks/

run-lab:
	jupyter lab notebooks/

preprocess:
	python src/data/preprocess.py --input data/raw/loan_data.csv --output data/processed/

train:
	python src/models/train_model.py --data data/processed/train.csv --config configs/model_config.yaml

evaluate:
	python src/models/evaluate_model.py --model models/saved_models/model.pkl --data data/processed/test.csv

docker-build:
	docker build -t loan-eligibility-prediction .

docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data loan-eligibility-prediction

docker-compose-up:
	docker-compose up

docker-compose-down:
	docker-compose down

all: install test lint
