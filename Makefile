.PHONY: setup train predict lab mlflow test format

setup:
	uv sync --all-groups

train:
	uv run python -m modeling.train

predict:
	uv run python -m modeling.predict models/linear data/X_test.csv data/y_test.csv

lab:
	uv run jupyter lab

mlflow:
	uv run mlflow ui

test:
	uv run pytest

format:
	uv run black .
