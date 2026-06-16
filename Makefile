PKG = causaldt_ad

.PHONY: help install dev lint format type test smoke docker clean

help:
	@echo "targets: install dev lint format type test smoke docker clean"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	black --check .
	isort --check-only .

format:
	black .
	isort .
	ruff check --fix .

type:
	python -m mypy $(PKG)

test:
	python -m pytest

smoke:
	python -m $(PKG).headworks --config _smoke route

docker:
	docker build -t $(PKG) .

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache *.egg-info runs
