.PHONY: test lint typecheck all install

install:
	pip install -e ".[dev]" --break-system-packages

test:
	python -m pytest tests/ -v --tb=short

lint:
	python -m ruff check src/ tests/

typecheck:
	python -m mypy src/

all: lint typecheck test
