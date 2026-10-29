.PHONY: install run debug clean lint typecheck

PYTHON = python3
MAIN = src

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	uv run cleanpy .

lint:
	uv run flake8 src
	uv run mypy src
