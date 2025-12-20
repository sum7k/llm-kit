.PHONY: lint format

lint:
	poetry run ruff check .
	poetry run ruff format --check .
	poetry run mypy .

format:
	poetry run ruff check --fix .
	poetry run ruff format .
