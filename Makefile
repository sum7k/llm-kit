.PHONY: lint format

lint:
	poetry run black --check .
	poetry run isort --check-only .
	poetry run ruff check .
	poetry run ruff format --check .
	poetry run mypy .

format:
	poetry run black .
	poetry run isort .
	poetry run ruff format .
	poetry run ruff check --fix .
