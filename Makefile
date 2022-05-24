.PHONY: format check-format

format:
	poetry run black .
	poetry run isort .
	poetry run pylama .

check-format:
	poetry run black . --check
	poetry run isort . --check
	poetry run pylama .
