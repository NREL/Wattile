.PHONY: format check-format

format:
	poetry run black .
	poetry run isort .

check-format:
	poetry run black . --check
	poetry run isort . --check
