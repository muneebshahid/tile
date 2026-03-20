.PHONY: type_check format test

type_check:
	uv run ty check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest tests
