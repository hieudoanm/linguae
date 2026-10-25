.PHONY: sync run test lint format clean

sync:
	uv sync

dev:
	uv run linguae-server

test:
	uv run pytest tests -v

lint:
	uv run ruff check src

format:
	uv run ruff format src

clean:
	rm -rf .venv
