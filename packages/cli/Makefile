.PHONY: install lint test build run smoke

install:
	uv sync

lint:
	uv run ruff check src

test:
	@echo "No tests configured for CLI package"

build:
	uv run pyinstaller linguae.spec

run:
	uv run linguae "Bonjour tout le monde"

smoke: run
	printf 'Hello world' | uv run linguae --confidence