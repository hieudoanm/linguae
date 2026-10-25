.PHONY: install run smoke

install:
	uv sync

run:
	uv run linguae "Bonjour tout le monde"

smoke: run
	printf 'Hello world' | uv run linguae --confidence
