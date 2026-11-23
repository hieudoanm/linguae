.PHONY: sync export clean

ARTIFACT_DIR ?= ../artifacts/linguae
OUT_DIR ?= ../artifacts/models

sync:
	uv sync

export: sync
	uv run python export_models.py $(ARTIFACT_DIR) $(OUT_DIR)

clean:
	rm -rf $(OUT_DIR) .venv uv.lock
