.PHONY: sync export clean

PYTHON ?= 3.13
ARTIFACT_DIR ?= ../artifacts/linguae
OUT_DIR ?= ../artifacts/models

sync:
	uv sync --python $(PYTHON)

export: sync
	uv run --python $(PYTHON) python export_models.py $(ARTIFACT_DIR) $(OUT_DIR)

clean:
	rm -rf $(OUT_DIR) .venv uv.lock
