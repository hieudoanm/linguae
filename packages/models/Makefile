.PHONY: install wili tatoeba words csv notebook run train train-notebook clean

CARGO ?= cargo
CRATES_MANIFEST ?= ../../crates/Cargo.toml
DATA_DIR ?= ../../crates/linguae-data/data
LINGUAE_DATA ?= $(CARGO) run --quiet --release --manifest-path $(CRATES_MANIFEST) -p linguae-data -- --data-dir $(DATA_DIR)

install:
	uv sync

wili:
	$(LINGUAE_DATA) wili

tatoeba:
	$(LINGUAE_DATA) tatoeba

words:
	$(LINGUAE_DATA) words

csv: wili tatoeba words

notebook: csv

run: csv

# Training regenerates the exported artifacts, so allow replacing a previous
# run by default. Pass LINGUAE_ALLOW_OVERWRITE=0 to keep the safety check.
LINGUAE_ALLOW_OVERWRITE ?= 1

train:
	LINGUAE_DATA_DIR=$(DATA_DIR) LINGUAE_ALLOW_OVERWRITE=$(LINGUAE_ALLOW_OVERWRITE) uv run python scripts/train_language_detector.py

train-notebook:
	LINGUAE_DATA_DIR=$(DATA_DIR) LINGUAE_ALLOW_OVERWRITE=$(LINGUAE_ALLOW_OVERWRITE) uv run jupyter nbconvert --to notebook --execute --inplace notebooks/train_language_detector.ipynb

clean:
	rm -f $(DATA_DIR)/csv/wili-2018.csv $(DATA_DIR)/csv/tatoeba.csv $(DATA_DIR)/csv/wili-2018-words.csv
