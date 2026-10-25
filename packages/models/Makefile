.PHONY: install wili tatoeba run train train-notebook clean

install:
	uv sync

wili: scripts/wili_2018_txt_to_csv.py
	uv run python scripts/wili_2018_txt_to_csv.py

tatoeba: scripts/tatoeba_tsv_to_csv.py
	uv run python scripts/tatoeba_tsv_to_csv.py

run: wili tatoeba

# Training regenerates the exported artifacts, so allow replacing a previous
# run by default. Pass LINGUAE_ALLOW_OVERWRITE=0 to keep the safety check.
LINGUAE_ALLOW_OVERWRITE ?= 1

train:
	LINGUAE_ALLOW_OVERWRITE=$(LINGUAE_ALLOW_OVERWRITE) uv run python scripts/train_language_detector.py

train-notebook:
	LINGUAE_ALLOW_OVERWRITE=$(LINGUAE_ALLOW_OVERWRITE) uv run jupyter nbconvert --to notebook --execute --inplace notebooks/train_language_detector.ipynb

clean:
	rm -f data/csv/wili-2018.csv data/csv/tatoeba.csv
