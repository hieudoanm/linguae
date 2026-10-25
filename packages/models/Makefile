.PHONY: install csv notebook run clean

install:
	uv sync

csv: notebooks/wili_2018_txt_to_csv.ipynb
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/wili_2018_txt_to_csv.ipynb

notebook: csv

run: csv

clean:
	rm -f data/csv/wili-2018.csv

train:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/train_language_detector.ipynb
