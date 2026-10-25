.PHONY: wili tatoeba words csv train convert verify clean

CARGO ?= cargo
CRATES_MANIFEST ?= ../Cargo.toml
DATA_DIR ?= ../linguae-data/data
LINGUAE_DATA ?= $(CARGO) run --quiet --release --manifest-path $(CRATES_MANIFEST) -p linguae-data -- --data-dir $(DATA_DIR)
LINGUAE_TRAIN ?= $(CARGO) run --quiet --release --manifest-path $(CRATES_MANIFEST) -p linguae-train --

wili:
	$(LINGUAE_DATA) wili

tatoeba:
	$(LINGUAE_DATA) tatoeba

words:
	$(LINGUAE_DATA) words

csv: wili tatoeba words

train:
	LINGUAE_DATA_DIR=$(DATA_DIR) $(LINGUAE_TRAIN) train

convert:
	$(LINGUAE_TRAIN) convert --source $(SOURCE) --destination $(DESTINATION) --overwrite

verify:
	$(LINGUAE_TRAIN) verify --device cpu

clean:
	rm -f $(DATA_DIR)/csv/wili-2018.csv $(DATA_DIR)/csv/tatoeba.csv $(DATA_DIR)/csv/wili-2018-words.csv
