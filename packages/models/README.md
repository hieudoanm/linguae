# Linguae Models

Training pipeline and data for the Linguae character-level language detector.

## Structure

```
packages/models/
├── notebooks/                  # Training notebooks
│   ├── wili-2018-txt-to-csv.ipynb
│   └── train_language_detector.ipynb
├── data/
│   ├── csv/wili-2018.csv      # Combined text + language dataset
│   ├── processed/              # Training outputs
│   │   ├── dataset.csv
│   │   ├── metrics.json
│   │   ├── misclassified.csv
│   │   ├── training_curves.png
│   │   └── confusion_matrix.png
│   └── txt/wili-2018/          # Raw WiLI-2018 data
├── models/                     # language_detector.pt
├── artifacts/linguae/          # CLI artifact
├── pyproject.toml
└── Makefile
```

## Getting started

```bash
uv sync
make csv                      # build data/csv/wili-2018.csv from raw txt
make notebook                 # alias for csv
```

Then run `notebooks/train_language_detector.ipynb` in Jupyter to train the
model and produce the artifact in `artifacts/linguae`.

## Data

The WiLI-2018 dataset (without labels) is used for training. The raw files live
in `data/txt/wili-2018/`:

- `x_train.txt`, `y_train.txt` (117,500 examples)
- `x_test.txt`, `y_test.txt` (117,500 examples)
- `labels.csv` maps ISO 639-3 codes to language names and families.

`data/csv/wili-2018.csv` is the combined CSV with columns `text,language`.
`data/processed/dataset.csv` is the cleaned version used by the training
notebook.

## Training

The training notebook (`notebooks/train_language_detector.ipynb`) performs:

1. **Data validation**: checks required columns, missing values, duplicates,
   per-language counts, and character-length distributions.
2. **Splitting**: deterministic 80/10/10 train/val/test split with exact
   duplicate texts grouped within a single split.
3. **Vocabulary**: character vocabulary built from the training split only,
   with `<pad>` (0) and `<unk>` (1) special tokens.
4. **Model**: character-level CNN with embedding, 1D convolutions (kernel sizes
   3, 4, 5), masked max-pooling, and a fully connected head.
5. **Training**: AdamW, cross-entropy on logits, early stopping on validation
   loss, best checkpoint restoration.
6. **Evaluation**: accuracy, per-class precision/recall/F1, macro-F1,
   confusion matrix, and misclassified examples.

## Outputs

- `models/language_detector.pt` - trained model state dict
- `data/processed/metrics.json` - evaluation report
- `data/processed/misclassified.csv` - error analysis
- `data/processed/training_curves.png` - loss and accuracy curves
- `data/processed/confusion_matrix.png` - confusion matrix heatmap
- `artifacts/linguae/` - CLI artifact (config.json, vocabulary.json,
  labels.json, model.pt, linguae_inference.py)

## Notebooks

- `notebooks/wili-2018-txt-to-csv.ipynb` - converts raw txt files to CSV.
- `notebooks/train_language_detector.ipynb` - full training pipeline.

Run `make csv` to regenerate the CSV from raw data.