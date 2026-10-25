# Linguae Artifacts

This directory holds the exported **Linguae language detector artifact** that is
consumed by the CLI (`packages/cli`), the FastAPI server (`packages/app/server`),
and any downstream integrations.

The artifact is a self-contained directory: it bundles the model weights, the
vocabulary, the label mapping, the architecture definition, and the inference
helpers so that no training notebook, no raw dataset, and no external model
registry is required at runtime.

## Directory layout

```
artifacts/linguae/
├── config.json              # Hyperparameters and preprocessing settings
├── vocabulary.json          # Character -> integer ID mapping
├── labels.json              # label2idx / idx2label mappings (ISO 639-3 codes)
├── model.pt                 # PyTorch state_dict (torch.save format)
├── linguae_inference.py     # Standalone inference helpers (CharCNN, predict_text, ...)
├── example_inference.py     # Minimal runnable example
├── manifest.json            # Provenance: dataset, training config, file hashes
├── metrics.json             # Final train/val/test metrics
├── training_history.csv     # Per-epoch loss/accuracy
├── confusion_matrix.csv     # Full confusion matrix over the test split
└── misclassified_examples.csv  # Misclassified test examples with scores
```

Training visualisations are written next to the artifact, in
`artifacts/images/`:

```
artifacts/images/
├── network_architecture.png       # VisualTorch flow view of the CNN
├── layer_dimensions.png           # Per-layer shapes and parameter counts
├── training_curves.png            # Loss and accuracy over epochs
├── confusion_matrix.png           # Test-set confusion matrix
├── prediction_probabilities.png   # Top-k softmax scores for sample inputs
├── dataset_quality.png            # Length/class balance by source
└── accuracy_by_length.png         # Accuracy by input length
```

## How to use

### Python (native)

```python
from linguae_inference import load_artifact, predict_text

artifact = load_artifact("./artifacts/linguae")
print(predict_text("Bonjour tout le monde", artifact))
```

### CLI

```bash
linguae "Bonjour tout le monde"
```

### HTTP server

```bash
linguae-server
curl -X POST http://127.0.0.1:8000/detect -d '{"text":"Hello world"}'
```

## Model format

The weights are stored as a **PyTorch `state_dict`** via `torch.save`, i.e. a
Python-specific serialization format (a zip archive containing tensor data and
metadata). It is **not** a pickle file in the `sklearn` sense, and it is **not**
ONNX.

### Can the model be imported to other formats?

**`.pkl` (scikit-learn pickle) — No, not directly.**

The model is a `torch.nn.Module`, not a scikit-learn estimator, so it cannot be
loaded with `joblib`/`pickle` in a meaningful way. Pickling a `torch.nn.Module`
is possible (`torch.save` uses pickle under the hood), but the result is not
portable to non-Python environments and is not a standard `.pkl` artifact. If a
`.pkl`-style artifact is required, the practical path is:

1. Export the architecture + weights to **ONNX** (see below).
2. Wrap the ONNX model in a small scikit-learn-style class and `joblib.dump`
   it. This is an extra packaging step, not a one-line conversion.

**`.onnx` (ONNX) — Yes, with caveats.**

The model can be exported to ONNX using `torch.onnx.export`. Because the model
uses **masked max pooling** (the mask depends on the input `padding_idx`), the
export requires either:

- A **dynamic sequence length** axis, and a precomputed mask passed as an
  additional input, or
- A **fixed maximum length** (256) with the mask baked into the graph.

The recommended export snippet:

```python
import torch
from linguae_inference import build_model, load_artifact

artifact = load_artifact("./artifacts/linguae")
model = artifact["model"]
config = artifact["config"]

dummy = torch.randint(0, config["vocab_size"], (1, config["max_length"]))
torch.onnx.export(
    model,
    dummy,
    "linguae.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {1: "sequence_length"}},
    opset_version=17,
)
```

After export, inference requires the same preprocessing (NFC normalization,
character encoding via `vocabulary.json`, right-padding to `max_length`) and the
same `labels.json` mapping. The ONNX model alone is not a drop-in replacement for
the `linguae_inference.py` helpers.

**Other formats**

- **TorchScript** — `torch.jit.script(model)` or `torch.jit.trace(model, dummy)`
  works for frozen inference and can be loaded with `torch.jit.load`.
- **TensorFlow SavedModel / TFLite** — requires ONNX as an intermediate step
  and a manual translation of the vocabulary + preprocessing logic.

## Reproducibility

See `manifest.json` for the full provenance record: dataset path and SHA-256,
random seed, split fractions, hyperparameters, environment (Python, torch,
platform), and per-file content hashes.
