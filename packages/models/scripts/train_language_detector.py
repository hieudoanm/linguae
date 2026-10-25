"""Train, evaluate and export the Linguae character-level language detector.

This script mirrors ``notebooks/train_language_detector.ipynb`` and is intended
for headless runs, CI smoke tests and long training jobs. Progress is reported
through the standard :mod:`logging` module.

Usage::

    cd packages/models
    uv run python scripts/train_language_detector.py

Training data comes from both datasets listed in ``CONFIG.sources``:

* ``data/csv/tatoeba.tsv`` - short conversational sentences.
* ``data/csv/wili-2018.csv`` - long sentences and Wikipedia paragraphs.

Each source is normalised, duplicate/contradictory texts are resolved, and the
combined frame is split per language. Short and long inputs therefore share one
label space, and metrics are reported per source as well as overall.

Environment overrides (all optional)::

    LINGUAE_LOG_LEVEL             DEBUG, INFO (default), WARNING, ...
    LINGUAE_PROGRESS_EVERY        log every N batches (default 200)
    LINGUAE_WILI_CSV              long-form CSV path
    LINGUAE_TATOEBA_TSV           short-form TSV path
    LINGUAE_DISABLE_WILI          "1" skips the WiLI source
    LINGUAE_DISABLE_TATOEBA       "1" skips the Tatoeba source
    LINGUAE_DATASET_PATH          single custom CSV instead of the two defaults
    LINGUAE_MIN_EXAMPLES_PER_LANGUAGE  drop rarer labels (default 100)
    LINGUAE_MAX_ROWS_PER_LANGUAGE_PER_SOURCE  0 keeps all; default 1000
    LINGUAE_MAX_ROWS_PER_LANGUAGE 0 keeps all rows; > 0 caps the combined frame
    LINGUAE_MAX_LENGTH            character sequence length (default 256)
    LINGUAE_BATCH_SIZE            batch size (default 256)
    LINGUAE_EPOCHS                maximum epochs (default 10)
    LINGUAE_PROCESSED_DIR         labels.json output directory
    LINGUAE_PLOTS_DIR             image output directory (default artifacts/images)
    LINGUAE_ARTIFACT_DIR          export directory (default artifacts/linguae)
    LINGUAE_ALLOW_OVERWRITE       "1" replaces a non-empty artifact directory
    LINGUAE_PACKAGE_ROOT          override package-root discovery

The exported artifact performs single-label language classification per string.
It does not detect mixed-language spans.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import sys
import time
import unicodedata
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import visualtorch
from matplotlib.colors import LogNorm
from torch import nn
from torch.utils.data import DataLoader, Dataset

plt.rcParams["figure.dpi"] = 110
plt.rcParams["figure.max_open_warning"] = 0

LOG_LEVEL = os.environ.get("LINGUAE_LOG_LEVEL", "INFO").upper()
PROGRESS_EVERY = max(1, int(os.environ.get("LINGUAE_PROGRESS_EVERY", "200")))
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("linguae.train")
SCRIPT_STARTED = time.perf_counter()
logger.info("Linguae language-detector pipeline started")


def log_section(title: str) -> None:
    """Log a stage banner with elapsed wall-clock time."""
    logger.info("=" * 78)
    logger.info("STEP: %s (t+%.1fs)", title, time.perf_counter() - SCRIPT_STARTED)
    logger.info("=" * 78)


def log_frame(frame: object, title: str) -> None:
    """Log a DataFrame or Series without truncating rows."""
    text = frame.to_string() if hasattr(frame, "to_string") else str(frame)
    logger.info("%s:\n%s", title, text)


def save_figure(path: Path, title: str = "") -> None:
    """Save the current Matplotlib figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("saved figure: %s%s", path, f" ({title})" if title else "")


def format_shape(shape: tuple[int, ...]) -> str:
    """Render a tensor shape (without the batch dimension) for a table cell."""
    if not shape:
        return "scalar"
    return " x ".join(f"{int(dimension):,}" for dimension in shape)


def render_network_architecture(model: nn.Module, path: Path, max_length: int) -> None:
    """Render the model with VisualTorch and frame the result with Matplotlib."""
    model_device = next(model.parameters()).device
    rendered = None
    if model_device.type != "cpu":
        model.to("cpu")
    try:
        rendered = visualtorch.render(
            model,
            input_shape=(1, max_length),
            input_dtype=torch.long,
            style="flow",
            show_dimension=True,
            legend=True,
            scale_xy=3,
            scale_z=0.3,
            min_xy=30,
            min_z=30,
            spacing=25,
            padding=20,
        )
    finally:
        if model_device.type != "cpu":
            model.to(model_device)

    assert rendered is not None
    width, height = rendered.size
    figure, axis = plt.subplots(figsize=(max(10.0, width / 100), max(4.5, height / 100)))
    axis.imshow(np.asarray(rendered), interpolation="lanczos")
    axis.set_axis_off()
    axis.set_title(
        "Character-level CNN architecture (VisualTorch flow view)",
        fontsize=13,
        pad=12,
    )
    figure.text(
        0.5,
        0.01,
        "Blocks are layers; the label under each block is its output shape. "
        f"Input: (batch, {max_length}) character IDs.",
        ha="center",
        fontsize=8,
        color="dimgray",
    )
    save_figure(path, path.name)


def collect_layer_dimensions(
    model: nn.Module, sample_input: torch.Tensor
) -> list[dict[str, object]]:
    """Run one sample through the model and record each leaf layer's dimensions."""
    records: list[dict[str, object]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str, module: nn.Module):
        def hook(
            hooked_module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            input_tensor = inputs[0] if inputs else None
            records.append(
                {
                    "layer": name,
                    "type": hooked_module.__class__.__name__,
                    "input": (
                        tuple(int(dimension) for dimension in input_tensor.shape[1:])
                        if torch.is_tensor(input_tensor)
                        else ()
                    ),
                    "output": (
                        tuple(int(dimension) for dimension in output.shape[1:])
                        if torch.is_tensor(output)
                        else ()
                    ),
                    "params": sum(
                        parameter.numel()
                        for parameter in hooked_module.parameters(recurse=False)
                    ),
                }
            )

        return hook

    for name, module in model.named_modules():
        if name and not any(module.children()):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            model(sample_input)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return records


def plot_layer_dimensions(
    records: list[dict[str, object]], path: Path, total_params: int
) -> None:
    """Draw a layer/parameter table plus a per-layer parameter bar chart."""
    if not records:
        raise RuntimeError("No layer activations were recorded for the dimension report.")

    row_count = len(records)
    figure = plt.figure(figsize=(17, max(5.0, 1.2 + 0.42 * row_count)))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.65, 1.0), wspace=0.18)

    table_axis = figure.add_subplot(grid[0, 0])
    table_axis.set_axis_off()
    table_axis.set_title("Layer dimensions", fontsize=12, loc="left", pad=10)
    table = table_axis.table(
        cellText=[
            [
                str(record["layer"]),
                str(record["type"]),
                format_shape(record["input"]),
                format_shape(record["output"]),
                f"{int(record['params']):,}",
            ]
            for record in records
        ],
        colLabels=["Layer", "Type", "Input (no batch)", "Output (no batch)", "Parameters"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1faee")

    chart_axis = figure.add_subplot(grid[0, 1])
    named_parameters = [
        (str(record["layer"]), int(record["params"]))
        for record in records
        if int(record["params"]) > 0
    ]
    positions = np.arange(len(named_parameters))
    values = np.array([parameter_count for _, parameter_count in named_parameters])
    bars = chart_axis.barh(positions, values, color="#457b9d")
    chart_axis.set_yticks(positions)
    chart_axis.set_yticklabels([name for name, _ in named_parameters], fontsize=8)
    chart_axis.invert_yaxis()
    chart_axis.set_xlabel("trainable parameters")
    chart_axis.set_title(
        f"Parameters per layer (total {total_params:,})", fontsize=12, loc="left", pad=10
    )
    chart_axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        chart_axis.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:,}",
            va="center",
            fontsize=7,
        )
    chart_axis.set_xlim(0, max(float(values.max()) * 1.18, 1.0))

    figure.suptitle(
        "Linguae character-level CNN — layer dimensions and parameter budget",
        fontsize=14,
        y=0.99,
    )
    save_figure(path, path.name)


def plot_prediction_probabilities(
    model: nn.Module,
    examples: list[dict[str, str]],
    path: Path,
    top_k: int = 5,
) -> None:
    """Plot top-k softmax scores for a handful of example strings."""
    if not examples:
        logger.warning("no examples available for the prediction-probability figure")
        return

    columns = min(2, len(examples))
    rows = math.ceil(len(examples) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(7.2 * columns, 3.9 * rows), squeeze=False
    )
    model.eval()
    for axis, example in zip(axes.flat, examples):
        text = str(example["text"])
        expected = example.get("expected")
        encoded = encode_text(text, vocabulary, CONFIG.max_length)
        batch = pad_sequences([encoded], max_length=CONFIG.max_length).to(DEVICE)
        with torch.inference_mode():
            probabilities = torch.softmax(model(batch), dim=1)[0].detach().cpu()

        k = min(top_k, len(labels))
        values, indices = torch.topk(probabilities, k=k)
        names = [idx2label[int(index)] for index in indices]
        scores = values.numpy()
        predicted = names[0]
        colors = [
            "#2a9d8f" if name == predicted else "#a8dadc" for name in names
        ]
        axis.barh(np.arange(k), scores, color=colors)
        axis.set_yticks(np.arange(k))
        axis.set_yticklabels(names)
        axis.invert_yaxis()
        axis.set_xlim(0, 1.0)
        axis.set_xlabel("softmax score")
        expected_line = f"expected: {expected}" if expected else "expected: unknown"
        axis.set_title(
            f"{expected_line} | predicted: {predicted} ({scores[0]:.3f})\n{text[:90]!r}",
            fontsize=8.5,
            loc="left",
        )
        for position, score in enumerate(scores):
            axis.text(
                min(float(score) + 0.01, 0.99),
                position,
                f"{score:.3f}",
                va="center",
                fontsize=8,
            )
        axis.grid(axis="x", alpha=0.25)

    for axis in axes.flat[len(examples) :]:
        axis.set_visible(False)

    figure.suptitle(
        "Prediction probabilities — top-k softmax scores on held-out test examples",
        fontsize=13,
    )
    save_figure(path, path.name)



def find_package_root(start: Path | None = None) -> Path:
    '''Locate packages/models by walking up from the current directory or script.'''
    override = os.environ.get("LINGUAE_PACKAGE_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if (root / "pyproject.toml").is_file() and (root / "data").is_dir():
            return root
        raise FileNotFoundError(
            f"LINGUAE_PACKAGE_ROOT={override!r} does not look like packages/models."
        )
    origins = [Path(start)] if start else [Path.cwd()]
    script_file = globals().get("__file__")
    if script_file:
        origins.append(Path(str(script_file)).resolve().parent)
    for origin in origins:
        for candidate in (origin.resolve(), *origin.resolve().parents):
            for option in (candidate, candidate / "packages" / "models"):
                if (
                    (option / "pyproject.toml").is_file()
                    and (option / "data").is_dir()
                    and (option / "notebooks").is_dir()
                ):
                    return option
    raise FileNotFoundError(
        "Could not locate packages/models. Start Jupyter inside the repository, or set "
        "LINGUAE_PACKAGE_ROOT=/absolute/path/to/packages/models."
    )


PACKAGE_ROOT = find_package_root()

# Tatoeba uses individual ISO 639-3 codes while WiLI-2018 uses a few older or
# macro-language codes. Merge the known equivalents so short Tatoeba sentences
# and long WiLI paragraphs share one output class instead of splitting a
# language into two labels.
TATOEBA_LABEL_ALIASES = {
    "cmn": "zho",       # Mandarin -> Standard Chinese
    "yue": "zh-yue",    # Cantonese -> WiLI Cantonese code
    "pes": "fas",       # Iranian Persian -> Persian
    "lvs": "lav",       # Standard Latvian -> Latvian
    "npi": "nep",       # Nepali (individual) -> Nepali (macrolanguage)
    "zsm": "msa",       # Standard Malay -> Malay
    "zlm": "msa",       # Malay (individual) -> Malay
    "kmr": "kur",       # Northern Kurdish -> Kurdish
    "swh": "swa",       # Swahili (individual) -> Swahili (macrolanguage)
    "swc": "swa",       # Congo Swahili -> Swahili (macrolanguage)
    "gsw": "als",       # Swiss German -> Alemannic German
    "bua": "bxr",       # Buriat (macrolanguage) -> Russia Buriat
    "gom": "kok",       # Goan Konkani -> Konkani
    "kpv": "kom",       # Komi-Zyrian -> Komi
}


@dataclass(frozen=True)
class DatasetSource:
    '''One raw text/language file that contributes rows to the training frame.'''

    name: str
    path: Path
    text_column: str = "text"
    label_column: str = "language"
    format: str = "csv"  # "csv" (header row) or "tsv" (explicit columns)
    columns: tuple[str, ...] | None = None  # names to assign when there is no header
    role: str = "mixed"  # "short", "long" or "mixed"; used for reporting only


def build_sources() -> tuple[DatasetSource, ...]:
    '''Resolve the default two-source layout, allowing environment overrides.'''
    legacy = os.environ.get("LINGUAE_DATASET_PATH")
    if legacy:
        return (DatasetSource(name="custom", path=Path(legacy).expanduser()),)

    sources = []
    if os.environ.get("LINGUAE_DISABLE_WILI", "0") != "1":
        sources.append(
            DatasetSource(
                name="wili-2018",
                path=Path(
                    os.environ.get(
                        "LINGUAE_WILI_CSV",
                        str(PACKAGE_ROOT / "data" / "csv" / "wili-2018.csv"),
                    )
                ).expanduser(),
                role="long",
            )
        )
    if os.environ.get("LINGUAE_DISABLE_TATOEBA", "0") != "1":
        sources.append(
            DatasetSource(
                name="tatoeba",
                path=Path(
                    os.environ.get(
                        "LINGUAE_TATOEBA_TSV",
                        str(PACKAGE_ROOT / "data" / "csv" / "tatoeba.tsv"),
                    )
                ).expanduser(),
                format="tsv",
                columns=("id", "language", "text"),
                role="short",
            )
        )
    return tuple(sources)


@dataclass(frozen=True)
class Config:
    '''Single configuration object for data, model, training and export.'''

    # Data sources: short Tatoeba sentences plus long WiLI paragraphs.
    sources: tuple[DatasetSource, ...] = build_sources()
    min_examples_per_language: int = 100  # drop labels too rare to evaluate
    max_rows_per_language_per_source: int = 1000  # 0 keeps every row per source
    max_rows_per_language: int = 0  # 0 keeps every row; > 0 caps the combined frame

    # Output locations
    processed_dir: Path = PACKAGE_ROOT / "data" / "processed"
    plots_dir: Path = PACKAGE_ROOT / "artifacts" / "images"

    # Deterministic splitting
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    # Encoding and model
    max_length: int = 256
    embedding_dim: int = 32
    num_filters: tuple[int, ...] = (128, 128, 128)  # one entry per kernel size
    kernel_sizes: tuple[int, ...] = (3, 5, 7)
    dropout: float = 0.3

    # Optimisation
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    early_stopping_patience: int = 2
    num_workers: int = 0

    # Evaluation and export
    max_misclassified_examples: int = 2000
    artifact_dir: Path = PACKAGE_ROOT / "artifacts" / "linguae"
    model_version: str = "1.0.0"
    allow_overwrite: bool = True

    # Runtime
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda" or "mps"


CONFIG = Config(
    sources=build_sources(),
    min_examples_per_language=int(
        os.environ.get("LINGUAE_MIN_EXAMPLES_PER_LANGUAGE", "100")
    ),
    max_rows_per_language_per_source=int(
        os.environ.get("LINGUAE_MAX_ROWS_PER_LANGUAGE_PER_SOURCE", "1000")
    ),
    max_rows_per_language=int(os.environ.get("LINGUAE_MAX_ROWS_PER_LANGUAGE", "0")),
    processed_dir=Path(
        os.environ.get("LINGUAE_PROCESSED_DIR", str(Config.processed_dir))
    ),
    plots_dir=Path(os.environ.get("LINGUAE_PLOTS_DIR", str(Config.plots_dir))),
    max_length=int(os.environ.get("LINGUAE_MAX_LENGTH", "256")),
    batch_size=int(os.environ.get("LINGUAE_BATCH_SIZE", "256")),
    epochs=int(os.environ.get("LINGUAE_EPOCHS", "10")),
    artifact_dir=Path(os.environ.get("LINGUAE_ARTIFACT_DIR", str(Config.artifact_dir))),
    allow_overwrite=os.environ.get("LINGUAE_ALLOW_OVERWRITE", "0") == "1",
)
PLOTS_DIR = CONFIG.plots_dir
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
logger.info("configuration:\n%s", json.dumps(asdict(CONFIG), indent=2, default=str))


def seed_everything(seed: int) -> None:
    '''Seed Python, NumPy and PyTorch (CPU and CUDA) random number generators.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(CONFIG.seed)

logger.info(f"package root  : {PACKAGE_ROOT}")
logger.info(
    "sources       : "
    + ", ".join(f"{source.name} ({source.role}: {source.path})" for source in CONFIG.sources)
)
logger.info(f"processed dir : {CONFIG.processed_dir}")
logger.info(f"artifacts     : {CONFIG.artifact_dir}")
logger.info(f"python        : {sys.version.split()[0]}")
logger.info(f"numpy         : {np.__version__}")
logger.info(f"pandas        : {pd.__version__}")
logger.info(f"torch         : {torch.__version__}")
logger.info(f"cuda available: {torch.cuda.is_available()}")
logger.info(f"mps available : {torch.backends.mps.is_available()}")

log_section('1.1 Reusable inference code (single source of truth)')

INFERENCE_MODULE_SOURCE = r'''"""Standalone inference utilities for the exported Linguae language detector.

The artifact directory contains this module next to ``model.pt``, ``config.json``,
``vocabulary.json`` and ``labels.json`` so that inference needs neither the
training notebook nor the training data.

Typical use::

    from linguae_inference import load_artifact, predict_text

    artifact = load_artifact("./artifacts/linguae")
    print(predict_text("Bonjour tout le monde", artifact))

The model predicts exactly one language for a whole string. It does not perform
mixed-language span detection.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import torch
from torch import nn

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC-normalise and collapse whitespace without changing case or accents."""
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", str(text))).strip()


def encode_text(text: str, vocabulary: dict[str, int], max_length: int) -> list[int]:
    """Encode one string as character IDs, mapping unknown characters to ``UNK_ID``."""
    characters = normalize_text(text)[:max_length]
    if not characters:
        return [UNK_ID]
    return [vocabulary.get(character, UNK_ID) for character in characters]


def pad_sequences(sequences: list[list[int]], max_length: int | None = None) -> torch.Tensor:
    """Right-pad ID sequences with ``PAD_ID`` and return a long tensor ``[batch, length]``."""
    if not sequences:
        return torch.empty((0, 0), dtype=torch.long)
    length = max(len(sequence) for sequence in sequences)
    if max_length is not None:
        length = min(length, max_length)
    length = max(length, 1)
    batch = torch.full((len(sequences), length), PAD_ID, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids = sequence[:length]
        if ids:
            batch[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return batch


class CharCNN(nn.Module):
    """Character-level CNN with masked max pooling over configurable kernels."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 32,
        num_filters: tuple[int, ...] = (128, 128),
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.3,
        padding_idx: int = PAD_ID,
    ) -> None:
        super().__init__()
        if len(num_filters) != len(kernel_sizes):
            raise ValueError("num_filters and kernel_sizes must have the same length.")
        if any(kernel_size % 2 == 0 for kernel_size in kernel_sizes):
            raise ValueError("kernel_sizes must be odd so that padding preserves the length.")
        self.padding_idx = int(padding_idx)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, filters, kernel_size, padding=kernel_size // 2)
            for filters, kernel_size in zip(num_filters, kernel_sizes)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(sum(num_filters), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map character IDs ``[batch, length]`` to logits ``[batch, num_classes]``."""
        mask = input_ids.ne(self.padding_idx)
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))
            features = features.masked_fill(~mask.unsqueeze(1), float("-inf"))
            maximum = features.max(dim=2).values
            pooled.append(torch.nan_to_num(maximum, neginf=0.0))
        representation = self.dropout(torch.cat(pooled, dim=1))
        return self.classifier(representation)


def build_model(config: dict, device: str = "cpu") -> CharCNN:
    """Construct a :class:`CharCNN` from an exported ``config.json`` mapping."""
    model = CharCNN(
        vocab_size=int(config["vocab_size"]),
        num_classes=int(config["num_classes"]),
        embedding_dim=int(config["embedding_dim"]),
        num_filters=tuple(int(value) for value in config["num_filters"]),
        kernel_sizes=tuple(int(value) for value in config["kernel_sizes"]),
        dropout=float(config["dropout"]),
        padding_idx=int(config.get("padding_idx", PAD_ID)),
    )
    return model.to(device)


def load_artifact(directory: str | Path, device: str = "cpu") -> dict:
    """Load config, vocabulary, labels and weights from an artifact directory."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    model = build_model(config, device=device)
    state_dict = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "directory": directory,
        "config": config,
        "vocabulary": vocabulary,
        "label2idx": labels["label2idx"],
        "idx2label": {int(index): label for index, label in labels["idx2label"].items()},
        "model": model,
        "device": torch.device(device),
    }


def predict_text(text: str, artifact: dict, top_k: int = 3) -> dict:
    """Predict one language label for a whole string and return top-k scores."""
    model = artifact["model"]
    max_length = int(artifact["config"]["max_length"])
    encoded = encode_text(text, artifact["vocabulary"], max_length)
    batch = pad_sequences([encoded], max_length=max_length).to(artifact["device"])
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch), dim=1)[0]
    k = max(1, min(int(top_k), probabilities.numel()))
    scores, indices = torch.topk(probabilities, k)
    predictions = [
        {"language": artifact["idx2label"][int(index)], "score": float(score)}
        for score, index in zip(scores.tolist(), indices.tolist())
    ]
    return {
        "text": text,
        "language": predictions[0]["language"],
        "confidence": predictions[0]["score"],
        "top_k": predictions,
    }


def predict_batch(texts: list[str], artifact: dict, batch_size: int = 256) -> list[dict]:
    """Predict one language label per string in a list."""
    model = artifact["model"]
    max_length = int(artifact["config"]["max_length"])
    predictions = []
    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        sequences = [encode_text(text, artifact["vocabulary"], max_length) for text in chunk]
        batch = pad_sequences(sequences, max_length=max_length).to(artifact["device"])
        with torch.inference_mode():
            probabilities = torch.softmax(model(batch), dim=1)
        best = probabilities.argmax(dim=1)
        confidences = probabilities.gather(1, best.unsqueeze(1)).squeeze(1)
        predictions.extend(
            {
                "text": text,
                "language": artifact["idx2label"][int(index)],
                "confidence": float(confidence),
            }
            for text, index, confidence in zip(chunk, best.tolist(), confidences.tolist())
        )
    return predictions
'''

_module_namespace: dict[str, object] = {}
exec(compile(INFERENCE_MODULE_SOURCE, "linguae_inference.py", "exec"), _module_namespace)

CharCNN = _module_namespace["CharCNN"]
build_model = _module_namespace["build_model"]
load_artifact = _module_namespace["load_artifact"]
predict_text = _module_namespace["predict_text"]
predict_batch = _module_namespace["predict_batch"]
normalize_text = _module_namespace["normalize_text"]
encode_text = _module_namespace["encode_text"]
pad_sequences = _module_namespace["pad_sequences"]
PAD_ID = _module_namespace["PAD_ID"]
UNK_ID = _module_namespace["UNK_ID"]
SPECIAL_TOKENS = _module_namespace["SPECIAL_TOKENS"]

logger.info(f"inference module lines: {len(INFERENCE_MODULE_SOURCE.splitlines())}")
logger.info(f"special tokens: {SPECIAL_TOKENS} (pad={PAD_ID}, unk={UNK_ID})")

log_section('2. Load and validate the combined dataset')

def read_dataset_source(source: DatasetSource) -> pd.DataFrame:
    '''Read one source and return text/language/source columns as strings.'''
    logger.info("reading %s source %r: %s", source.role, source.name, source.path)
    started = time.perf_counter()
    common = dict(dtype=str, keep_default_na=False, na_values=[])
    if source.format == "tsv":
        if source.columns is None:
            raise ValueError(
                f"TSV source {source.name!r} needs explicit columns, got {source.columns!r}."
            )
        frame = pd.read_csv(
            source.path,
            sep="\t",
            header=None,
            names=list(source.columns),
            usecols=[source.label_column, source.text_column],
            quoting=csv.QUOTE_NONE,
            **common,
        )
    elif source.format == "csv":
        frame = pd.read_csv(source.path, **common)
    else:
        raise ValueError(
            f"Unsupported format {source.format!r} for source {source.name!r}; "
            "use 'csv' or 'tsv'."
        )
    missing = [
        column
        for column in (source.text_column, source.label_column)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing} in {source.path}; "
            f"found columns {list(frame.columns)}."
        )
    selected = frame[[source.text_column, source.label_column]].rename(
        columns={source.text_column: "text", source.label_column: "language"}
    )
    selected["source"] = source.name
    selected["dataset_role"] = source.role
    logger.info(
        "loaded %-10s %9d rows in %5.1f s (%.1f MiB)",
        source.name,
        len(selected),
        time.perf_counter() - started,
        source.path.stat().st_size / (1024 * 1024),
    )
    return selected


def load_sources(sources: tuple[DatasetSource, ...]) -> pd.DataFrame:
    '''Read every configured source, warning about and skipping missing files.'''
    frames = []
    for source in sources:
        if not source.path.is_file():
            logger.warning(
                "dataset source %r not found at %s; skipping it",
                source.name,
                source.path,
            )
            continue
        frames.append(read_dataset_source(source))
    if not frames:
        raise FileNotFoundError(
            "No dataset sources found. Expected data/csv/wili-2018.csv and "
            "data/csv/tatoeba.tsv, or set LINGUAE_DATASET_PATH to a custom CSV."
        )
    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "combined %d source(s): %d rows, columns %s",
        len(frames),
        len(combined),
        list(combined.columns),
    )
    return combined


raw_df = load_sources(CONFIG.sources)
logger.info(f"rows: {len(raw_df):,}  columns: {list(raw_df.columns)}")
logger.info(f"missing values per column: {raw_df.isna().sum().to_dict()}")
logger.info("rows by source:\n%s", raw_df.groupby("source").size().to_string())
log_frame(raw_df.head(3), "first rows")

log_section('2.1 Normalisation')

_WHITESPACE_RE = re.compile(r"\s+")


def normalise_text(value: object) -> str:
    '''NFC-normalise and collapse whitespace; case and accents are preserved.'''
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", str(value))).strip()


def validate_and_normalise(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    '''Return (clean, invalid, report) without silently discarding invalid records.'''
    text = frame["text"].astype(str)
    label = frame["language"].astype(str).str.strip().replace({"\\N": ""})
    label = label.replace(TATOEBA_LABEL_ALIASES)

    normalised = pd.DataFrame(
        {
            "text": text.map(normalise_text),
            "language": label,
            "source": frame["source"],
            "dataset_role": frame["dataset_role"],
            "raw_text": text,
            "raw_language": frame["language"].astype(str),
        }
    )

    empty_text = text.str.strip().eq("")
    empty_label = label.eq("")
    empty_after_normalisation = normalised["text"].eq("")
    invalid_mask = empty_text | empty_label | empty_after_normalisation

    invalid = frame.loc[invalid_mask].copy()
    if len(invalid):
        invalid["reason"] = np.select(
            [
                empty_text[invalid_mask].to_numpy(),
                empty_label[invalid_mask].to_numpy(),
                empty_after_normalisation[invalid_mask].to_numpy(),
            ],
            ["empty text after strip", "empty language label", "empty text after normalisation"],
            default="unknown",
        )
    else:
        invalid["reason"] = pd.Series(dtype="object")

    clean = normalised.loc[~invalid_mask].copy()
    label_counts = clean["language"].value_counts()

    report = {
        "input_rows": int(len(frame)),
        "valid_rows": int(len(clean)),
        "invalid_rows": int(invalid_mask.sum()),
        "missing_values": {column: int(frame[column].isna().sum()) for column in frame.columns},
        "rows_by_source": {
            str(source): int(count) for source, count in clean.groupby("source").size().items()
        },
        "empty_text_rows": int(empty_text.sum()),
        "empty_label_rows": int(empty_label.sum()),
        "empty_after_normalisation_rows": int(empty_after_normalisation.sum()),
        "rows_changed_by_normalisation": int((normalised["text"] != text).sum()),
        "unique_languages": int(clean["language"].nunique()),
        "duplicate_text_rows": int(clean.duplicated(subset=["text"]).sum()),
        "duplicate_text_label_rows": int(clean.duplicated(subset=["text", "language"]).sum()),
        "texts_with_conflicting_labels": int(
            clean.groupby("text")["language"].nunique().gt(1).sum()
        ),
        "min_examples_per_language": int(label_counts.min()) if len(label_counts) else 0,
        "max_examples_per_language": int(label_counts.max()) if len(label_counts) else 0,
    }

    if clean.empty:
        raise ValueError("No valid rows remain after validation; check the dataset.")
    if report["empty_label_rows"]:
        logger.warning(
            "%d row(s) have an empty language label (for example Tatoeba '\\N') "
            "and are excluded from training; inspect invalid_df for details.",
            report["empty_label_rows"],
        )
    return clean, invalid, report


clean_df, invalid_df, validation_report = validate_and_normalise(raw_df)
logger.info(json.dumps(validation_report, indent=2))
if not invalid_df.empty:
    logger.info(f"Invalid rows kept for inspection ({len(invalid_df):,}):")
    log_frame(invalid_df.head(10), "invalid rows (first 10)")

log_section('2.2 Resolve duplicates and sample the combined frame')

def drop_conflicting_and_duplicate_texts(frame: pd.DataFrame) -> pd.DataFrame:
    '''Remove contradictory labels and exact duplicates before splitting.'''
    labels_per_text = frame.groupby("text", sort=False)["language"].nunique()
    conflicting_texts = labels_per_text[labels_per_text > 1].index
    if len(conflicting_texts):
        before = len(frame)
        frame = frame.loc[~frame["text"].isin(conflicting_texts)].copy()
        logger.warning(
            "dropped %d row(s) with contradictory labels across %d text(s); "
            "keeping both labels would teach the model noise",
            before - len(frame),
            len(conflicting_texts),
        )
    duplicates = frame.duplicated(subset=["text", "language"], keep="first")
    if duplicates.any():
        logger.info(
            "dropped %d exact duplicate (text, language) row(s)",
            int(duplicates.sum()),
        )
        frame = frame.loc[~duplicates].copy()
    return frame


def subsample_by_group(
    frame: pd.DataFrame,
    group_columns: tuple[str, ...],
    limit: int,
    seed: int,
    label: str,
) -> pd.DataFrame:
    '''Deterministically cap each group, for example each language/source pair.'''
    if limit <= 0:
        logger.info("%s: keeping every row (limit=0)", label)
        return frame
    rng = np.random.default_rng(seed)
    selected = []
    groups = frame.groupby(list(group_columns), sort=True)
    for index, (_, group) in enumerate(groups, start=1):
        if len(group) > limit:
            keep = np.sort(rng.choice(len(group), size=limit, replace=False))
            selected.append(group.iloc[keep])
        else:
            selected.append(group)
        if index % 100 == 0:
            logger.info("  %s: sampled %d/%d groups", label, index, groups.ngroups)
    capped = pd.concat(selected).sort_index()
    logger.info(
        "%s: %d -> %d rows (limit=%d per group)",
        label,
        len(frame),
        len(capped),
        limit,
    )
    return capped


def drop_rare_languages(frame: pd.DataFrame, minimum: int) -> pd.DataFrame:
    '''Drop labels with too few examples for a meaningful train/test split.'''
    counts = frame["language"].value_counts()
    rare = counts[counts < minimum]
    if len(rare):
        examples = ", ".join(
            f"{language}={count}" for language, count in rare.sort_values().head(20).items()
        )
        logger.info(
            "dropping %d language(s) with fewer than %d examples: %s%s",
            len(rare),
            minimum,
            examples,
            " ..." if len(rare) > 20 else "",
        )
        frame = frame.loc[frame["language"].isin(counts[counts >= minimum].index)].copy()
    else:
        logger.info("all %d languages have at least %d examples", len(counts), minimum)
    return frame


clean_df = drop_conflicting_and_duplicate_texts(clean_df)
clean_df = drop_rare_languages(clean_df, CONFIG.min_examples_per_language)
clean_df = subsample_by_group(
    clean_df,
    ("language", "source"),
    CONFIG.max_rows_per_language_per_source,
    CONFIG.seed,
    "per-source cap",
)
if CONFIG.max_rows_per_language > 0:
    clean_df = subsample_by_group(
        clean_df,
        ("language",),
        CONFIG.max_rows_per_language,
        CONFIG.seed,
        "combined cap",
    )
clean_df = clean_df.reset_index(drop=True)
logger.info(
    "using %d rows, %d languages, %d source(s)",
    len(clean_df),
    clean_df["language"].nunique(),
    clean_df["source"].nunique(),
)

log_section('2.3 Dataset quality report')

label_counts = clean_df["language"].value_counts().sort_index()
text_character_lengths = clean_df["text"].str.len()
source_counts = clean_df.groupby("source").size().sort_values(ascending=False)

logger.info(f"unique labels: {clean_df['language'].nunique():,}")
logger.info(f"sources: {list(source_counts.index)}")
logger.info("rows by source:\n%s", source_counts.to_string())
logger.info(
    "examples per label: "
    f"min={label_counts.min():,} median={int(label_counts.median()):,} max={label_counts.max():,}"
)
logger.info(
    "character length: "
    f"min={text_character_lengths.min():,} median={text_character_lengths.median():.0f} "
    f"p90={text_character_lengths.quantile(0.9):.0f} p99={text_character_lengths.quantile(0.99):.0f} "
    f"max={text_character_lengths.max():,}"
)
logger.info("character length by source:")
for source, group in clean_df.groupby("source", sort=True):
    lengths = group["text"].str.len()
    logger.info(
        "  %-10s rows=%7d min=%4d median=%5.0f p90=%5.0f p99=%7.0f max=%7d",
        source,
        len(group),
        lengths.min(),
        lengths.median(),
        lengths.quantile(0.9),
        lengths.quantile(0.99),
        lengths.max(),
    )
logger.info(f"duplicate text rows: {clean_df.duplicated(subset=['text']).sum():,}")
logger.info(
    "conflicting duplicate texts before resolution: "
    f"{validation_report['texts_with_conflicting_labels']:,}"
)

logger.info("most common labels:")
logger.info(label_counts.sort_values(ascending=False).head(10).to_string())
logger.info("rarest labels:")
logger.info(label_counts.sort_values().head(10).to_string())

figure, axes = plt.subplots(1, 3, figsize=(18, 4))
for source, group in clean_df.groupby("source", sort=True):
    lengths = group["text"].str.len().clip(upper=1500)
    axes[0].hist(lengths, bins=60, alpha=0.6, label=source)
axes[0].legend()
axes[0].set(
    title="Character length by source (clipped at 1500)",
    xlabel="characters",
    ylabel="examples",
)
axes[1].bar(
    range(len(label_counts)),
    label_counts.sort_values(ascending=False).to_numpy(),
    color="darkorange",
)
axes[1].set(title="Examples per language (sorted)", xlabel="language rank", ylabel="examples")
axes[2].bar(
    source_counts.index.astype(str),
    source_counts.to_numpy(),
    color="seagreen",
)
axes[2].set(title="Rows by dataset source", xlabel="source", ylabel="examples")
plt.tight_layout()
save_figure(PLOTS_DIR / "dataset_quality.png", "dataset_quality.png")

log_frame(text_character_lengths.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_frame("characters"), "character length percentiles")

log_section('3. Train/validation/test split')

def split_dataset(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict]:
    '''Deterministically split per language while keeping duplicate texts together.'''
    fractions = (config.train_fraction, config.val_fraction, config.test_fraction)
    if any(fraction <= 0 for fraction in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(
            f"train/val/test fractions must be positive and sum to 1.0, got {fractions}"
        )

    frame = frame.copy()
    frame["group_id"] = frame.groupby("text", sort=False).ngroup()
    label_of_group = frame.drop_duplicates("group_id").set_index("group_id")["language"]

    rng = np.random.default_rng(config.seed)
    split_of_group = pd.Series(index=label_of_group.index, dtype="object")

    grouped = label_of_group.groupby(label_of_group).groups
    total_languages = len(grouped)
    logger.info(
        "splitting %d languages deterministically (seed=%d, fractions=%.2f/%.2f/%.2f)",
        total_languages,
        config.seed,
        config.train_fraction,
        config.val_fraction,
        config.test_fraction,
    )

    for language_index, (language, group_ids) in enumerate(grouped.items(), start=1):
        shuffled = rng.permutation(np.asarray(group_ids))
        count = len(shuffled)
        n_val = max(1, int(round(count * config.val_fraction)))
        n_test = max(1, int(round(count * config.test_fraction)))
        n_train = count - n_val - n_test
        if n_train < 1:
            raise ValueError(
                f"Language {language!r} has only {count} unique text(s); at least 3 are "
                "needed for a train/validation/test split. Increase the dataset or adjust "
                "the split fractions."
            )
        split_of_group.loc[shuffled[:n_train]] = "train"
        split_of_group.loc[shuffled[n_train : n_train + n_val]] = "validation"
        split_of_group.loc[shuffled[n_train + n_val :]] = "test"

        if language_index % 25 == 0 or language_index == total_languages:
            logger.info(
                "  split progress: %d/%d languages (%.1f%%), %d rows assigned",
                language_index,
                total_languages,
                100.0 * language_index / total_languages,
                int(split_of_group.notna().sum()),
            )

    frame["split"] = frame["group_id"].map(split_of_group)
    if frame["split"].isna().any():
        raise RuntimeError("Some examples were not assigned to a split.")

    train_labels = set(frame.loc[frame["split"] == "train", "language"])
    evaluation_labels = set(frame.loc[frame["split"].isin(["validation", "test"]), "language"])
    missing_from_train = evaluation_labels - train_labels
    if missing_from_train:
        raise ValueError(
            f"Evaluation languages missing from training: {sorted(missing_from_train)[:10]}"
        )

    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_texts = set(frame.loc[frame["split"] == left, "text"])
        right_texts = set(frame.loc[frame["split"] == right, "text"])
        overlaps[f"{left}|{right}"] = len(left_texts & right_texts)
    if any(overlaps.values()):
        raise RuntimeError(f"Duplicate texts leaked across splits: {overlaps}")

    per_label = frame.pivot_table(
        index="language", columns="split", values="group_id", aggfunc="nunique", fill_value=0
    )
    details = {
        "rows_per_split": {
            split: int((frame["split"] == split).sum())
            for split in ("train", "validation", "test")
        },
        "languages_per_split": {
            split: int(frame.loc[frame["split"] == split, "language"].nunique())
            for split in ("train", "validation", "test")
        },
        "rows_per_source": {
            str(source): int(count)
            for source, count in frame.groupby("source").size().sort_index().items()
        },
        "rows_per_source_split": {
            split: {
                str(source): int(count)
                for source, count in frame.loc[frame["split"] == split]
                .groupby("source")
                .size()
                .sort_index()
                .items()
            }
            for split in ("train", "validation", "test")
        },
        "text_overlap": overlaps,
        "examples_per_language_by_split": (
            per_label.describe().loc[["min", "max"]].round(1).to_dict()
        ),
    }
    return frame, details


clean_df, split_details = split_dataset(clean_df, CONFIG)
logger.info(json.dumps(split_details, indent=2))
logger.info("rows per split:")
logger.info(clean_df.groupby("split").size().to_string())
logger.info("rows per split and source:")
logger.info(clean_df.groupby(["split", "source"]).size().to_string())

log_section('4. Character vocabulary (training split only)')

def build_vocabulary(texts: list[str]) -> dict[str, int]:
    '''Build a character vocabulary; indices 0 and 1 are reserved for pad and unk.'''
    characters = sorted({character for text in texts for character in text})
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for character in characters:
        vocabulary[character] = len(vocabulary)
    return vocabulary


train_df = clean_df.loc[clean_df["split"] == "train"].reset_index(drop=True)
validation_df = clean_df.loc[clean_df["split"] == "validation"].reset_index(drop=True)
test_df = clean_df.loc[clean_df["split"] == "test"].reset_index(drop=True)

vocabulary = build_vocabulary(train_df["text"].tolist())

unknown_character_rate = {}
for split_name, split_frame in (("train", train_df), ("validation", validation_df), ("test", test_df)):
    total_characters = int(split_frame["text"].str.len().sum())
    unseen_characters = sum(
        character not in vocabulary
        for text in split_frame["text"]
        for character in text
    )
    unknown_character_rate[split_name] = (
        round(unseen_characters / total_characters, 6) if total_characters else 0.0
    )

logger.info(f"vocabulary size: {len(vocabulary):,} ids ({len(vocabulary) - len(SPECIAL_TOKENS):,} characters)")
logger.info(f"unknown character rate: {unknown_character_rate}")
logger.info(f"first characters: {[character for character in vocabulary if character not in SPECIAL_TOKENS][:40]}")

log_section('5. Encode strings as character-ID sequences')

def encode_frame(
    frame: pd.DataFrame, vocabulary: dict[str, int], max_length: int, name: str = "split"
) -> tuple[np.ndarray, np.ndarray]:
    '''Encode texts as right-padded int32 ID matrices and return raw character lengths.'''
    logger.info("encoding %s split: %d rows, max_length=%d", name, len(frame), max_length)
    ids = np.full((len(frame), max_length), PAD_ID, dtype=np.int32)
    lengths = np.zeros(len(frame), dtype=np.int32)
    for row, text in enumerate(frame["text"]):
        encoded = encode_text(text, vocabulary, max_length)
        ids[row, : len(encoded)] = encoded
        lengths[row] = len(text)
        if (row + 1) % 25000 == 0:
            logger.info(
                "  encoded %s rows: %d/%d (%.1f%%)",
                name,
                row + 1,
                len(frame),
                100.0 * (row + 1) / len(frame),
            )
    return ids, lengths


encoded: dict[str, np.ndarray] = {}
character_lengths: dict[str, np.ndarray] = {}
for split_name, split_frame in (("train", train_df), ("validation", validation_df), ("test", test_df)):
    encoded[split_name], character_lengths[split_name] = encode_frame(
        split_frame, vocabulary, CONFIG.max_length, split_name
    )
    truncated = int((character_lengths[split_name] > CONFIG.max_length).sum())
    logger.info(
        f"{split_name:<10} shape={encoded[split_name].shape} "
        f"chars median={np.median(character_lengths[split_name]):.0f} "
        f"p95={np.percentile(character_lengths[split_name], 95):.0f} "
        f"truncated={truncated:,} ({truncated / len(split_frame):.1%})"
    )
    for source, group in split_frame.groupby("source", sort=True):
        lengths = group["text"].str.len()
        source_truncated = int((lengths > CONFIG.max_length).sum())
        logger.info(
            "  %-10s %-10s rows=%7d truncated=%7d (%.1f%%)",
            split_name,
            source,
            len(group),
            source_truncated,
            100.0 * source_truncated / len(group),
        )

log_section('6. PyTorch datasets and data loaders')

class LanguageDataset(Dataset):
    '''Character-ID matrix and integer language label per example.'''

    def __init__(self, ids: np.ndarray, labels: np.ndarray) -> None:
        if len(ids) != len(labels):
            raise ValueError(f"ids and labels length mismatch: {len(ids)} != {len(labels)}")
        self.ids = ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.ids[index].astype(np.int64)),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
        )


labels = sorted(clean_df["language"].unique())
label2idx = {label: index for index, label in enumerate(labels)}
idx2label = {index: label for label, index in label2idx.items()}

datasets: dict[str, LanguageDataset] = {}
loaders: dict[str, DataLoader] = {}
for split_name, split_frame in (("train", train_df), ("validation", validation_df), ("test", test_df)):
    label_ids = split_frame["language"].map(label2idx).to_numpy(dtype=np.int64)
    datasets[split_name] = LanguageDataset(encoded[split_name], label_ids)
    loaders[split_name] = DataLoader(
        datasets[split_name],
        batch_size=CONFIG.batch_size,
        shuffle=split_name == "train",
        num_workers=CONFIG.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(CONFIG.seed) if split_name == "train" else None,
    )

labels_path = CONFIG.processed_dir / "labels.json"
labels_path.parent.mkdir(parents=True, exist_ok=True)
labels_path.write_text(
    json.dumps(
        {
            "label2idx": label2idx,
            "idx2label": {str(index): label for index, label in idx2label.items()},
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

logger.info(f"languages: {len(labels):,}")
logger.info("batches: " + ", ".join(f"{name}={len(loader):,}" for name, loader in loaders.items()))
logger.info(f"label mappings written to {labels_path}")
preview_inputs, preview_targets = next(iter(loaders["train"]))
logger.info(f"batch input shape {tuple(preview_inputs.shape)} dtype {preview_inputs.dtype}")
logger.info(
    f"batch target shape {tuple(preview_targets.shape)} "
    f"example labels {[idx2label[int(index)] for index in preview_targets[:5]]}"
)

log_section('7. Character-level CNN architecture')

def select_device(preference: str) -> torch.device:
    '''Pick CUDA, Apple MPS or CPU according to availability and configuration.'''
    requested = preference.lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device preference {preference!r}")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps":
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device("cpu")


DEVICE = select_device(CONFIG.device)

model = CharCNN(
    vocab_size=len(vocabulary),
    num_classes=len(labels),
    embedding_dim=CONFIG.embedding_dim,
    num_filters=CONFIG.num_filters,
    kernel_sizes=CONFIG.kernel_sizes,
    dropout=CONFIG.dropout,
    padding_idx=PAD_ID,
).to(DEVICE)

example_batch = pad_sequences([[2, 3, 4, 5, 6, 7], [2, 3]], max_length=CONFIG.max_length).to(DEVICE)
with torch.no_grad():
    example_embeddings = model.embedding(example_batch)
    example_logits = model(example_batch)

logger.info("model architecture:\n%s", model)
logger.info(f"device            : {DEVICE}")
logger.info(f"input ids         : {tuple(example_batch.shape)}  (batch, length)")
logger.info(f"embeddings        : {tuple(example_embeddings.shape)}  (batch, length, embedding_dim)")
logger.info(f"logits            : {tuple(example_logits.shape)}  (batch, num_classes)")
logger.info(f"trainable params  : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

log_section('7.1 Network architecture and layer-dimension visualisations')

render_network_architecture(
    model, PLOTS_DIR / "network_architecture.png", CONFIG.max_length
)

layer_dimension_records = collect_layer_dimensions(model, example_batch[:1])
layer_dimension_frame = pd.DataFrame(
    [
        {
            "layer": record["layer"],
            "type": record["type"],
            "input": format_shape(record["input"]),
            "output": format_shape(record["output"]),
            "parameters": int(record["params"]),
        }
        for record in layer_dimension_records
    ]
)
log_frame(layer_dimension_frame, "layer dimensions")
plot_layer_dimensions(
    layer_dimension_records,
    PLOTS_DIR / "layer_dimensions.png",
    total_params=sum(parameter.numel() for parameter in model.parameters()),
)

log_section('8. Training loop')

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    '''Run one training epoch and return mean loss and accuracy.'''
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    batch_count = len(loader)
    started = time.perf_counter()
    logger.info(
        "training epoch: %d batches, batch_size=%s",
        batch_count,
        getattr(loader, "batch_size", "?"),
    )
    for batch_index, (input_ids, targets) in enumerate(loader, start=1):
        if input_ids.numel() == 0:
            logger.warning("skipping empty training batch %d/%d", batch_index, batch_count)
            continue
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip_norm)
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += batch_size

        if batch_index % PROGRESS_EVERY == 0 or batch_index == batch_count:
            elapsed = time.perf_counter() - started
            logger.info(
                "  batch %d/%d (%.1f%%) | running loss %.4f | running accuracy %.4f | %.0f samples/s",
                batch_index,
                batch_count,
                100.0 * batch_index / max(batch_count, 1),
                total_loss / total,
                correct / total,
                total / elapsed if elapsed > 0 else 0.0,
            )

    if total == 0:
        raise RuntimeError("Training loader produced no examples.")
    elapsed = time.perf_counter() - started
    logger.info(
        "training epoch done: loss %.4f | accuracy %.4f | %.0f samples/s",
        total_loss / total,
        correct / total,
        total / elapsed if elapsed > 0 else 0.0,
    )
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    name: str = "evaluation",
) -> tuple[float, float]:
    '''Compute mean loss and accuracy in eval mode without gradients.'''
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    batch_count = len(loader)
    started = time.perf_counter()
    logger.info("%s: %d batches", name, batch_count)
    for batch_index, (input_ids, targets) in enumerate(loader, start=1):
        if input_ids.numel() == 0:
            logger.warning("skipping empty %s batch %d/%d", name, batch_index, batch_count)
            continue
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(input_ids)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss: {loss.item()}")

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += batch_size

        if batch_index % PROGRESS_EVERY == 0 or batch_index == batch_count:
            elapsed = time.perf_counter() - started
            logger.info(
                "  %s batch %d/%d (%.1f%%) | running loss %.4f | running accuracy %.4f | %.0f samples/s",
                name,
                batch_index,
                batch_count,
                100.0 * batch_index / max(batch_count, 1),
                total_loss / total,
                correct / total,
                total / elapsed if elapsed > 0 else 0.0,
            )

    if total == 0:
        raise RuntimeError("Evaluation loader produced no examples.")
    elapsed = time.perf_counter() - started
    logger.info(
        "%s done: loss %.4f | accuracy %.4f | %.0f samples/s",
        name,
        total_loss / total,
        correct / total,
        total / elapsed if elapsed > 0 else 0.0,
    )
    return total_loss / total, correct / total


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay
)

logger.info(
    "training plan: epochs=%d, batch_size=%d, lr=%g, weight_decay=%g, patience=%d, device=%s",
    CONFIG.epochs,
    CONFIG.batch_size,
    CONFIG.learning_rate,
    CONFIG.weight_decay,
    CONFIG.early_stopping_patience,
    DEVICE,
)
history: list[dict] = []
best_state: dict[str, torch.Tensor] | None = None
best_val_loss = float("inf")
best_epoch = 0
training_started = time.perf_counter()

for epoch in range(1, CONFIG.epochs + 1):
    started = time.time()
    train_loss, train_accuracy = train_one_epoch(
        model, loaders["train"], optimizer, criterion, DEVICE
    )
    val_loss, val_accuracy = evaluate_loader(
        model, loaders["validation"], criterion, DEVICE, name="validation"
    )
    elapsed = time.time() - started

    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = {
            name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
        }

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "seconds": round(elapsed, 1),
            "improved": improved,
        }
    )
    logger.info(
        f"epoch {epoch:>2}/{CONFIG.epochs} | "
        f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
        f"val loss {val_loss:.4f} acc {val_accuracy:.4f} | "
        f"{elapsed:5.1f}s {'*' if improved else ''}"
    )

    if epoch - best_epoch >= CONFIG.early_stopping_patience:
        logger.info(
            f"early stopping at epoch {epoch}: validation loss has not improved for "
            f"{CONFIG.early_stopping_patience} epoch(s)."
        )
        break

training_seconds = time.perf_counter() - training_started

if best_state is None:
    raise RuntimeError("Training finished without a checkpoint.")

model.load_state_dict(best_state)

history_df = pd.DataFrame(history)
logger.info(
    "training wall time: %.1f s (%.1f min) across %d epoch(s)",
    training_seconds,
    training_seconds / 60.0,
    len(history_df),
)
best_row = history_df.loc[history_df["epoch"] == best_epoch].iloc[0]
logger.info(
    f"\nbest epoch {best_epoch}: val loss {best_row['val_loss']:.4f}, "
    f"val accuracy {best_row['val_accuracy']:.4f}; restored best checkpoint."
)
log_frame(history_df, "training history")

figure, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train")
axes[0].plot(history_df["epoch"], history_df["val_loss"], marker="o", label="validation")
axes[0].axvline(best_epoch, color="grey", linestyle="--", label=f"best epoch ({best_epoch})")
axes[0].set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss")
axes[0].legend()

axes[1].plot(history_df["epoch"], history_df["train_accuracy"], marker="o", label="train")
axes[1].plot(history_df["epoch"], history_df["val_accuracy"], marker="o", label="validation")
axes[1].axvline(best_epoch, color="grey", linestyle="--")
axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
axes[1].legend()

plt.tight_layout()
save_figure(PLOTS_DIR / "training_curves.png", "training_curves.png")

last_row = history_df.iloc[-1]
logger.info(
    f"epochs run: {len(history_df)} | best epoch: {best_epoch} | "
    f"final train loss/accuracy: {last_row['train_loss']:.4f}/{last_row['train_accuracy']:.4f} | "
    f"final validation loss/accuracy: {last_row['val_loss']:.4f}/{last_row['val_accuracy']:.4f}"
)

log_section('9. Final evaluation on the held-out test set')

def prepare_artifact_dir(config: Config) -> Path:
    '''Create the artifact directory unless an existing version must not be overwritten.'''
    directory = config.artifact_dir
    if directory.exists() and any(directory.iterdir()) and not config.allow_overwrite:
        raise RuntimeError(
            f"{directory} already contains files. Bump CONFIG.model_version, choose another "
            "CONFIG.artifact_dir, or set CONFIG.allow_overwrite = True to replace it."
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


ARTIFACT_DIR = prepare_artifact_dir(CONFIG)

model.eval()
test_probability_batches = []
test_batch_count = len(loaders["test"])
logger.info("running test inference: %d batches", test_batch_count)
test_started = time.perf_counter()
with torch.inference_mode():
    for batch_index, (input_ids, _) in enumerate(loaders["test"], start=1):
        if input_ids.numel() == 0:
            logger.warning("skipping empty test batch %d/%d", batch_index, test_batch_count)
            continue
        logits = model(input_ids.to(DEVICE))
        test_probability_batches.append(torch.softmax(logits, dim=1).cpu())
        if batch_index % PROGRESS_EVERY == 0 or batch_index == test_batch_count:
            logger.info(
                "  test batch %d/%d (%.1f%%) | %.0f samples/s",
                batch_index,
                test_batch_count,
                100.0 * batch_index / max(test_batch_count, 1),
                (batch_index * getattr(loaders["test"], "batch_size", 1))
                / max(time.perf_counter() - test_started, 1e-9),
            )
logger.info("test inference done in %.1f s", time.perf_counter() - test_started)

if not test_probability_batches:
    raise RuntimeError("Test loader produced no examples.")
test_probabilities = torch.cat(test_probability_batches, dim=0)

y_true = test_df["language"].map(label2idx).to_numpy(dtype=np.int64)
if len(test_probabilities) != len(y_true):
    raise RuntimeError(
        f"Prediction/truth length mismatch: {len(test_probabilities)} != {len(y_true)}"
    )
y_pred = test_probabilities.argmax(dim=1).numpy()
y_confidence = test_probabilities.max(dim=1).values.numpy()

logger.info(f"test examples : {len(y_true):,}")
logger.info(f"test accuracy : {(y_pred == y_true).mean():.4f}")
logger.info(f"artifact dir  : {ARTIFACT_DIR}")

def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    '''Accuracy, per-class precision/recall/F1 and macro-F1 using NumPy only.'''
    matrix = np.zeros((len(label_names), len(label_names)), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)

    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)

    precision = np.divide(
        true_positive, predicted, out=np.zeros(len(label_names), dtype=float), where=predicted > 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros(len(label_names), dtype=float), where=support > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(label_names), dtype=float),
        where=(precision + recall) > 0,
    )

    present = support > 0
    per_language = pd.DataFrame(
        {
            "language": label_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    summary = {
        "accuracy": float(true_positive.sum() / matrix.sum()),
        "macro_precision": float(precision[present].mean()),
        "macro_recall": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "weighted_f1": float((f1[present] * support[present]).sum() / support[present].sum()),
        "examples": int(matrix.sum()),
        "languages": int(present.sum()),
    }
    return summary, per_language, matrix


metric_summary, per_language_metrics, confusion = classification_metrics(
    y_true, y_pred, labels
)
logger.info(json.dumps(metric_summary, indent=2))

logger.info("worst 10 languages by F1 (support shown):")
log_frame(per_language_metrics.nsmallest(10, "f1").round(4), "worst 10 languages by F1")
logger.info("best 10 languages by F1:")
log_frame(per_language_metrics.nlargest(10, "f1").round(4), "best 10 languages by F1")

figure, axis = plt.subplots(figsize=(26, 24))
image = axis.imshow(confusion, cmap="Blues", norm=LogNorm(vmin=1, vmax=max(int(confusion.max()), 1)))
axis.set_xticks(range(len(labels)))
axis.set_yticks(range(len(labels)))
axis.set_xticklabels(labels, rotation=90, fontsize=4)
axis.set_yticklabels(labels, fontsize=4)
axis.set(
    title="Confusion matrix — test set (row = expected, column = predicted)",
    xlabel="predicted language",
    ylabel="expected language",
)
figure.colorbar(image, ax=axis, fraction=0.025, pad=0.01)
plt.tight_layout()
save_figure(PLOTS_DIR / "confusion_matrix.png", "confusion_matrix.png")

off_diagonal = confusion.copy()
np.fill_diagonal(off_diagonal, 0)
confusions = []
for flat_index in np.argsort(off_diagonal, axis=None)[::-1][:10]:
    expected_index, predicted_index = np.unravel_index(flat_index, off_diagonal.shape)
    count = int(off_diagonal[expected_index, predicted_index])
    if count == 0:
        break
    confusions.append(
        {
            "expected": labels[expected_index],
            "predicted": labels[predicted_index],
            "count": count,
        }
    )
logger.info("most frequent confusions:")
log_frame(pd.DataFrame(confusions), "most frequent confusions")

results = pd.DataFrame(
    {
        "text": test_df["text"],
        "expected": test_df["language"],
        "source": test_df["source"],
        "dataset_role": test_df["dataset_role"],
        "predicted": [labels[index] for index in y_pred],
        "confidence": y_confidence,
        "expected_idx": y_true,
        "predicted_idx": y_pred,
        "char_length": test_df["text"].str.len(),
    }
)
results["correct"] = results["expected"] == results["predicted"]

misclassified = (
    results.loc[~results["correct"]]
    .sort_values("confidence", ascending=False)
    .reset_index(drop=True)
)
logger.info(
    f"misclassified: {len(misclassified):,} / {len(results):,} "
    f"({len(misclassified) / len(results):.2%})"
)
log_frame(
    misclassified.assign(text=lambda frame: frame["text"].str.slice(0, 110)).head(15),
    "most confident misclassifications",
)

confidence_summary = {
    "mean_confidence": float(results["confidence"].mean()),
    "median_confidence": float(results["confidence"].median()),
    "share_confidence_at_least_0_9": float((results["confidence"] >= 0.9).mean()),
    "accuracy_when_confidence_at_least_0_9": (
        float(results.loc[results["confidence"] >= 0.9, "correct"].mean())
        if (results["confidence"] >= 0.9).any()
        else None
    ),
    "accuracy_when_confidence_below_0_5": (
        float(results.loc[results["confidence"] < 0.5, "correct"].mean())
        if (results["confidence"] < 0.5).any()
        else None
    ),
    "mean_confidence_correct": float(results.loc[results["correct"], "confidence"].mean()),
    "mean_confidence_incorrect": (
        float(results.loc[~results["correct"], "confidence"].mean())
        if (~results["correct"]).any()
        else None
    ),
}
logger.info(json.dumps(confidence_summary, indent=2))
logger.info("Softmax scores are not calibrated probabilities; treat them as ranking scores.")

log_section('9.0 Prediction-probability visualisation')

probability_examples: list[dict[str, str]] = []
seen_texts: set[str] = set()
for subset in (results.loc[results["correct"]], results.loc[~results["correct"]]):
    for _, row in subset.nlargest(3, "confidence").iterrows():
        text = str(row["text"])
        if text in seen_texts:
            continue
        seen_texts.add(text)
        probability_examples.append({"text": text, "expected": str(row["expected"])})
if len(probability_examples) < 6:
    for _, row in results.nlargest(6, "confidence").iterrows():
        text = str(row["text"])
        if text in seen_texts:
            continue
        seen_texts.add(text)
        probability_examples.append({"text": text, "expected": str(row["expected"])})
        if len(probability_examples) >= 6:
            break
plot_prediction_probabilities(
    model,
    probability_examples[:6],
    PLOTS_DIR / "prediction_probabilities.png",
)

log_section('9.1 Analysis by input length and data-quality checks')

length_bins = [0, 100, 200, 400, 800, 1600, np.inf]
length_bucket_labels = ["<100", "100-199", "200-399", "400-799", "800-1599", ">=1600"]
results["length_bucket"] = pd.cut(
    results["char_length"], bins=length_bins, labels=length_bucket_labels, right=False
)

length_summary = (
    results.groupby("length_bucket", observed=True)
    .agg(examples=("correct", "size"), accuracy=("correct", "mean"))
    .reset_index()
)
length_summary["share"] = length_summary["examples"] / len(results)
logger.info(length_summary.round(4).to_string(index=False))

length_summary_by_source = (
    results.groupby(["source", "length_bucket"], observed=True)
    .agg(examples=("correct", "size"), accuracy=("correct", "mean"))
    .reset_index()
)
logger.info("accuracy by input length and source:")
logger.info(length_summary_by_source.round(4).to_string(index=False))

length_correlation = float(
    np.corrcoef(results["char_length"].to_numpy(), results["correct"].to_numpy().astype(float))[0, 1]
)
logger.info(f"correlation between character length and correctness: {length_correlation:.3f}")

figure, axis = plt.subplots(figsize=(9, 4))
sources = sorted(length_summary_by_source["source"].unique())
bar_width = 0.8 / max(len(sources), 1)
for offset, source in enumerate(sources):
    subset = (
        length_summary_by_source.loc[length_summary_by_source["source"] == source]
        .set_index("length_bucket")
        .reindex(length_bucket_labels)
    )
    positions = np.arange(len(length_bucket_labels)) + (
        offset - (len(sources) - 1) / 2
    ) * bar_width
    axis.bar(
        positions,
        subset["accuracy"].to_numpy(),
        width=bar_width,
        label=source,
    )
axis.set_xticks(range(len(length_bucket_labels)))
axis.set_xticklabels(length_bucket_labels, rotation=20)
axis.set(
    title="Test accuracy by input length and source",
    xlabel="characters",
    ylabel="accuracy",
    ylim=(0, 1.02),
)
axis.legend(title="source")
plt.tight_layout()
save_figure(PLOTS_DIR / "accuracy_by_length.png", "accuracy_by_length.png")

test_count_min = int(test_df["language"].value_counts().min())
test_count_max = int(test_df["language"].value_counts().max())
leakage_checks = {
    "duplicate_texts_train_vs_validation": len(set(train_df["text"]) & set(validation_df["text"])),
    "duplicate_texts_train_vs_test": len(set(train_df["text"]) & set(test_df["text"])),
    "duplicate_texts_validation_vs_test": len(set(validation_df["text"]) & set(test_df["text"])),
    "duplicate_rows_test": int(test_df.duplicated(subset=["text", "language"]).sum()),
    "evaluation_languages_missing_from_train": len(
        (set(validation_df["language"]) | set(test_df["language"]))
        - set(train_df["language"])
    ),
    "test_examples_per_language_min": test_count_min,
    "test_examples_per_language_max": test_count_max,
    "class_imbalance_ratio": round(test_count_max / test_count_min, 3) if test_count_min else None,
    "suspiciously_high_accuracy": bool(metric_summary["accuracy"] > 0.999),
    "test_examples": int(len(test_df)),
    "test_rows_per_source": {
        str(source): int(count) for source, count in results["source"].value_counts().items()
    },
}
logger.info(json.dumps(leakage_checks, indent=2))

leaked = max(
    leakage_checks["duplicate_texts_train_vs_validation"],
    leakage_checks["duplicate_texts_train_vs_test"],
    leakage_checks["duplicate_texts_validation_vs_test"],
)
if leaked:
    raise RuntimeError("Text leakage detected between splits; review the split code.")
if leakage_checks["evaluation_languages_missing_from_train"]:
    raise RuntimeError("Some evaluation languages never appear in training.")
if leakage_checks["suspiciously_high_accuracy"]:
    logger.info("WARNING: near-perfect accuracy is unusual; check for duplicated documents or label leakage.")

if len(misclassified):
    logger.info("weakest languages by F1 among the worst 10:")
    weakest = per_language_metrics.nsmallest(10, "f1")["language"].tolist()
    log_frame(
        results[results["expected"].isin(weakest)]
        .groupby("expected", observed=True)
        .agg(
            examples=("correct", "size"),
            accuracy=("correct", "mean"),
            median_length=("char_length", "median"),
        )
        .sort_values("accuracy")
        .round(4),
        "weakest languages among the worst 10",
    )

log_section('9.2 Persist evaluation results')

def write_text_atomic(path: Path, content: str) -> None:
    '''Write text through a temporary file and atomically replace the target.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
    logger.info("wrote %s (%.1f KiB)", path, len(content.encode("utf-8")) / 1024)


def write_json_atomic(path: Path, payload: object) -> None:
    '''Write JSON (sorted-stable formatting) atomically.'''
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    '''Return the SHA-256 hex digest of a file.'''
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


created_at = datetime.now(timezone.utc).isoformat()

source_summary = (
    clean_df.groupby("source")
    .agg(rows=("text", "size"), languages=("language", "nunique"))
    .reset_index()
)
source_summary["source"] = source_summary["source"].astype(str)

by_source_metrics: dict[str, dict] = {}
for source_name, source_results in results.groupby("source"):
    correct = int((source_results["predicted"] == source_results["expected"]).sum())
    by_source_metrics[str(source_name)] = {
        "accuracy": float(correct / len(source_results)),
        "examples": int(len(source_results)),
        "languages": int(source_results["expected"].nunique()),
        "predictions": {
            str(label): int(count)
            for label, count in source_results["predicted"].value_counts().items()
        },
    }

evaluation_payload = {
    "model_version": CONFIG.model_version,
    "created_at": created_at,
    "datasets": [
        {
            "name": source.name,
            "path": str(source.path),
            "format": source.format,
            "role": source.role,
            "rows_used": int((clean_df["source"] == source.name).sum()),
        }
        for source in CONFIG.sources
    ],
    "dataset_rows": int(len(clean_df)),
    "split_rows": split_details["rows_per_split"],
    "split_rows_by_source": split_details["rows_per_source_split"],
    "max_length": CONFIG.max_length,
    "vocabulary_size": len(vocabulary),
    "num_classes": len(labels),
    "best_epoch": int(best_epoch),
    "best_val_loss": float(best_val_loss),
    "best_val_accuracy": float(best_row["val_accuracy"]),
    "test": metric_summary,
    "test_by_source": by_source_metrics,
    "per_language": {
        row["language"]: {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "support": int(row["support"]),
        }
        for _, row in per_language_metrics.iterrows()
    },
    "length_buckets": length_summary.astype({"length_bucket": str}).to_dict(orient="records"),
    "length_buckets_by_source": length_summary_by_source.astype(
        {"length_bucket": str}
    ).to_dict(orient="records"),
    "source_summary": source_summary.to_dict(orient="records"),
    "confidence": confidence_summary,
    "leakage_checks": leakage_checks,
    "device": str(DEVICE),
}

write_json_atomic(ARTIFACT_DIR / "metrics.json", evaluation_payload)
write_text_atomic(
    ARTIFACT_DIR / "confusion_matrix.csv",
    pd.DataFrame(confusion, index=labels, columns=labels).to_csv(),
)
write_text_atomic(
    ARTIFACT_DIR / "misclassified_examples.csv",
    misclassified.head(CONFIG.max_misclassified_examples).to_csv(index=False),
)
write_text_atomic(ARTIFACT_DIR / "training_history.csv", history_df.to_csv(index=False))

kept_errors = min(len(misclassified), CONFIG.max_misclassified_examples)
logger.info(f"wrote {ARTIFACT_DIR / 'metrics.json'}")
logger.info(f"wrote {ARTIFACT_DIR / 'confusion_matrix.csv'}")
logger.info(f"wrote {ARTIFACT_DIR / 'misclassified_examples.csv'} ({kept_errors:,} rows)")
logger.info(f"wrote {ARTIFACT_DIR / 'training_history.csv'} ({len(history_df)} epochs)")

log_section('10. Export a versioned, self-contained artifact')

export_config = {
    "schema_version": 1,
    "model_type": "character_cnn",
    "framework": "pytorch",
    "torch_version": torch.__version__,
    "embedding_dim": int(CONFIG.embedding_dim),
    "num_filters": [int(value) for value in CONFIG.num_filters],
    "kernel_sizes": [int(value) for value in CONFIG.kernel_sizes],
    "dropout": float(CONFIG.dropout),
    "vocab_size": int(len(vocabulary)),
    "num_classes": int(len(labels)),
    "max_length": int(CONFIG.max_length),
    "padding_idx": int(PAD_ID),
    "unknown_idx": int(UNK_ID),
    "special_tokens": list(SPECIAL_TOKENS),
    "padding_side": "right",
    "pooling": "masked_max",
    "normalisation": "NFC + whitespace collapse",
    "lowercase": False,
    "prediction_scope": "single-label-per-string",
}

write_json_atomic(ARTIFACT_DIR / "config.json", export_config)
write_json_atomic(ARTIFACT_DIR / "vocabulary.json", vocabulary)
write_json_atomic(
    ARTIFACT_DIR / "labels.json",
    {
        "label2idx": label2idx,
        "idx2label": {str(index): label for index, label in idx2label.items()},
    },
)

model_path = ARTIFACT_DIR / "model.pt"
temporary_model_path = model_path.with_name(model_path.name + ".tmp")
torch.save(
    {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
    temporary_model_path,
)
os.replace(temporary_model_path, model_path)

write_text_atomic(ARTIFACT_DIR / "linguae_inference.py", INFERENCE_MODULE_SOURCE)

example_source = '''# Minimal standalone example for the exported Linguae artifact.
#
# Run from this directory:  python example_inference.py

from pathlib import Path

from linguae_inference import load_artifact, predict_text

ARTIFACT_DIR = Path(__file__).resolve().parent
artifact = load_artifact(ARTIFACT_DIR, device="cpu")

for text in [
    "Bonjour tout le monde, comment allez-vous ?",
    "Hello world",
    "Привет, мир",
]:
    prediction = predict_text(text, artifact)
    print(f"{prediction['language']} ({prediction['confidence']:.3f}) <- {text}")
'''
write_text_atomic(ARTIFACT_DIR / "example_inference.py", example_source)

logger.info("artifact directory: %s", ARTIFACT_DIR)
for path in sorted(ARTIFACT_DIR.iterdir()):
    logger.info(f"  {path.name:<32} {path.stat().st_size / 1024:>9.1f} KiB")

hashed_files = [
    "config.json",
    "vocabulary.json",
    "labels.json",
    "model.pt",
    "metrics.json",
    "linguae_inference.py",
    "example_inference.py",
    "confusion_matrix.csv",
    "misclassified_examples.csv",
    "training_history.csv",
]

manifest = {
    "schema_version": 1,
    "artifact_schema": "linguae.language-detector/1",
    "model_version": CONFIG.model_version,
    "created_at": created_at,
    "datasets": [
        {
            "name": source.name,
            "path": str(source.path),
            "sha256": sha256_file(source.path),
            "format": source.format,
            "role": source.role,
            "text_column": source.text_column,
            "label_column": source.label_column,
            "rows_used": int((clean_df["source"] == source.name).sum()),
        }
        for source in CONFIG.sources
    ],
    "dataset_rows": int(len(clean_df)),
    "training": {
        "seed": CONFIG.seed,
        "train_fraction": CONFIG.train_fraction,
        "val_fraction": CONFIG.val_fraction,
        "test_fraction": CONFIG.test_fraction,
        "max_length": CONFIG.max_length,
        "embedding_dim": CONFIG.embedding_dim,
        "num_filters": list(CONFIG.num_filters),
        "kernel_sizes": list(CONFIG.kernel_sizes),
        "dropout": CONFIG.dropout,
        "batch_size": CONFIG.batch_size,
        "epochs_configured": CONFIG.epochs,
        "epochs_run": int(len(history_df)),
        "training_seconds": round(training_seconds, 1),
        "best_epoch": int(best_epoch),
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss",
        "learning_rate": CONFIG.learning_rate,
        "weight_decay": CONFIG.weight_decay,
        "grad_clip_norm": CONFIG.grad_clip_norm,
        "early_stopping_patience": CONFIG.early_stopping_patience,
        "device": str(DEVICE),
        "min_examples_per_language": CONFIG.min_examples_per_language,
        "max_rows_per_language_per_source": CONFIG.max_rows_per_language_per_source,
        "max_rows_per_language": CONFIG.max_rows_per_language,
    },
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    },
    "files": {name: sha256_file(ARTIFACT_DIR / name) for name in hashed_files},
    "prediction_scope": (
        "single-language classification per string; mixed-language span detection is not supported"
    ),
}

write_json_atomic(ARTIFACT_DIR / "manifest.json", manifest)
logger.info(f"wrote {ARTIFACT_DIR / 'manifest.json'}")
for file_name, digest in manifest["files"].items():
    logger.info("  sha256 %s  %s", digest, file_name)
logger.info(json.dumps({"model_version": manifest["model_version"], "files": manifest["files"]}, indent=2))

reloaded = load_artifact(ARTIFACT_DIR, device="cpu")

fixed_texts = test_df["text"].head(3).tolist() + [
    "Bonjour tout le monde",
    "The quick brown fox jumps over the lazy dog.",
]
fixed_sequences = [encode_text(text, vocabulary, CONFIG.max_length) for text in fixed_texts]
fixed_batch = pad_sequences(fixed_sequences, max_length=CONFIG.max_length)
model.cpu()
model.eval()

with torch.inference_mode():
    reference_probabilities = torch.softmax(model(fixed_batch), dim=1)
    reloaded_probabilities = torch.softmax(reloaded["model"](fixed_batch), dim=1)

maximum_difference = float((reference_probabilities - reloaded_probabilities).abs().max())
reference_predictions = reference_probabilities.argmax(dim=1).tolist()
reloaded_predictions = reloaded_probabilities.argmax(dim=1).tolist()

if reference_predictions != reloaded_predictions:
    raise RuntimeError(
        f"Artifact predictions differ from the trained model: "
        f"{reference_predictions} != {reloaded_predictions}"
    )
if maximum_difference > 1e-5:
    raise RuntimeError(f"Artifact probabilities differ by {maximum_difference:.2e}")

logger.info(f"verification: {len(fixed_texts)} fixed strings, max probability difference {maximum_difference:.2e}")
logger.info("artifact predictions match the trained model.\n")

logger.info("example inference from the reloaded artifact:")
for text in fixed_texts[-2:] + fixed_texts[:1]:
    prediction = predict_text(text, reloaded, top_k=3)
    top_three = ", ".join(
        f"{item['language']}={item['score']:.3f}" for item in prediction["top_k"]
    )
    logger.info(f"  {text[:70]!r} -> {prediction['language']} ({prediction['confidence']:.3f}) | {top_three}")

logger.info("generated artifact files:")
for path in sorted(ARTIFACT_DIR.iterdir()):
    logger.info(f"  {path}")
logger.info(f"standalone module : {ARTIFACT_DIR / 'linguae_inference.py'}")
logger.info(f"example script    : {ARTIFACT_DIR / 'example_inference.py'}")
logger.info(f"run it with       : cd {ARTIFACT_DIR} && python example_inference.py")
logger.info("saved images in %s", PLOTS_DIR)
logger.info("=" * 78)
logger.info(
    "pipeline complete in %.1f s (%.1f min)",
    time.perf_counter() - SCRIPT_STARTED,
    (time.perf_counter() - SCRIPT_STARTED) / 60.0,
)
logger.info("training wall time: %.1f s (%.1f min)", training_seconds, training_seconds / 60.0)
logger.info("artifact directory: %s", ARTIFACT_DIR)
logger.info("=" * 78)
