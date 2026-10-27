"""Standalone inference utilities for the exported Linguae language detector.

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
