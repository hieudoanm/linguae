#!/usr/bin/env python3
"""Export the linguae CharCNN artifact to self-contained single-file formats.

Reads config.json, vocabulary.json and model.safetensors from the artifact
directory, rebuilds the model, sanity-checks a few predictions, and writes
three files that each bundle weights + config + vocab + labels:

    linguae.pt      PyTorch checkpoint (torch.load)
    linguae.pkl     pickle of config + numpy weights + vocab + labels
    linguae.joblib  same payload via joblib (joblib.load)

Run with the bundled venv (Python 3.13):
    .venv/bin/python export_models.py [artifact_dir] [out_dir]

Defaults: ../artifacts/linguae, ../artifacts/models
"""

import json
import pickle
import struct
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """Minimal safetensors reader (F32/F16/BF16/F64/I64 only as needed)."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        tensors = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = meta["dtype"]
            shape = meta["shape"]
            begin, end = meta["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            np_dtype = {
                "F32": np.float32,
                "F64": np.float64,
                "F16": np.float16,
                "BF16": np.uint16,  # handled below
                "I64": np.int64,
                "I32": np.int32,
            }[dtype]
            if dtype == "BF16":
                arr = (
                    np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                ).view(np.float32)
            else:
                arr = np.frombuffer(raw, dtype=np_dtype)
            tensors[name] = torch.from_numpy(arr.reshape(shape).copy())
        return tensors


class CharCNN(nn.Module):
    """Mirror of crates/linguae-core/src/model.rs CharCnn::forward."""

    def __init__(self, cfg: dict, sd: dict[str, torch.Tensor]):
        super().__init__()
        self.embedding = nn.Embedding(
            cfg["vocab_size"], cfg["embedding_dim"], padding_idx=PAD_ID
        )
        self.embedding.weight.data = sd["embedding.weight"]

        self.convolutions = nn.ModuleList()
        for i, (filters, kernel) in enumerate(
            zip(cfg["num_filters"], cfg["kernel_sizes"])
        ):
            conv = nn.Conv1d(cfg["embedding_dim"], filters, kernel, padding=kernel // 2)
            conv.weight.data = sd[f"convolutions.{i}.weight"]
            conv.bias.data = sd[f"convolutions.{i}.bias"]
            self.convolutions.append(conv)

        in_features = sum(cfg["num_filters"]) * (2 if cfg["mean_pooling"] else 1)
        if cfg.get("length_embedding"):
            in_features += cfg["length_embedding_dim"]
        self.classifier = nn.Linear(in_features, cfg["num_classes"])
        self.classifier.weight.data = sd["classifier.weight"]
        self.classifier.bias.data = sd["classifier.bias"]

        self.mean_pooling = cfg["mean_pooling"]
        self.length_embedding = (
            nn.Embedding(6, cfg["length_embedding_dim"])
            if cfg.get("length_embedding")
            else None
        )
        if self.length_embedding is not None:
            self.length_embedding.weight.data = sd["length_embedding.weight"]

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, L] int64, mask: [B, 1, L] float32 (1 = real token)
        x = self.embedding(input_ids).transpose(1, 2) * mask  # [B, E, L]
        lengths = mask.squeeze(1).sum(1)  # [B] float, integer-valued

        pooled = []
        for conv in self.convolutions:
            features = F.relu(conv(x)) * mask  # [B, F, L]
            pooled.append(features.amax(dim=2))
            if self.mean_pooling:
                count = lengths.clamp(min=1.0).unsqueeze(1)  # [B, 1]
                pooled.append(features.sum(2) / count)

        if self.length_embedding is not None:
            # Buckets: <=10 -> 0, <=25 -> 1, <=50 -> 2, <=100 -> 3,
            # <=160 -> 4, else 5  (== count of thresholds exceeded)
            bounds = torch.tensor(
                [10.0, 25.0, 50.0, 100.0, 160.0], device=lengths.device
            )
            buckets = (lengths.unsqueeze(1) > bounds).sum(1).long()
            pooled.append(self.length_embedding(buckets))

        return self.classifier(torch.cat(pooled, dim=1))


def normalize_text(text: str) -> str:
    """Mirror of crates/linguae-core/src/text.rs normalize_text."""
    normalized = unicodedata.normalize("NFC", text)
    out = []
    pending_space = False
    for ch in normalized:
        if ch.isspace() or "\x1c" <= ch <= "\x1f":
            if out:
                pending_space = True
        else:
            if pending_space:
                out.append(" ")
                pending_space = False
            out.append(ch)
    return "".join(out)


def encode(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    ids = [vocab.get(ch, 1) for ch in normalize_text(text)[:max_length]]
    return ids or [1]


def export_pt(model: CharCNN, cfg: dict, vocab: dict, labels: dict, out_path: Path) -> None:
    torch.save(
        {
            "config": cfg,
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "labels": labels,
        },
        out_path,
    )


def export_pickle_payload(
    model: CharCNN, cfg: dict, vocab: dict, labels: dict, out_dir: Path
) -> dict:
    payload = {
        "config": cfg,
        "weights": {k: v.detach().numpy() for k, v in model.state_dict().items()},
        "vocab": vocab,
        "labels": labels,
    }
    with open(out_dir / "linguae.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        import joblib

        joblib.dump(payload, out_dir / "linguae.joblib")
    except ImportError:
        # joblib files are plain pickles; joblib.load() reads them either way.
        with open(out_dir / "linguae.joblib", "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> int:
    artifact = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[1] / "artifacts" / "linguae"
    )
    out_dir = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path(__file__).resolve().parents[1] / "artifacts" / "models"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((artifact / "config.json").read_text())
    vocab = json.loads((artifact / "vocabulary.json").read_text())
    labels = json.loads((artifact / "labels.json").read_text())["idx2label"]
    sd = load_safetensors(artifact / "model.safetensors")

    model = CharCNN(cfg, sd).eval()

    # Sanity check: a few sentences should hit their obvious labels.
    samples = [
        (
            "This is a longer English sentence that should be much easier for the model to classify correctly because it contains many common English words and patterns.",
            "eng",
        ),
        (
            "Bonjour, comment allez-vous aujourd'hui? Je voudrais un café s'il vous plaît.",
            "fra",
        ),
        ("Das ist ein deutscher Satz zum Testen des Modells.", "deu"),
        ("Привет, как дела? Это тестовое предложение на русском языке.", "rus"),
        ("Ciao, come stai? Questo è un test in italiano.", "ita"),
    ]
    max_len = cfg["max_length"]
    with torch.no_grad():
        for text, expected in samples:
            ids = encode(text, vocab, max_len)
            padded = ids + [PAD_ID] * (max_len - len(ids))
            input_ids = torch.tensor([padded], dtype=torch.long)
            mask = torch.zeros(1, 1, max_len)
            mask[0, 0, : len(ids)] = 1.0
            logits = model(input_ids, mask)
            top = logits.argmax(-1).item()
            top_label = labels[str(top)]
            status = "OK " if top_label == expected else "MISS"
            print(f"{status} {text!r:45} -> {top_label} (expected {expected})")
            if top_label != expected:
                print(
                    "     top5:",
                    sorted(
                        (
                            (labels[str(i)], float(logits[0, i]))
                            for i in range(len(labels))
                        ),
                        key=lambda t: -t[1],
                    )[:5],
                )
                return 1

    export_pt(model, cfg, vocab, labels, out_dir / "linguae.pt")
    print(f"exported {out_dir / 'linguae.pt'}")

    export_pickle_payload(model, cfg, vocab, labels, out_dir)
    print(f"exported {out_dir / 'linguae.pkl'} and {out_dir / 'linguae.joblib'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
