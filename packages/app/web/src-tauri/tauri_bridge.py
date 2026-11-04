"""JSON bridge for Tauri desktop invocation.

Reads a JSON request from stdin, runs inference using the bundled Linguae
artifact, and writes a JSON response to stdout. This lets the Tauri Rust backend
call into the Python inference code without embedding CPython.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# In development this file lives next to ``inference.py`` in ``src-tauri``; in a
# bundle Tauri copies both files into the same resource directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_ARTIFACT_DIR = Path(
    os.environ.get("LINGUAE_ARTIFACT_DIR", str(_SCRIPT_DIR))
)

from inference import load_artifact, predict_text  # noqa: E402


def main() -> int:
    raw = sys.stdin.read()
    request = json.loads(raw) if raw else {}
    text = request.get("text", "")
    top_k = int(request.get("top_k", request.get("topK", 3)))

    artifact = load_artifact(_ARTIFACT_DIR)
    prediction = predict_text(text, artifact, top_k=top_k)
    print(json.dumps(prediction))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())