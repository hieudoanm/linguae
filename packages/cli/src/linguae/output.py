"""Output formatting for the Linguae CLI."""

from __future__ import annotations

import json


def json_line(prediction: dict) -> str:
    """Format one prediction as a JSON object."""
    payload = {
        "text": prediction["text"],
        "language": prediction["language"],
        "confidence": round(float(prediction["confidence"]), 6),
    }
    if "top_k" in prediction:
        payload["top_k"] = [
            {
                "language": item["language"],
                "score": round(float(item["score"]), 6),
            }
            for item in prediction["top_k"]
        ]
    return json.dumps(payload, ensure_ascii=False)