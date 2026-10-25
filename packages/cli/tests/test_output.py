"""Tests for linguae.output module."""

from __future__ import annotations

import json

from linguae.output import json_line


def test_json_line_basic():
    """Test json_line with basic prediction."""
    prediction = {
        "text": "Hello world",
        "language": "eng",
        "confidence": 0.95,
    }
    result = json_line(prediction)
    data = json.loads(result)
    assert data["text"] == "Hello world"
    assert data["language"] == "eng"
    assert data["confidence"] == 0.95
    assert "top_k" not in data


def test_json_line_with_top_k():
    """Test json_line with top_k alternatives."""
    prediction = {
        "text": "Hello world",
        "language": "eng",
        "confidence": 0.95,
        "top_k": [
            {"language": "eng", "score": 0.95},
            {"language": "sco", "score": 0.03},
            {"language": "ibo", "score": 0.01},
        ],
    }
    result = json_line(prediction)
    data = json.loads(result)
    assert data["text"] == "Hello world"
    assert data["language"] == "eng"
    assert data["confidence"] == 0.95
    assert "top_k" in data
    assert len(data["top_k"]) == 3
    assert data["top_k"][0]["language"] == "eng"
    assert data["top_k"][0]["score"] == 0.95
    assert data["top_k"][1]["language"] == "sco"
    assert data["top_k"][1]["score"] == 0.03


def test_json_line_confidence_rounding():
    """Test that confidence is rounded to 6 decimal places."""
    prediction = {
        "text": "Hello world",
        "language": "eng",
        "confidence": 0.95123456789,
    }
    result = json_line(prediction)
    data = json.loads(result)
    assert data["confidence"] == 0.951235


def test_json_line_score_rounding():
    """Test that top_k scores are rounded to 6 decimal places."""
    prediction = {
        "text": "Hello world",
        "language": "eng",
        "confidence": 0.95,
        "top_k": [
            {"language": "eng", "score": 0.95123456789},
            {"language": "sco", "score": 0.03123456789},
        ],
    }
    result = json_line(prediction)
    data = json.loads(result)
    assert data["top_k"][0]["score"] == 0.951235
    assert data["top_k"][1]["score"] == 0.031235


def test_json_line_ensure_ascii_false():
    """Test that non-ASCII text is preserved."""
    prediction = {
        "text": "Привет мир",
        "language": "rus",
        "confidence": 0.9,
    }
    result = json_line(prediction)
    # Should not have escaped unicode
    assert "Привет мир" in result
    assert "\\u" not in result