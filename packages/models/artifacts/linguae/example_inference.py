# Minimal standalone example for the exported Linguae artifact.
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
