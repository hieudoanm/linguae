"""Command-line interface for the Linguae language detector."""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Detect the language of text with the Linguae character-level model.",
)


def _get_dependencies():
    """Lazy import of dependencies to allow mocking in tests."""
    from linguae.artifact import load_detector, resolve_artifact
    from linguae.input import gather_texts
    from linguae.output import json_line
    from linguae.version import version
    return load_detector, resolve_artifact, gather_texts, json_line, version


@app.command()
def main(
    ctx: typer.Context,
    text: str | None = typer.Argument(
        None, help="Text to classify. Reads stdin when omitted."
    ),
    input_file: str | None = typer.Option(
        None,
        "--file",
        "-f",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Classify each non-empty line of a UTF-8 file.",
    ),
    artifact: str | None = typer.Option(
        None,
        "--artifact",
        "-a",
        envvar="LINGUAE_ARTIFACT_DIR",
        help="Artifact directory (default: packages/models/artifacts/linguae).",
    ),
    device: str = typer.Option(
        "cpu", help="Torch device for inference, for example cpu, mps or cuda."
    ),
    top_k: int = typer.Option(
        3,
        "--top-k",
        "-k",
        min=1,
        max=20,
        help="Alternatives included in JSON output.",
    ),
    confidence: bool = typer.Option(
        False,
        "--confidence",
        "-c",
        help="Print the confidence score after the language code.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON Lines instead of plain text."
    ),
    show_version: bool = typer.Option(
        False, "--version", "-V", help="Print the CLI version and exit."
    ),
) -> None:
    """Detect the language of TEXT, a file, or standard input."""
    load_detector, resolve_artifact, gather_texts, json_line, version = _get_dependencies()

    if show_version:
        typer.echo(f"linguae {version()}")
        raise typer.Exit()

    # Convert string paths to Path objects
    from pathlib import Path
    artifact_path = Path(artifact) if artifact else None
    file_path = Path(input_file) if input_file else None

    texts, is_batch = gather_texts(ctx, text, file_path)
    artifact_dir = resolve_artifact(artifact_path)
    module, detector = load_detector(artifact_dir, device)

    if is_batch:
        predictions = module.predict_batch(texts, detector)
    else:
        predictions = [module.predict_text(texts[0], detector, top_k=top_k)]

    for prediction in predictions:
        if json_output:
            typer.echo(json_line(prediction))
        elif confidence:
            typer.echo(f"{prediction['language']}\t{prediction['confidence']:.4f}")
        else:
            typer.echo(prediction["language"])


if __name__ == "__main__":
    app()