//! Command-line interface for the Linguae language detector.

use std::io::{IsTerminal, Read};
use std::path::PathBuf;

use clap::{CommandFactory, Parser};
use linguae_core::{Artifact, Prediction};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "linguae",
    version,
    about = "Detect the language of text with the Linguae character-level model.",
    long_about = None
)]
struct Cli {
    /// Text to classify. Reads stdin when omitted.
    #[arg(value_name = "TEXT")]
    text: Option<String>,

    /// Classify each non-empty line of a UTF-8 file.
    #[arg(short = 'f', long = "file", value_name = "PATH")]
    input_file: Option<PathBuf>,

    /// Artifact directory (default: crates/linguae-train/artifacts/linguae).
    #[arg(
        short = 'a',
        long = "artifact",
        value_name = "PATH",
        env = "LINGUAE_ARTIFACT_DIR"
    )]
    artifact: Option<PathBuf>,

    /// Device for inference: cpu, cuda or metal (mps is an alias for metal).
    #[arg(long, default_value = "cpu", value_name = "TEXT")]
    device: String,

    /// Alternatives included in JSON output.
    #[arg(
        short = 'k',
        long = "top-k",
        default_value_t = 3,
        value_parser = clap::value_parser!(u16).range(1..=20),
        value_name = "INT"
    )]
    top_k: u16,

    /// Print the confidence score after the language code.
    #[arg(short = 'c', long = "confidence")]
    confidence: bool,

    /// Emit JSON Lines instead of plain text.
    #[arg(long = "json")]
    json: bool,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {}", error.message);
            std::process::ExitCode::from(error.code)
        }
    }
}

#[derive(Debug)]
struct CliError {
    message: String,
    code: u8,
}

impl CliError {
    fn usage(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: 2,
        }
    }

    fn runtime(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: 1,
        }
    }
}

fn run(cli: &Cli) -> Result<(), CliError> {
    let (texts, is_batch) = gather_texts(cli)?;
    let artifact_dir = linguae_core::resolve(cli.artifact.as_deref())
        .map_err(|error| CliError::runtime(error.to_string()))?;
    let artifact = Artifact::load(&artifact_dir, &cli.device)
        .map_err(|error| CliError::runtime(error.to_string()))?;

    let predictions = if is_batch {
        artifact
            .predict_batch(&texts, linguae_core::DEFAULT_BATCH_SIZE)
            .map_err(|error| CliError::runtime(error.to_string()))?
    } else {
        vec![artifact
            .predict_text(&texts[0], usize::from(cli.top_k))
            .map_err(|error| CliError::runtime(error.to_string()))?]
    };

    for prediction in &predictions {
        if cli.json {
            println!("{}", json_line(prediction));
        } else if cli.confidence {
            println!("{}\t{:.4}", prediction.language, prediction.confidence);
        } else {
            println!("{}", prediction.language);
        }
    }

    Ok(())
}

fn gather_texts(cli: &Cli) -> Result<(Vec<String>, bool), CliError> {
    if cli.input_file.is_some() && cli.text.is_some() {
        return Err(CliError::usage("provide either TEXT or --file, not both"));
    }

    if let Some(path) = &cli.input_file {
        let content = std::fs::read_to_string(path).map_err(|error| {
            CliError::usage(format!("could not read {}: {error}", path.display()))
        })?;
        let lines: Vec<String> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::to_string)
            .collect();
        if lines.is_empty() {
            return Err(CliError::usage(format!(
                "{} contains no non-empty lines",
                path.display()
            )));
        }
        return Ok((lines, true));
    }

    if let Some(text) = &cli.text {
        let stripped = text.trim();
        if stripped.is_empty() {
            return Err(CliError::usage("text is empty"));
        }
        return Ok((vec![stripped.to_string()], false));
    }

    if std::io::stdin().is_terminal() {
        Cli::command().print_help().ok();
        println!();
        std::process::exit(1);
    }

    let mut content = String::new();
    std::io::stdin()
        .read_to_string(&mut content)
        .map_err(|error| CliError::usage(format!("could not read stdin: {error}")))?;
    let stripped = content.trim();
    if stripped.is_empty() {
        return Err(CliError::usage("no input text received on stdin"));
    }
    Ok((vec![stripped.to_string()], false))
}

#[derive(Serialize)]
struct JsonAlternative<'a> {
    language: &'a str,
    score: f64,
}

#[derive(Serialize)]
struct JsonPrediction<'a> {
    text: &'a str,
    language: &'a str,
    confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<Vec<JsonAlternative<'a>>>,
}

fn json_line(prediction: &Prediction) -> String {
    let payload = JsonPrediction {
        text: &prediction.text,
        language: &prediction.language,
        confidence: round6(prediction.confidence),
        top_k: prediction.top_k.as_ref().map(|alternatives| {
            alternatives
                .iter()
                .map(|alternative| JsonAlternative {
                    language: &alternative.language,
                    score: round6(alternative.score),
                })
                .collect()
        }),
    };
    serde_json::to_string(&payload).expect("a prediction always serializes to JSON")
}

fn round6(value: f32) -> f64 {
    let value = f64::from(value);
    (value * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_to_six_decimal_places() {
        assert_eq!(round6(0.951_234_6), 0.951235);
        assert_eq!(round6(0.95), 0.95);
    }
}
