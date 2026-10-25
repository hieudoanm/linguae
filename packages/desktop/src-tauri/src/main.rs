// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::path::PathBuf;
use std::sync::OnceLock;

use linguae_core::{Artifact, Prediction};
use serde::Serialize;

/// Artifact directory resolved at startup. Defaults to the repository's
/// `crates/linguae-train/artifacts/linguae`, overridable via
/// `LINGUAE_ARTIFACT_DIR` or `--artifact`.
static ARTIFACT_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Inference device, overridable via `LINGUAE_DEVICE`.
static DEVICE: OnceLock<String> = OnceLock::new();

#[derive(Serialize)]
struct PredictionResponse {
    text: String,
    language: String,
    confidence: f64,
    top_k: Option<Vec<AlternativeResponse>>,
}

#[derive(Serialize)]
struct AlternativeResponse {
    language: String,
    score: f64,
}

fn artifact_dir() -> &'static PathBuf {
    ARTIFACT_DIR.get().expect("artifact directory not initialised")
}

fn device() -> &'static str {
    DEVICE.get().map(String::as_str).unwrap_or("cpu")
}

/// Load the artifact once and reuse it for every command.
fn artifact() -> &'static Artifact {
    use std::sync::OnceLock;
    static ARTIFACT: OnceLock<Artifact> = OnceLock::new();
    ARTIFACT.get_or_init(|| {
        Artifact::load(artifact_dir(), device())
            .expect("failed to load Linguae artifact")
    })
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! Welcome to Linguae.", name)
}

#[tauri::command]
fn detect(text: String, top_k: Option<u32>) -> PredictionResponse {
    let k = top_k.unwrap_or(3).clamp(1, 20) as usize;
    let prediction = artifact()
        .predict_text(&text, k)
        .expect("inference failed");

    PredictionResponse {
        text: prediction.text,
        language: prediction.language,
        confidence: prediction.confidence as f64,
        top_k: prediction.top_k.map(|alternatives| {
            alternatives
                .iter()
                .map(|alternative| AlternativeResponse {
                    language: alternative.language.clone(),
                    score: alternative.score as f64,
                })
                .collect()
        }),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let artifact_dir = std::env::var("LINGUAE_ARTIFACT_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("crates")
                .join("linguae-train")
                .join("artifacts")
                .join("linguae")
        });
    let _ = ARTIFACT_DIR.set(artifact_dir);

    let device = std::env::var("LINGUAE_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    let _ = DEVICE.set(device);

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, detect])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() {
    run();
}