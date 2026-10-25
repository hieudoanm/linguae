use std::path::Path;

use log::info;

use crate::error::Result;

/// Load an artifact with the inference crate and print sample predictions.
pub fn verify_artifact(directory: &Path, device: &str, texts: &[String]) -> Result<()> {
    let artifact = linguae_core::Artifact::load(directory, device)?;
    info!(
        "loaded artifact {} ({} labels, vocabulary {}, device {device})",
        directory.display(),
        artifact.labels.len(),
        artifact.vocabulary.len()
    );

    let defaults = [
        "Bonjour tout le monde, comment allez-vous ?".to_string(),
        "Hello world".to_string(),
        "Привет, мир".to_string(),
        "こんにちは世界".to_string(),
    ];
    let samples = if texts.is_empty() {
        defaults.as_slice()
    } else {
        texts
    };

    for text in samples {
        let prediction = artifact.predict_text(text, 3)?;
        let top: Vec<String> = prediction
            .top_k
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|item| format!("{}={:.3}", item.language, item.score))
            .collect();
        println!(
            "{} ({:.3}) <- {:?} | {}",
            prediction.language,
            prediction.confidence,
            text,
            top.join(", ")
        );
    }
    Ok(())
}
