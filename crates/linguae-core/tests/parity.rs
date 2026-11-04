//! Golden parity tests against predictions produced by the Python artifact.

use std::path::PathBuf;

use linguae_core::{Artifact, DEFAULT_BATCH_SIZE};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GoldenAlternative {
    language: String,
    score: f32,
}

#[derive(Debug, Deserialize)]
struct GoldenPrediction {
    text: String,
    language: String,
    confidence: f32,
    top_k: Vec<GoldenAlternative>,
}

fn artifact_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../linguae-train/artifacts/linguae")
        .canonicalize()
        .expect("the repository artifact must exist")
}

fn golden() -> Vec<GoldenPrediction> {
    serde_json::from_str(include_str!("fixtures/golden.json"))
        .expect("golden fixture must be valid JSON")
}

#[test]
fn single_text_matches_python() {
    let artifact = Artifact::load(artifact_directory(), "cpu").expect("load artifact");

    for case in golden() {
        let prediction = artifact.predict_text(&case.text, 5).expect("predict text");

        assert_eq!(
            prediction.language, case.language,
            "label mismatch for {:?}",
            case.text
        );
        assert!(
            (prediction.confidence - case.confidence).abs() < 1e-5,
            "confidence mismatch for {:?}: rust={} python={}",
            case.text,
            prediction.confidence,
            case.confidence
        );

        let alternatives = prediction.top_k.expect("single predictions include top-k");
        assert_eq!(alternatives.len(), case.top_k.len());
        for (actual, expected) in alternatives.iter().zip(case.top_k.iter()) {
            assert_eq!(actual.language, expected.language);
            assert!(
                (actual.score - expected.score).abs() < 1e-5,
                "top-k score mismatch for {:?}",
                case.text
            );
        }
    }
}

#[test]
fn batch_matches_single_text_predictions() {
    let artifact = Artifact::load(artifact_directory(), "cpu").expect("load artifact");
    let cases = golden();
    let texts: Vec<String> = cases.iter().map(|case| case.text.clone()).collect();

    let batch = artifact
        .predict_batch(&texts, DEFAULT_BATCH_SIZE)
        .expect("predict batch");

    assert_eq!(batch.len(), cases.len());
    for (prediction, case) in batch.iter().zip(cases.iter()) {
        assert_eq!(prediction.language, case.language, "for {:?}", case.text);
        assert!(
            (prediction.confidence - case.confidence).abs() < 1e-5,
            "confidence mismatch for {:?}",
            case.text
        );
        assert!(prediction.top_k.is_none());
    }
}

#[test]
fn batch_chunking_does_not_change_results() {
    let artifact = Artifact::load(artifact_directory(), "cpu").expect("load artifact");
    let texts: Vec<String> = golden().into_iter().map(|case| case.text).collect();

    let chunked = artifact.predict_batch(&texts, 3).expect("predict chunked");
    let whole = artifact
        .predict_batch(&texts, texts.len())
        .expect("predict whole");

    for (left, right) in chunked.iter().zip(whole.iter()) {
        assert_eq!(left.language, right.language);
        assert!((left.confidence - right.confidence).abs() < 1e-5);
    }
}
