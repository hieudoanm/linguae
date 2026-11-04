//! Inference for the Linguae character-level language detector.
//!
//! The crate loads the exported artifact directory (`config.json`,
//! `vocabulary.json`, `labels.json`, `model.safetensors`) and predicts exactly
//! one language label per input string. Legacy `model.pt` artifacts are also
//! supported. It does not detect mixed-language spans.

pub mod artifact;
pub mod config;
pub mod error;
pub mod model;
pub mod text;

pub use artifact::{
    candidates, parse_device, resolve, Artifact, DEFAULT_ARTIFACT, REQUIRED_FILES, WEIGHT_FILES,
};
pub use config::Config;
pub use error::{Error, Result};
pub use model::CharCnn;

use candle_nn::ops::softmax;

/// Default chunk size used when classifying a batch of strings.
pub const DEFAULT_BATCH_SIZE: usize = 256;

/// One alternative in a top-k prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct Alternative {
    pub language: String,
    pub score: f32,
}

/// A single-label prediction for one input string.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    pub text: String,
    pub language: String,
    pub confidence: f32,
    /// Present for single-text predictions, absent for batch predictions.
    pub top_k: Option<Vec<Alternative>>,
}

impl Artifact {
    /// Predict one language label for a whole string and return top-k scores.
    pub fn predict_text(&self, text: &str, top_k: usize) -> Result<Prediction> {
        let encoded = text::encode_text(text, &self.vocabulary, self.config.max_length);
        let logits = self.forward(&[encoded])?;
        let probabilities = softmax(&logits, 1)?.to_vec2::<f32>()?;
        let scores = &probabilities[0];

        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.sort_by(|&left, &right| {
            scores[right]
                .total_cmp(&scores[left])
                .then_with(|| left.cmp(&right))
        });

        let k = top_k.clamp(1, scores.len());
        let alternatives: Vec<Alternative> = indices[..k]
            .iter()
            .map(|&index| Alternative {
                language: self.labels[index].clone(),
                score: scores[index],
            })
            .collect();

        let best = &alternatives[0];
        Ok(Prediction {
            text: text.to_string(),
            language: best.language.clone(),
            confidence: best.score,
            top_k: Some(alternatives),
        })
    }

    /// Predict one language label per string in a list.
    pub fn predict_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        batch_size: usize,
    ) -> Result<Vec<Prediction>> {
        let batch_size = batch_size.max(1);
        let mut predictions = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(batch_size) {
            let sequences: Vec<Vec<u32>> = chunk
                .iter()
                .map(|text| {
                    text::encode_text(text.as_ref(), &self.vocabulary, self.config.max_length)
                })
                .collect();
            let logits = self.forward(&sequences)?;
            let probabilities = softmax(&logits, 1)?.to_vec2::<f32>()?;

            for (text, scores) in chunk.iter().zip(probabilities) {
                let mut best_index = 0;
                for (index, score) in scores.iter().enumerate() {
                    if *score > scores[best_index] {
                        best_index = index;
                    }
                }
                predictions.push(Prediction {
                    text: text.as_ref().to_string(),
                    language: self.labels[best_index].clone(),
                    confidence: scores[best_index],
                    top_k: None,
                });
            }
        }

        Ok(predictions)
    }
}
