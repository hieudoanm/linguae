use serde::Deserialize;

/// The subset of `config.json` that inference needs.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub embedding_dim: usize,
    pub num_filters: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    #[serde(default)]
    pub dropout: f32,
    pub vocab_size: usize,
    pub num_classes: usize,
    pub max_length: usize,
    #[serde(default)]
    pub padding_idx: u32,
    #[serde(default = "default_unknown_idx")]
    pub unknown_idx: u32,
    #[serde(default)]
    pub lowercase: bool,
}

fn default_unknown_idx() -> u32 {
    1
}

impl Config {
    pub fn validate(&self) -> crate::Result<()> {
        use crate::Error;

        if self.num_filters.len() != self.kernel_sizes.len() {
            return Err(Error::InvalidArtifact(
                "num_filters and kernel_sizes must have the same length".into(),
            ));
        }
        if self.num_filters.is_empty() {
            return Err(Error::InvalidArtifact("num_filters is empty".into()));
        }
        if self.kernel_sizes.iter().any(|size| size % 2 == 0) {
            return Err(Error::InvalidArtifact(
                "kernel_sizes must be odd so that padding preserves the length".into(),
            ));
        }
        if self.vocab_size == 0 || self.num_classes == 0 {
            return Err(Error::InvalidArtifact(
                "vocab_size and num_classes must be greater than zero".into(),
            ));
        }
        if self.max_length == 0 {
            return Err(Error::InvalidArtifact(
                "max_length must be greater than zero".into(),
            ));
        }
        if self.lowercase {
            return Err(Error::InvalidArtifact(
                "lowercase artifacts are not supported by this inference build".into(),
            ));
        }
        Ok(())
    }
}
