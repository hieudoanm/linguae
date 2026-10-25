use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Format of a raw text/language file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    Csv,
    Tsv,
}

impl SourceFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Tsv => "tsv",
        }
    }
}

/// One raw text/language file that contributes rows to the training frame.
#[derive(Debug, Clone)]
pub struct SourceConfig {
    pub name: String,
    pub path: PathBuf,
    pub text_column: String,
    pub label_column: String,
    pub format: SourceFormat,
    pub columns: Option<Vec<String>>,
    pub role: String,
}

/// Single configuration object for data, model, training and export.
#[derive(Debug, Clone)]
pub struct Config {
    pub sources: Vec<SourceConfig>,
    pub min_examples_per_language: usize,
    pub max_rows_per_language_per_source: usize,
    pub max_words_per_language: usize,
    pub max_rows_per_language: usize,
    pub max_rows: Option<usize>,
    pub processed_dir: PathBuf,
    pub plots_dir: PathBuf,
    pub artifact_dir: PathBuf,
    pub train_fraction: f64,
    pub val_fraction: f64,
    pub test_fraction: f64,
    pub max_length: usize,
    pub embedding_dim: usize,
    pub num_filters: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    pub dropout: f32,
    pub batch_size: usize,
    pub length_bucketing: bool,
    pub dynamic_padding: bool,
    pub length_embedding: bool,
    pub mean_pooling: bool,
    pub length_bucket_boundaries: Vec<usize>,
    pub epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub grad_clip_norm: f64,
    pub early_stopping_patience: usize,
    pub max_misclassified_examples: usize,
    pub model_version: String,
    pub seed: u64,
    pub device: String,
    pub progress_every: usize,
    pub plots: bool,
}

impl Config {
    /// Build the configuration from environment variables and discovered paths.
    pub fn from_env() -> Result<Self> {
        let train_root = env_path("LINGUAE_TRAIN_ROOT")
            .or_else(|| env_path("LINGUAE_PACKAGE_ROOT"))
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
        let data_root = discover_data_root(env_path("LINGUAE_DATA_DIR").as_deref())?;
        let dataset_path = env_path("LINGUAE_DATASET_PATH");
        let sources = build_sources(&data_root, dataset_path.as_deref());

        let config = Self {
            sources,
            min_examples_per_language: env_usize("LINGUAE_MIN_EXAMPLES_PER_LANGUAGE", 100),
            max_rows_per_language_per_source: env_usize(
                "LINGUAE_MAX_ROWS_PER_LANGUAGE_PER_SOURCE",
                1000,
            ),
            max_words_per_language: env_usize("LINGUAE_MAX_WORDS_PER_LANGUAGE", 100_000),
            max_rows_per_language: env_usize("LINGUAE_MAX_ROWS_PER_LANGUAGE", 0),
            max_rows: env_usize_opt("LINGUAE_MAX_ROWS"),
            processed_dir: env_path("LINGUAE_PROCESSED_DIR")
                .unwrap_or_else(|| data_root.join("processed")),
            plots_dir: env_path("LINGUAE_PLOTS_DIR")
                .unwrap_or_else(|| train_root.join("artifacts").join("images")),
            artifact_dir: env_path("LINGUAE_ARTIFACT_DIR")
                .unwrap_or_else(|| train_root.join("artifacts").join("linguae")),
            train_fraction: env_f64("LINGUAE_TRAIN_FRACTION", 0.8),
            val_fraction: env_f64("LINGUAE_VAL_FRACTION", 0.1),
            test_fraction: env_f64("LINGUAE_TEST_FRACTION", 0.1),
            max_length: env_usize(
                "LINGUAE_MAX_SEQUENCE_LENGTH",
                env_usize("LINGUAE_MAX_LENGTH", 256),
            ),
            embedding_dim: env_usize("LINGUAE_EMBEDDING_DIM", 32),
            num_filters: env_usize_list("LINGUAE_NUM_FILTERS", &[128, 128, 128]),
            kernel_sizes: env_usize_list("LINGUAE_KERNEL_SIZES", &[3, 5, 7]),
            dropout: env_f64("LINGUAE_DROPOUT", 0.3) as f32,
            batch_size: env_usize("LINGUAE_BATCH_SIZE", 256),
            length_bucketing: env_bool("LINGUAE_LENGTH_BUCKETING", true),
            dynamic_padding: env_bool("LINGUAE_DYNAMIC_PADDING", true),
            length_embedding: env_bool("LINGUAE_LENGTH_EMBEDDING", true),
            mean_pooling: env_bool("LINGUAE_MEAN_POOLING", true),
            length_bucket_boundaries: env_usize_list(
                "LINGUAE_LENGTH_BUCKET_BOUNDARIES",
                &[16, 32, 64, 128, 256],
            ),
            epochs: env_usize("LINGUAE_EPOCHS", 10),
            learning_rate: env_f64("LINGUAE_LEARNING_RATE", 1e-3),
            weight_decay: env_f64("LINGUAE_WEIGHT_DECAY", 1e-4),
            grad_clip_norm: env_f64("LINGUAE_GRAD_CLIP_NORM", 5.0),
            early_stopping_patience: env_usize("LINGUAE_EARLY_STOPPING_PATIENCE", 2),
            max_misclassified_examples: env_usize("LINGUAE_MAX_MISCLASSIFIED_EXAMPLES", 2000),
            model_version: env_string("LINGUAE_MODEL_VERSION", "1.0.0"),
            seed: env_u64("LINGUAE_SEED", 42),
            device: env_string("LINGUAE_DEVICE", "auto"),
            progress_every: env_usize("LINGUAE_PROGRESS_EVERY", 200).max(1),
            plots: !env_bool("LINGUAE_NO_PLOTS", false),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.sources.is_empty() {
            return Err(Error::Config("no dataset sources configured".into()));
        }
        if self.max_length == 0 {
            return Err(Error::Config("max_length must be greater than zero".into()));
        }
        if self.batch_size == 0 {
            return Err(Error::Config("batch_size must be greater than zero".into()));
        }
        if self.length_bucket_boundaries.is_empty()
            || self
                .length_bucket_boundaries
                .windows(2)
                .any(|w| w[0] >= w[1])
        {
            return Err(Error::Config(
                "length bucket boundaries must be non-empty and strictly increasing".into(),
            ));
        }
        if self.epochs == 0 {
            return Err(Error::Config("epochs must be greater than zero".into()));
        }
        if self.embedding_dim == 0 {
            return Err(Error::Config(
                "embedding_dim must be greater than zero".into(),
            ));
        }
        if self.num_filters.len() != self.kernel_sizes.len() {
            return Err(Error::Config(
                "num_filters and kernel_sizes must have the same length".into(),
            ));
        }
        if self.num_filters.is_empty() {
            return Err(Error::Config("num_filters must not be empty".into()));
        }
        if self.kernel_sizes.iter().any(|size| size % 2 == 0) {
            return Err(Error::Config(
                "kernel_sizes must be odd so padding preserves the length".into(),
            ));
        }
        let fractions = [self.train_fraction, self.val_fraction, self.test_fraction];
        if fractions.iter().any(|value| *value <= 0.0)
            || (fractions.iter().sum::<f64>() - 1.0).abs() > 1e-6
        {
            return Err(Error::Config(format!(
                "train/validation/test fractions must be positive and sum to 1.0, got {fractions:?}"
            )));
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(Error::Config("dropout must be in [0, 1)".into()));
        }
        Ok(())
    }

    /// Replace the source list when an explicit data root or dataset path is used.
    pub fn set_sources(
        &mut self,
        data_root: Option<&Path>,
        dataset_path: Option<&Path>,
    ) -> Result<()> {
        if let Some(path) = dataset_path {
            let path = expand(path);
            if !path.is_file() {
                return Err(Error::NotFound(format!("dataset file {}", path.display())));
            }
            self.sources = vec![SourceConfig {
                name: "custom".into(),
                path,
                text_column: "text".into(),
                label_column: "language".into(),
                format: SourceFormat::Csv,
                columns: None,
                role: "mixed".into(),
            }];
            return Ok(());
        }
        if let Some(root) = data_root {
            let root = discover_data_root(Some(root))?;
            self.sources = build_sources(&root, None);
        }
        Ok(())
    }
}

/// Locate `crates/linguae-data/data` by walking up from the current directory,
/// the executable and the training crate root.
pub fn discover_data_root(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        let path = expand(path);
        if is_data_root(&path) {
            return Ok(path);
        }
        return Err(Error::Config(format!(
            "LINGUAE_DATA_DIR={} is not a Linguae data directory",
            path.display()
        )));
    }

    let mut starts = vec![std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))];
    starts.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            starts.push(parent.to_path_buf());
        }
    }

    for start in starts {
        for ancestor in start.ancestors() {
            for option in [
                ancestor.join("crates").join("linguae-data").join("data"),
                ancestor.join("data"),
                ancestor.to_path_buf(),
            ] {
                if is_data_root(&option) {
                    return Ok(option);
                }
            }
        }
    }

    Err(Error::NotFound(
        "crates/linguae-data/data; set LINGUAE_DATA_DIR to override".into(),
    ))
}

fn is_data_root(path: &Path) -> bool {
    path.is_dir() && (path.join("csv").is_dir() || path.join("txt").is_dir())
}

fn build_sources(data_root: &Path, dataset_path: Option<&Path>) -> Vec<SourceConfig> {
    if let Some(path) = dataset_path {
        return vec![SourceConfig {
            name: "custom".into(),
            path: expand(path),
            text_column: "text".into(),
            label_column: "language".into(),
            format: SourceFormat::Csv,
            columns: None,
            role: "mixed".into(),
        }];
    }

    let mut sources = Vec::new();
    if !env_bool("LINGUAE_DISABLE_WILI", false) {
        sources.push(SourceConfig {
            name: "wili-2018".into(),
            path: env_path("LINGUAE_WILI_CSV")
                .unwrap_or_else(|| data_root.join("csv").join("wili-2018.csv")),
            text_column: "text".into(),
            label_column: "language".into(),
            format: SourceFormat::Csv,
            columns: None,
            role: "long".into(),
        });
    }
    if !env_bool("LINGUAE_DISABLE_TATOEBA", false) {
        sources.push(SourceConfig {
            name: "tatoeba".into(),
            path: env_path("LINGUAE_TATOEBA_TSV")
                .unwrap_or_else(|| data_root.join("csv").join("tatoeba.tsv")),
            text_column: "text".into(),
            label_column: "language".into(),
            format: SourceFormat::Tsv,
            columns: Some(vec!["id".into(), "language".into(), "text".into()]),
            role: "short".into(),
        });
    }
    if !env_bool("LINGUAE_DISABLE_WILI_WORDS", false) {
        let path = env_path("LINGUAE_WILI_WORDS_CSV")
            .unwrap_or_else(|| data_root.join("csv").join("wili-2018-words.csv"));
        if path.is_file() {
            sources.push(SourceConfig {
                name: "wili-words".into(),
                path,
                text_column: "text".into(),
                label_column: "language".into(),
                format: SourceFormat::Csv,
                columns: None,
                role: "words".into(),
            });
        }
    }
    sources
}

fn expand(path: &Path) -> PathBuf {
    let Some(raw) = path.to_str() else {
        return path.to_path_buf();
    };
    if raw == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    path.to_path_buf()
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn env_usize_opt(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn env_usize_list(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| {
            let parsed: Vec<usize> = value
                .split(',')
                .filter_map(|item| item.trim().parse().ok())
                .collect();
            (!parsed.is_empty()).then_some(parsed)
        })
        .unwrap_or_else(|| default.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_data_root() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("csv")).unwrap();
        directory
    }

    #[test]
    fn rejects_even_kernels() {
        let root = temp_data_root();
        std::env::set_var("LINGUAE_DATA_DIR", root.path());
        let mut config = Config::from_env().unwrap();
        std::env::remove_var("LINGUAE_DATA_DIR");
        config.kernel_sizes = vec![2, 4];
        assert!(config.validate().is_err());
    }

    #[test]
    fn discovers_an_explicit_data_root() {
        let root = temp_data_root();
        let discovered = discover_data_root(Some(root.path())).unwrap();
        assert_eq!(discovered, root.path());
    }
}
