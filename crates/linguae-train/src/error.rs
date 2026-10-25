use std::path::PathBuf;

/// Errors raised by the training pipeline.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read {}: {source}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not write {}: {source}", path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid CSV {}: {source}", path.display())]
    Csv {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },

    #[error("invalid JSON {}: {source}", path.display())]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid configuration: {0}")]
    Config(String),

    #[error("invalid dataset: {0}")]
    Dataset(String),

    #[error("training failed: {0}")]
    Training(String),

    #[error("could not find {0}")]
    NotFound(String),

    #[error("plotting failed: {0}")]
    Plot(String),

    #[error(transparent)]
    Core(#[from] linguae_core::Error),

    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
