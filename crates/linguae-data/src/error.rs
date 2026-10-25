use std::path::PathBuf;

/// Errors raised while preparing the Linguae datasets.
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

    #[error("could not parse CSV in {}: {source}", path.display())]
    Csv {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },

    #[error("missing required file: {}", .0.display())]
    MissingFile(PathBuf),

    #[error("invalid dataset: {0}")]
    InvalidData(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error(
        "could not locate crates/linguae-data/data; run from the repository or set LINGUAE_DATA_DIR"
    )]
    DataRootNotFound,
}

pub type Result<T> = std::result::Result<T, Error>;
