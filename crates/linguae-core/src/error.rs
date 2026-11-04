use std::path::PathBuf;

/// Errors raised while resolving or loading a Linguae artifact.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not find the Linguae artifact. Pass --artifact PATH or set LINGUAE_ARTIFACT_DIR. Expected files: config.json, vocabulary.json, labels.json, model.pt")]
    ArtifactNotFound,

    #[error("artifact directory not found: {}", .0.display())]
    ArtifactDirectoryNotFound(PathBuf),

    #[error("{} is missing: {}", .0.display(), .1.join(", "))]
    MissingArtifactFiles(PathBuf, Vec<&'static str>),

    #[error("could not read {}: {source}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid JSON in {}: {source}", path.display())]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid artifact: {0}")]
    InvalidArtifact(String),

    #[error("unsupported device {0:?}; supported values are cpu, cuda and metal (mps is an alias for metal)")]
    UnsupportedDevice(String),

    #[error("device {0:?} is not available in this build; rebuild with --features cuda or --features metal")]
    DeviceUnavailable(String),

    #[error(transparent)]
    Model(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
