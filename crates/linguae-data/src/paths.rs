use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Locate the `crates/linguae-data/data` directory by walking up from the
/// current directory and the executable.
pub fn find_data_root() -> Result<PathBuf> {
    if let Some(raw) = std::env::var_os("LINGUAE_DATA_DIR") {
        if !raw.is_empty() {
            let root = expand_tilde(Path::new(&raw));
            if is_data_root(&root) {
                return Ok(root);
            }
            return Err(Error::InvalidArgument(format!(
                "LINGUAE_DATA_DIR={raw:?} is not a Linguae data directory"
            )));
        }
    }

    let mut origins = Vec::new();
    if let Ok(current_dir) = std::env::current_dir() {
        origins.push(current_dir);
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            origins.push(parent.to_path_buf());
        }
    }

    for origin in origins {
        for candidate in origin.ancestors() {
            for option in [
                candidate.join("crates").join("linguae-data").join("data"),
                candidate.to_path_buf(),
            ] {
                if is_data_root(&option) {
                    return Ok(option);
                }
            }
        }
    }

    Err(Error::DataRootNotFound)
}

fn is_data_root(path: &Path) -> bool {
    path.is_dir() && (path.join("csv").is_dir() || path.join("txt").is_dir())
}

/// Resolved input and output locations for every dataset conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub data_root: PathBuf,
    pub wili_dir: PathBuf,
    pub wili_csv: PathBuf,
    pub tatoeba_tsv: PathBuf,
    pub tatoeba_csv: PathBuf,
    pub words_csv: PathBuf,
}

impl Layout {
    /// Resolve the default layout, honouring `LINGUAE_*` environment overrides.
    pub fn discover(data_root: Option<&Path>) -> Result<Self> {
        let data_root = match data_root {
            Some(path) => {
                let path = expand_tilde(path);
                if !is_data_root(&path) {
                    return Err(Error::InvalidArgument(format!(
                        "{} is not a Linguae data directory",
                        path.display()
                    )));
                }
                path
            }
            None => find_data_root()?,
        };

        Ok(Self {
            data_root: data_root.clone(),
            wili_dir: env_path("LINGUAE_WILI_DIR", data_root.join("txt").join("wili-2018")),
            wili_csv: env_path(
                "LINGUAE_WILI_CSV",
                data_root.join("csv").join("wili-2018.csv"),
            ),
            tatoeba_tsv: env_path(
                "LINGUAE_TATOEBA_TSV",
                data_root.join("csv").join("tatoeba.tsv"),
            ),
            tatoeba_csv: env_path(
                "LINGUAE_TATOEBA_CSV",
                data_root.join("csv").join("tatoeba.csv"),
            ),
            words_csv: env_path(
                "LINGUAE_WILI_WORDS_CSV",
                data_root.join("csv").join("wili-2018-words.csv"),
            ),
        })
    }
}

fn env_path(name: &str, default: PathBuf) -> PathBuf {
    match std::env::var_os(name) {
        Some(value) if !value.is_empty() => expand_tilde(Path::new(&value)),
        _ => default,
    }
}

/// Expand a leading `~` or `~/` using `$HOME`.
pub fn expand_tilde(path: &Path) -> PathBuf {
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
