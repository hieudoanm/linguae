use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::model::CharCnn;

/// Default artifact location, relative to a repository root.
pub const DEFAULT_ARTIFACT: &str = "packages/models/artifacts/linguae";

/// Files the Rust CLI needs. `linguae_inference.py` is intentionally not
/// required: it exists for Python consumers, not for this binary.
pub const REQUIRED_FILES: [&str; 4] = ["config.json", "vocabulary.json", "labels.json", "model.pt"];

#[derive(Debug, Deserialize)]
struct LabelsFile {
    idx2label: HashMap<String, String>,
}

/// A loaded, ready-to-use language detector.
pub struct Artifact {
    pub directory: PathBuf,
    pub config: Config,
    pub vocabulary: HashMap<char, u32>,
    pub labels: Vec<String>,
    pub device: Device,
    model: CharCnn,
}

impl Artifact {
    /// Load config, vocabulary, labels and weights from an artifact directory.
    pub fn load(directory: impl AsRef<Path>, device: &str) -> Result<Self> {
        let directory = directory.as_ref();
        let config: Config = read_json(&directory.join("config.json"))?;
        config.validate()?;

        let raw_vocabulary: HashMap<String, u32> = read_json(&directory.join("vocabulary.json"))?;
        let mut vocabulary = HashMap::with_capacity(raw_vocabulary.len());
        for (character, index) in raw_vocabulary {
            let mut chars = character.chars();
            // Multi-character keys are the `<pad>` and `<unk>` special tokens.
            // `encode_text` looks up individual characters, so they never match.
            if let (Some(character), None) = (chars.next(), chars.next()) {
                vocabulary.insert(character, index);
            }
        }

        let labels_file: LabelsFile = read_json(&directory.join("labels.json"))?;
        let mut labels = vec![String::new(); config.num_classes];
        for (index, label) in labels_file.idx2label {
            let index: usize = index.parse().map_err(|_| {
                Error::InvalidArtifact(format!("labels.json has a non-numeric index {index:?}"))
            })?;
            let slot = labels.get_mut(index).ok_or_else(|| {
                Error::InvalidArtifact(format!(
                    "labels.json index {index} is outside 0..{}",
                    config.num_classes
                ))
            })?;
            *slot = label;
        }
        if labels.iter().any(String::is_empty) {
            return Err(Error::InvalidArtifact(
                "labels.json is missing one or more label indices".into(),
            ));
        }

        let device = parse_device(device)?;
        let vb = VarBuilder::from_pth(directory.join("model.pt"), DType::F32, &device)?;
        let model = CharCnn::load(&config, vb)?;

        Ok(Self {
            directory: directory.to_path_buf(),
            config,
            vocabulary,
            labels,
            device,
            model,
        })
    }

    pub(crate) fn forward(&self, sequences: &[Vec<u32>]) -> Result<candle_core::Tensor> {
        debug_assert!(!sequences.is_empty());

        let batch = sequences.len();
        let length = sequences.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let mut ids = Vec::with_capacity(batch * length);
        let mut mask: Vec<f32> = Vec::with_capacity(batch * length);

        for sequence in sequences {
            for position in 0..length {
                let id = sequence
                    .get(position)
                    .copied()
                    .unwrap_or(crate::text::PAD_ID);
                ids.push(id);
                mask.push(if id == crate::text::PAD_ID { 0.0 } else { 1.0 });
            }
        }

        let input_ids = candle_core::Tensor::from_vec(ids, (batch, length), &self.device)?;
        let mask = candle_core::Tensor::from_vec(mask, (batch, 1, length), &self.device)?;
        Ok(self.model.forward(&input_ids, &mask)?)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let content = std::fs::read_to_string(path).map_err(|source| Error::Read {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&content).map_err(|source| Error::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn missing_files(directory: &Path) -> Vec<&'static str> {
    REQUIRED_FILES
        .iter()
        .copied()
        .filter(|name| !directory.join(name).is_file())
        .collect()
}

fn validate_directory(directory: &Path) -> Result<()> {
    if !directory.is_dir() {
        return Err(Error::ArtifactDirectoryNotFound(directory.to_path_buf()));
    }
    let missing = missing_files(directory);
    if !missing.is_empty() {
        return Err(Error::MissingArtifactFiles(
            directory.to_path_buf(),
            missing,
        ));
    }
    Ok(())
}

/// Build a deduplicated list of likely artifact locations.
pub fn candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    let mut add = |candidate: PathBuf| {
        if seen.insert(candidate.clone()) {
            candidates.push(candidate);
        }
    };

    if let Ok(current_dir) = std::env::current_dir() {
        for root in current_dir.ancestors() {
            add(root.join(DEFAULT_ARTIFACT));
            add(root.join("artifacts").join("linguae"));
        }
    }

    if let Ok(executable) = std::env::current_exe() {
        if let Some(directory) = executable.parent() {
            for root in directory.ancestors().take(4) {
                add(root.join(DEFAULT_ARTIFACT));
                add(root.join("artifacts").join("linguae"));
            }
        }
    }

    candidates
}

/// Resolve and validate the artifact directory.
pub fn resolve(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(directory) = explicit {
        let directory = expand(directory);
        validate_directory(&directory)?;
        return Ok(directory);
    }

    if let Some(environment) = std::env::var_os("LINGUAE_ARTIFACT_DIR") {
        if !environment.is_empty() {
            let directory = expand(Path::new(&environment));
            validate_directory(&directory)?;
            return Ok(directory);
        }
    }

    for candidate in candidates() {
        if candidate.is_dir() && missing_files(&candidate).is_empty() {
            return Ok(candidate);
        }
    }

    Err(Error::ArtifactNotFound)
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

/// Parse a device name, mirroring the Python CLI's `--device` option.
pub fn parse_device(name: &str) -> Result<Device> {
    let base = name.split(':').next().unwrap_or(name).to_ascii_lowercase();
    match base.as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => cuda_device(name),
        "metal" | "mps" => metal_device(name),
        _ => Err(Error::UnsupportedDevice(name.to_string())),
    }
}

#[cfg(feature = "cuda")]
fn cuda_device(_name: &str) -> Result<Device> {
    Device::new_cuda(0).map_err(Error::from)
}

#[cfg(not(feature = "cuda"))]
fn cuda_device(name: &str) -> Result<Device> {
    Err(Error::DeviceUnavailable(name.to_string()))
}

#[cfg(feature = "metal")]
fn metal_device(_name: &str) -> Result<Device> {
    Device::new_metal(0).map_err(Error::from)
}

#[cfg(not(feature = "metal"))]
fn metal_device(name: &str) -> Result<Device> {
    Err(Error::DeviceUnavailable(name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cpu() {
        assert!(matches!(parse_device("cpu"), Ok(Device::Cpu)));
    }

    #[test]
    fn rejects_unknown_devices() {
        assert!(matches!(
            parse_device("tpu"),
            Err(Error::UnsupportedDevice(_))
        ));
    }

    #[test]
    fn missing_files_lists_every_required_file_for_an_empty_directory() {
        let directory = std::env::temp_dir().join("linguae-empty-artifact");
        let _ = std::fs::create_dir_all(&directory);
        assert_eq!(missing_files(&directory).len(), REQUIRED_FILES.len());
    }
}
