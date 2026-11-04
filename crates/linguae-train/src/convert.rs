use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use log::info;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

const SKIP_FILES: [&str; 4] = [
    "model.pt",
    "linguae_inference.py",
    "example_inference.py",
    "__pycache__",
];

/// Convert a legacy PyTorch artifact into the safetensors artifact layout.
pub fn convert_artifact(source: &Path, destination: &Path, overwrite: bool) -> Result<()> {
    if !source.is_dir() {
        return Err(Error::NotFound(format!(
            "artifact directory {}",
            source.display()
        )));
    }
    let legacy_model = source.join("model.pt");
    if !legacy_model.is_file() {
        return Err(Error::NotFound(format!(
            "legacy weights {}",
            legacy_model.display()
        )));
    }
    if destination.exists()
        && fs::read_dir(destination)
            .map(|mut items| items.next().is_some())
            .unwrap_or(false)
        && !overwrite
    {
        return Err(Error::Config(format!(
            "{} already contains files; pass --overwrite to replace it",
            destination.display()
        )));
    }
    fs::create_dir_all(destination).map_err(|source| Error::Write {
        path: destination.to_path_buf(),
        source,
    })?;

    for entry in fs::read_dir(source).map_err(|source_error| Error::Read {
        path: source.to_path_buf(),
        source: source_error,
    })? {
        let entry = entry.map_err(|source_error| Error::Read {
            path: source.to_path_buf(),
            source: source_error,
        })?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if SKIP_FILES.contains(&name.as_ref()) {
            continue;
        }
        let target = destination.join(name.as_ref());
        if entry.path().is_file() {
            fs::copy(entry.path(), &target).map_err(|source_error| Error::Write {
                path: target.clone(),
                source: source_error,
            })?;
        }
    }

    let tensors = candle_core::pickle::read_all(&legacy_model)?;
    let map: HashMap<String, candle_core::Tensor> = tensors.into_iter().collect();
    candle_core::safetensors::save(&map, destination.join("model.safetensors"))?;
    info!(
        "converted {} tensors to {}",
        map.len(),
        destination.join("model.safetensors").display()
    );

    rewrite_config(destination)?;
    rewrite_manifest(destination)?;
    Ok(())
}

fn rewrite_config(destination: &Path) -> Result<()> {
    let path = destination.join("config.json");
    let mut config: Value = read_json(&path)?;
    if let Some(object) = config.as_object_mut() {
        object.insert("framework".into(), json!("candle"));
        object.insert(
            "artifact_schema".into(),
            json!("linguae.language-detector/2"),
        );
        object.insert("weights".into(), json!("model.safetensors"));
        object.remove("torch_version");
    }
    write_json(&path, &config)
}

fn rewrite_manifest(destination: &Path) -> Result<()> {
    let path = destination.join("manifest.json");
    let mut manifest: Value = read_json(&path)?;
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(destination).map_err(|source_error| Error::Read {
        path: destination.to_path_buf(),
        source: source_error,
    })? {
        let entry = entry.map_err(|source_error| Error::Read {
            path: destination.to_path_buf(),
            source: source_error,
        })?;
        if !entry.path().is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name == "manifest.json" {
            continue;
        }
        files.insert(name, sha256_file(&entry.path())?);
    }
    if let Some(object) = manifest.as_object_mut() {
        object.insert("schema_version".into(), json!(2));
        object.insert(
            "artifact_schema".into(),
            json!("linguae.language-detector/2"),
        );
        object.insert("files".into(), json!(files));
        object.insert(
            "environment".into(),
            json!({
                "linguae_train": env!("CARGO_PKG_VERSION"),
                "candle": "0.11.1",
                "platform": std::env::consts::OS,
            }),
        );
        if let Some(datasets) = object.get_mut("datasets").and_then(Value::as_array_mut) {
            for dataset in datasets {
                if let Some(dataset) = dataset.as_object_mut() {
                    dataset.remove("sha256");
                    if let Some(path) = dataset.get("path").and_then(Value::as_str) {
                        let path = PathBuf::from(path);
                        if path.is_file() {
                            dataset.insert("sha256".into(), json!(sha256_file(&path)?));
                        }
                    }
                }
            }
        }
        object.remove("source_artifact");
    }
    write_json(&path, &manifest)
}

fn read_json(path: &Path) -> Result<Value> {
    let content = fs::read_to_string(path).map_err(|source| Error::Read {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&content).map_err(|source| Error::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut content = serde_json::to_string_pretty(value).map_err(|source| Error::Json {
        path: path.to_path_buf(),
        source,
    })?;
    content.push('\n');
    fs::write(path, content).map_err(|source| Error::Write {
        path: path.to_path_buf(),
        source,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|source| Error::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}
