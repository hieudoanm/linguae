//! Shared fixtures for the `linguae-data` integration tests.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

/// A temporary `crates/linguae-data/data` tree with the directories the layout
/// probe requires.
pub struct Fixture {
    root: PathBuf,
    _temporary: TempDir,
}

impl Fixture {
    pub fn new() -> Self {
        let temporary = TempDir::new().expect("temporary directory is created");
        let root = temporary.path().join("data");
        fs::create_dir_all(root.join("txt").join("wili-2018")).expect("wili directory is created");
        fs::create_dir_all(root.join("csv")).expect("csv directory is created");
        Self {
            root,
            _temporary: temporary,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn wili_dir(&self) -> PathBuf {
        self.root.join("txt").join("wili-2018")
    }

    pub fn csv_dir(&self) -> PathBuf {
        self.root.join("csv")
    }

    /// Write a WiLI split: matching `x_{split}.txt` and `y_{split}.txt`.
    pub fn write_wili_split(&self, split: &str, rows: &[(&str, &str)]) {
        let texts = rows
            .iter()
            .map(|(text, _)| *text)
            .collect::<Vec<_>>()
            .join("\n");
        let labels = rows
            .iter()
            .map(|(_, label)| *label)
            .collect::<Vec<_>>()
            .join("\n");
        let texts = if texts.is_empty() {
            String::new()
        } else {
            texts + "\n"
        };
        let labels = if labels.is_empty() {
            String::new()
        } else {
            labels + "\n"
        };
        fs::write(self.wili_dir().join(format!("x_{split}.txt")), texts)
            .expect("text split is written");
        fs::write(self.wili_dir().join(format!("y_{split}.txt")), labels)
            .expect("label split is written");
    }

    pub fn write_file(&self, relative: &str, contents: &str) -> PathBuf {
        let path = self.root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("parent directory is created");
        }
        fs::write(&path, contents).expect("fixture file is written");
        path
    }
}
