use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::io::{read_lines, write_csv_atomic, FileSummary};
use crate::stats::{LengthHistogram, LengthStats};

/// Inputs and output for the WiLI-2018 conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WiliOptions {
    pub input_dir: PathBuf,
    pub output: PathBuf,
}

/// Summary of a completed WiLI-2018 conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WiliReport {
    pub train_rows: u64,
    pub test_rows: u64,
    pub rows: u64,
    pub languages: usize,
    pub duplicate_text_rows: u64,
    pub texts_with_conflicting_labels: u64,
    pub lengths: LengthStats,
    pub file: FileSummary,
}

/// Convert `x_train.txt`/`y_train.txt`/`x_test.txt`/`y_test.txt` into a single
/// `text,language` CSV. Texts and labels are kept in their original order.
pub fn convert(options: &WiliOptions) -> Result<WiliReport> {
    let train = read_split(&options.input_dir, "train")?;
    let test = read_split(&options.input_dir, "test")?;

    let train_rows = train.len() as u64;
    let test_rows = test.len() as u64;
    log::info!("train split: {train_rows} texts and labels (matched)");
    log::info!("test split: {test_rows} texts and labels (matched)");

    let mut rows: Vec<(String, String)> = Vec::with_capacity(train.len() + test.len());
    rows.extend(train);
    rows.extend(test);

    let mut lengths = LengthHistogram::default();
    let mut languages = HashSet::new();
    let mut seen_texts = HashSet::new();
    let mut label_of_text: HashMap<&str, &str> = HashMap::new();
    let mut conflicting_texts = HashSet::new();
    let mut duplicate_text_rows = 0u64;

    for (text, language) in &rows {
        lengths.push(text.chars().count());
        languages.insert(language.as_str());
        if !seen_texts.insert(text.as_str()) {
            duplicate_text_rows += 1;
        }
        match label_of_text.get(text.as_str()) {
            Some(existing) if *existing != language.as_str() => {
                conflicting_texts.insert(text.as_str());
            }
            Some(_) => {}
            None => {
                label_of_text.insert(text.as_str(), language.as_str());
            }
        }
    }

    let rows_count = rows.len() as u64;
    let languages_count = languages.len();
    log::info!("combined train={train_rows} + test={test_rows} -> {rows_count} rows");
    log::info!("unique languages: {languages_count}");
    log::info!(
        "duplicate text rows: {duplicate_text_rows}; texts with conflicting labels: {}",
        conflicting_texts.len()
    );

    let file = write_csv_atomic(&options.output, &["text", "language"], |writer| {
        for (text, language) in &rows {
            writer
                .write_record([text.as_str(), language.as_str()])
                .map_err(|source| Error::Csv {
                    path: options.output.clone(),
                    source,
                })?;
        }
        Ok(())
    })?;

    let lengths = lengths
        .finish()
        .ok_or_else(|| Error::InvalidData("the WiLI dataset is empty".into()))?;
    log::info!(
        "character length: min={} median={} max={}",
        lengths.min,
        lengths.median,
        lengths.max
    );
    log::info!(
        "wrote {} ({rows_count} rows, {:.1} MiB, sha256 {})",
        options.output.display(),
        file.bytes as f64 / (1024.0 * 1024.0),
        file.sha256
    );

    Ok(WiliReport {
        train_rows,
        test_rows,
        rows: rows_count,
        languages: languages_count,
        duplicate_text_rows,
        texts_with_conflicting_labels: conflicting_texts.len() as u64,
        lengths,
        file,
    })
}

fn read_split(input_dir: &Path, split: &str) -> Result<Vec<(String, String)>> {
    let text_path = input_dir.join(format!("x_{split}.txt"));
    let label_path = input_dir.join(format!("y_{split}.txt"));
    let texts = read_lines(&text_path)?;
    let labels = read_lines(&label_path)?;

    if texts.len() != labels.len() {
        return Err(Error::InvalidData(format!(
            "{split} split has {} texts but {} labels",
            texts.len(),
            labels.len()
        )));
    }
    if texts.is_empty() {
        return Err(Error::InvalidData(format!("{split} split is empty")));
    }

    Ok(texts.into_iter().zip(labels).collect())
}
