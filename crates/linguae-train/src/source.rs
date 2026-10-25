use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use log::{info, warn};

use crate::config::{SourceConfig, SourceFormat};
use crate::dataset::Split;
use crate::error::{Error, Result};

/// Tatoeba uses individual ISO 639-3 codes while WiLI-2018 uses a few older or
/// macro-language codes. Merge the known equivalents so short Tatoeba sentences
/// and long WiLI paragraphs share one output class.
pub fn alias_label(label: &str) -> &str {
    match label {
        "cmn" => "zho",
        "yue" => "zh-yue",
        "pes" => "fas",
        "lvs" => "lav",
        "npi" => "nep",
        "zsm" => "msa",
        "zlm" => "msa",
        "kmr" => "kur",
        "swh" => "swa",
        "swc" => "swa",
        "gsw" => "als",
        "bua" => "bxr",
        "gom" => "kok",
        "kpv" => "kom",
        _ => label,
    }
}

#[derive(Debug, Clone)]
pub struct Record {
    pub text: String,
    pub language: String,
    pub source: String,
    pub role: String,
    pub split: Split,
}

#[derive(Debug, Clone)]
pub struct InvalidRecord {
    pub text: String,
    pub language: String,
    pub source: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ValidationReport {
    pub input_rows: usize,
    pub valid_rows: usize,
    pub invalid_rows: usize,
    pub rows_by_source: BTreeMap<String, usize>,
    pub empty_text_rows: usize,
    pub empty_label_rows: usize,
    pub empty_after_normalisation_rows: usize,
    pub rows_changed_by_normalisation: usize,
    pub unique_languages: usize,
    pub duplicate_text_rows: usize,
    pub duplicate_text_label_rows: usize,
    pub texts_with_conflicting_labels: usize,
    pub min_examples_per_language: usize,
    pub max_examples_per_language: usize,
}

#[derive(Debug)]
struct RawRecord {
    text: String,
    language: String,
    source: String,
    role: String,
}

/// Read every configured source and return cleaned records plus invalid rows.
pub fn load_sources(
    sources: &[SourceConfig],
) -> Result<(Vec<Record>, Vec<InvalidRecord>, ValidationReport)> {
    let mut raw = Vec::new();
    let mut loaded_sources = 0usize;

    for source in sources {
        if !source.path.is_file() {
            warn!(
                "dataset source {:?} not found at {}; skipping it",
                source.name,
                source.path.display()
            );
            continue;
        }
        let rows = read_source(source)?;
        info!(
            "loaded {:<10} {:>9} rows from {}",
            source.name,
            rows.len(),
            source.path.display()
        );
        raw.extend(rows);
        loaded_sources += 1;
    }

    if loaded_sources == 0 {
        return Err(Error::Dataset(
            "no dataset sources found; run linguae-data first or set LINGUAE_DATA_DIR".into(),
        ));
    }

    validate_and_normalise(raw)
}

fn read_source(source: &SourceConfig) -> Result<Vec<RawRecord>> {
    match source.format {
        SourceFormat::Csv => read_csv(source),
        SourceFormat::Tsv => read_tsv(source),
    }
}

fn read_csv(source: &SourceConfig) -> Result<Vec<RawRecord>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&source.path)
        .map_err(|error| csv_error(&source.path, error))?;

    let headers = reader
        .headers()
        .map_err(|error| csv_error(&source.path, error))?
        .clone();
    let text_index = column_index(&headers, &source.text_column, &source.path)?;
    let label_index = column_index(&headers, &source.label_column, &source.path)?;

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|error| csv_error(&source.path, error))?;
        let text = record.get(text_index).unwrap_or("").to_string();
        let language = record.get(label_index).unwrap_or("").to_string();
        records.push(RawRecord {
            text,
            language,
            source: source.name.clone(),
            role: source.role.clone(),
        });
    }
    Ok(records)
}

fn read_tsv(source: &SourceConfig) -> Result<Vec<RawRecord>> {
    let columns = source.columns.as_ref().ok_or_else(|| {
        Error::Config(format!(
            "TSV source {:?} needs explicit column names",
            source.name
        ))
    })?;
    let text_index = columns
        .iter()
        .position(|column| column == &source.text_column)
        .ok_or_else(|| {
            Error::Config(format!(
                "TSV source {:?} is missing text column {:?}",
                source.name, source.text_column
            ))
        })?;
    let label_index = columns
        .iter()
        .position(|column| column == &source.label_column)
        .ok_or_else(|| {
            Error::Config(format!(
                "TSV source {:?} is missing label column {:?}",
                source.name, source.label_column
            ))
        })?;

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .quoting(false)
        .from_path(&source.path)
        .map_err(|error| csv_error(&source.path, error))?;

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|error| csv_error(&source.path, error))?;
        let text = record.get(text_index).unwrap_or("").to_string();
        let language = record.get(label_index).unwrap_or("").to_string();
        records.push(RawRecord {
            text,
            language,
            source: source.name.clone(),
            role: source.role.clone(),
        });
    }
    Ok(records)
}

fn column_index(headers: &csv::StringRecord, name: &str, path: &Path) -> Result<usize> {
    headers
        .iter()
        .position(|header| header.trim_start_matches('\u{feff}') == name)
        .ok_or_else(|| {
            Error::Dataset(format!(
                "missing required column {name:?} in {}; found {:?}",
                path.display(),
                headers.iter().collect::<Vec<_>>()
            ))
        })
}

fn validate_and_normalise(
    raw: Vec<RawRecord>,
) -> Result<(Vec<Record>, Vec<InvalidRecord>, ValidationReport)> {
    let input_rows = raw.len();
    let mut records = Vec::with_capacity(input_rows);
    let mut invalid = Vec::new();
    let mut report = ValidationReport {
        input_rows,
        ..ValidationReport::default()
    };

    for row in raw {
        let raw_text = row.text;
        let raw_language = row.language;
        let language = alias_label(raw_language.trim().replace("\\N", "").trim()).to_string();
        let text = linguae_core::text::normalize_text(&raw_text);

        let reason = if raw_text.trim().is_empty() {
            Some("empty text after strip")
        } else if language.is_empty() {
            Some("empty language label")
        } else if text.is_empty() {
            Some("empty text after normalisation")
        } else {
            None
        };

        if raw_text.trim().is_empty() {
            report.empty_text_rows += 1;
        }
        if language.is_empty() {
            report.empty_label_rows += 1;
        }
        if text.is_empty() {
            report.empty_after_normalisation_rows += 1;
        }
        if text != raw_text {
            report.rows_changed_by_normalisation += 1;
        }

        if let Some(reason) = reason {
            report.invalid_rows += 1;
            invalid.push(InvalidRecord {
                text: raw_text,
                language: raw_language,
                source: row.source,
                reason: reason.into(),
            });
            continue;
        }

        *report.rows_by_source.entry(row.source.clone()).or_insert(0) += 1;
        records.push(Record {
            text,
            language,
            source: row.source,
            role: row.role,
            split: Split::Train,
        });
    }

    report.valid_rows = records.len();
    if records.is_empty() {
        return Err(Error::Dataset(
            "no valid rows remain after validation; check the dataset".into(),
        ));
    }

    let mut language_counts: HashMap<&str, usize> = HashMap::new();
    let mut text_language: HashMap<&str, &str> = HashMap::new();
    let mut conflicting: HashMap<&str, ()> = HashMap::new();
    let mut duplicate_text_label = 0usize;

    for record in &records {
        *language_counts.entry(record.language.as_str()).or_insert(0) += 1;
        match text_language.get(record.text.as_str()) {
            Some(language) if *language != record.language => {
                conflicting.insert(record.text.as_str(), ());
            }
            None => {
                text_language.insert(record.text.as_str(), record.language.as_str());
            }
            _ => {}
        }
    }

    let mut seen_pairs: HashMap<(&str, &str), ()> = HashMap::new();
    for record in &records {
        if seen_pairs
            .insert((record.text.as_str(), record.language.as_str()), ())
            .is_some()
        {
            duplicate_text_label += 1;
        }
    }

    report.unique_languages = language_counts.len();
    report.min_examples_per_language = language_counts.values().copied().min().unwrap_or(0);
    report.max_examples_per_language = language_counts.values().copied().max().unwrap_or(0);
    report.texts_with_conflicting_labels = conflicting.len();
    report.duplicate_text_label_rows = duplicate_text_label;
    report.duplicate_text_rows = records.len().saturating_sub(text_language.len());

    if report.empty_label_rows > 0 {
        warn!(
            "{} row(s) have an empty language label and were excluded",
            report.empty_label_rows
        );
    }
    if report.texts_with_conflicting_labels > 0 {
        warn!(
            "{} text(s) have contradictory labels and will be dropped",
            report.texts_with_conflicting_labels
        );
    }

    info!(
        "validation: {} input rows, {} valid rows, {} invalid rows, {} languages",
        report.input_rows, report.valid_rows, report.invalid_rows, report.unique_languages
    );

    Ok((records, invalid, report))
}

fn csv_error(path: &Path, source: csv::Error) -> Error {
    Error::Csv {
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_known_tatoeba_codes() {
        assert_eq!(alias_label("cmn"), "zho");
        assert_eq!(alias_label("swh"), "swa");
        assert_eq!(alias_label("eng"), "eng");
    }
}
