use std::collections::HashSet;
use std::path::PathBuf;

use crate::error::{Error, Result};
use crate::io::{write_csv_atomic, FileSummary};
use crate::stats::{LengthHistogram, LengthStats};

/// Inputs and output for the Tatoeba conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TatoebaOptions {
    pub input: PathBuf,
    pub output: PathBuf,
}

/// Summary of a completed Tatoeba conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TatoebaReport {
    pub input_rows: u64,
    pub output_rows: u64,
    pub dropped_rows: u64,
    pub languages: usize,
    pub lengths: LengthStats,
    pub file: FileSummary,
}

/// Convert the headerless `id`, `language`, `text` Tatoeba TSV into a CSV.
///
/// Rows with an empty language label or the literal `\N` are dropped. Text is
/// preserved verbatim, including commas, quotes and tabs.
pub fn convert(options: &TatoebaOptions) -> Result<TatoebaReport> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .quoting(false)
        .flexible(true)
        .from_path(&options.input)
        .map_err(|source| Error::Csv {
            path: options.input.clone(),
            source,
        })?;

    let mut input_rows = 0u64;
    let mut output_rows = 0u64;
    let mut dropped_rows = 0u64;
    let mut lengths = LengthHistogram::default();
    let mut languages = HashSet::new();

    let file = write_csv_atomic(&options.output, &["id", "language", "text"], |writer| {
        for record in reader.records() {
            let record = record.map_err(|source| Error::Csv {
                path: options.input.clone(),
                source,
            })?;

            if record.is_empty()
                || (record.len() == 1 && record.get(0).map(str::is_empty).unwrap_or(false))
            {
                continue;
            }
            input_rows += 1;

            if record.len() < 3 {
                return Err(Error::InvalidData(format!(
                    "{}: row {input_rows} has {} fields; expected at least 3",
                    options.input.display(),
                    record.len()
                )));
            }

            let id = record.get(0).unwrap_or("");
            let language = record.get(1).unwrap_or("");
            if language.trim().is_empty() || language == "\\N" {
                dropped_rows += 1;
                continue;
            }

            let text = record.iter().skip(2).collect::<Vec<_>>().join("\t");
            lengths.push(text.chars().count());
            languages.insert(language.to_string());

            writer
                .write_record([id, language, text.as_str()])
                .map_err(|source| Error::Csv {
                    path: options.output.clone(),
                    source,
                })?;
            output_rows += 1;
        }
        Ok(())
    })?;

    log::info!(
        "read {} rows; dropped {} with empty or \\N labels",
        input_rows,
        dropped_rows
    );
    log::info!("unique languages: {}", languages.len());
    let lengths = lengths.finish().ok_or_else(|| {
        Error::InvalidData("no valid rows remain after removing invalid labels".into())
    })?;
    log::info!(
        "character length: min={} median={} max={}",
        lengths.min,
        lengths.median,
        lengths.max
    );
    log::info!(
        "wrote {} ({output_rows} rows, {:.1} MiB, sha256 {})",
        options.output.display(),
        file.bytes as f64 / (1024.0 * 1024.0),
        file.sha256
    );

    Ok(TatoebaReport {
        input_rows,
        output_rows,
        dropped_rows,
        languages: languages.len(),
        lengths,
        file,
    })
}
