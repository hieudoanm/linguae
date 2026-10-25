use std::collections::HashMap;
use std::path::{Path, PathBuf};

use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

use crate::error::{Error, Result};
use crate::io::{read_lines, write_csv_atomic, FileSummary};
use crate::stats::{LengthHistogram, LengthStats};

/// Default cap on the number of words written per language.
pub const DEFAULT_MAX_WORDS: usize = 100_000;

/// Inputs and output for the per-language word extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordOptions {
    pub input_dir: PathBuf,
    pub output: PathBuf,
    /// Maximum words per language; `0` keeps every word.
    pub max_words: usize,
    /// Include the WiLI test split in the vocabulary. The training split is
    /// always read.
    pub include_test: bool,
}

/// Summary of a completed word extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordReport {
    pub languages: usize,
    pub words: u64,
    pub max_words_per_language: usize,
    pub words_per_language: LengthStats,
    pub file: FileSummary,
}

/// Extract the most frequent words for every language in WiLI-2018.
///
/// The output has `text,language,count` columns, ordered by language and then
/// by descending frequency (ties break alphabetically). Tokens are Unicode
/// word segments, NFC-normalised and lowercased. Tokens without a letter and
/// tokens longer than 64 characters are ignored.
pub fn extract(options: &WordOptions) -> Result<WordReport> {
    let splits: &[&str] = if options.include_test {
        &["train", "test"]
    } else {
        &["train"]
    };

    let mut counts: HashMap<String, HashMap<String, u64>> = HashMap::new();
    let mut texts_seen = 0u64;

    for split in splits {
        let pairs = read_split(&options.input_dir, split)?;
        log::info!("tokenising {split} split: {} texts", pairs.len());
        for (text, language) in pairs {
            let language = language.trim();
            if language.is_empty() {
                continue;
            }
            let language_counts = counts.entry(language.to_string()).or_default();
            for word in tokenize(&text) {
                *language_counts.entry(word).or_default() += 1;
            }
            texts_seen += 1;
            if texts_seen % 25_000 == 0 {
                log::info!("tokenised {texts_seen} texts");
            }
        }
    }

    // Languages whose texts contain no usable words should not appear in the
    // output (or count as an empty dataset).
    counts.retain(|_, language_counts| !language_counts.is_empty());
    if counts.is_empty() {
        return Err(Error::InvalidData(
            "no words were extracted from the WiLI dataset".into(),
        ));
    }

    let mut languages: Vec<&String> = counts.keys().collect();
    languages.sort();

    let mut per_language = LengthHistogram::default();
    let mut total_words = 0u64;
    let mut max_words_per_language = 0usize;

    let file = write_csv_atomic(&options.output, &["text", "language", "count"], |writer| {
        for language in &languages {
            let language_counts = counts
                .get(*language)
                .expect("the language came from the counts map");
            let mut words: Vec<(&String, &u64)> = language_counts.iter().collect();
            words.sort_by(|(left_word, left_count), (right_word, right_count)| {
                right_count
                    .cmp(left_count)
                    .then_with(|| left_word.cmp(right_word))
            });

            let limit = if options.max_words == 0 {
                words.len()
            } else {
                options.max_words.min(words.len())
            };
            per_language.push(limit);
            max_words_per_language = max_words_per_language.max(limit);

            for (word, count) in words.into_iter().take(limit) {
                let count = count.to_string();
                writer
                    .write_record([word.as_str(), language.as_str(), count.as_str()])
                    .map_err(|source| Error::Csv {
                        path: options.output.clone(),
                        source,
                    })?;
                total_words += 1;
            }
        }
        Ok(())
    })?;

    let words_per_language = per_language
        .finish()
        .ok_or_else(|| Error::InvalidData("no words were written to the word dataset".into()))?;

    log::info!(
        "wrote {} ({total_words} words, {} languages, {:.1} MiB, sha256 {})",
        options.output.display(),
        languages.len(),
        file.bytes as f64 / (1024.0 * 1024.0),
        file.sha256
    );
    log::info!(
        "words per language: min={} median={} max={}",
        words_per_language.min,
        words_per_language.median,
        words_per_language.max
    );

    Ok(WordReport {
        languages: languages.len(),
        words: total_words,
        max_words_per_language,
        words_per_language,
        file,
    })
}

/// Tokenise one string into lowercase Unicode words suitable for the dataset.
pub fn tokenize(text: &str) -> Vec<String> {
    let normalized: String = text.nfc().collect();
    let mut words = Vec::new();
    for word in normalized.unicode_words() {
        let lower: String = word.to_lowercase().nfc().collect();
        if lower.chars().any(char::is_alphabetic) && lower.chars().count() <= 64 {
            words.push(lower);
        }
    }
    words
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_latin_words_and_keeps_apostrophes() {
        assert_eq!(
            tokenize("Hello, world! It's the user's choice."),
            vec!["hello", "world", "it's", "the", "user's", "choice"]
        );
    }

    #[test]
    fn lowercases_and_preserves_accents() {
        assert_eq!(tokenize("ÉLAN, déjà vu"), vec!["élan", "déjà", "vu"]);
    }

    #[test]
    fn splits_scripts_without_spaces_into_characters() {
        assert_eq!(tokenize("你好世界"), vec!["你", "好", "世", "界"]);
    }

    #[test]
    fn drops_numbers_and_punctuation() {
        assert_eq!(tokenize("2024 — 42, ok!"), vec!["ok"]);
    }
}
