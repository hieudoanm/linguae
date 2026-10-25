use std::collections::{BTreeMap, HashMap, HashSet};

use log::{info, warn};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::rng::Rng;
use crate::source::Record;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SplitDetails {
    pub rows_per_split: BTreeMap<String, usize>,
    pub languages_per_split: BTreeMap<String, usize>,
    pub rows_per_source: BTreeMap<String, usize>,
    pub rows_per_source_split: BTreeMap<String, BTreeMap<String, usize>>,
    pub text_overlap: BTreeMap<String, usize>,
    pub examples_per_language_by_split: BTreeMap<String, BTreeMap<String, usize>>,
}

#[derive(Debug, Clone, Default)]
pub struct LengthStats {
    pub min: usize,
    pub median: f64,
    pub p90: f64,
    pub p99: f64,
    pub max: usize,
    pub mean: f64,
}

#[derive(Debug, Clone, Default)]
pub struct DatasetStats {
    pub label_counts: BTreeMap<String, usize>,
    pub source_counts: BTreeMap<String, usize>,
    pub lengths: LengthStats,
    pub lengths_by_source: BTreeMap<String, LengthStats>,
    pub duplicate_text_rows: usize,
    pub duplicate_text_label_rows: usize,
}

/// Drop contradictory labels, exact duplicates, rare labels and oversized groups.
pub fn clean_records(mut records: Vec<Record>, config: &Config) -> Result<Vec<Record>> {
    records = drop_conflicting_texts(records);
    records = drop_duplicate_pairs(records);
    records = drop_rare_languages(records, config.min_examples_per_language);

    let word_sources: HashSet<&str> = config
        .sources
        .iter()
        .filter(|source| source.role == "words")
        .map(|source| source.name.as_str())
        .collect();

    let mut keep = vec![true; records.len()];
    let sentence_indices: Vec<usize> = records
        .iter()
        .enumerate()
        .filter(|(_, record)| !word_sources.contains(record.source.as_str()))
        .map(|(index, _)| index)
        .collect();
    let word_indices: Vec<usize> = records
        .iter()
        .enumerate()
        .filter(|(_, record)| word_sources.contains(record.source.as_str()))
        .map(|(index, _)| index)
        .collect();

    apply_group_cap(
        &mut keep,
        &records,
        &sentence_indices,
        config.max_rows_per_language_per_source,
        config.seed,
        "per-source cap",
    );
    apply_group_cap(
        &mut keep,
        &records,
        &word_indices,
        config.max_words_per_language,
        config.seed,
        "word cap",
    );

    if config.max_rows_per_language > 0 {
        let all: Vec<usize> = (0..records.len()).collect();
        apply_language_cap(
            &mut keep,
            &records,
            &all,
            config.max_rows_per_language,
            config.seed,
            "combined cap",
        );
    }

    if let Some(limit) = config.max_rows {
        let all: Vec<usize> = (0..records.len()).collect();
        apply_global_cap(&mut keep, &all, limit, config.seed);
    }

    let before = records.len();
    let mut mask = keep.into_iter();
    records.retain(|_| mask.next().unwrap_or(false));
    info!(
        "dataset: {} -> {} rows after capping",
        before,
        records.len()
    );
    Ok(records)
}

fn drop_conflicting_texts(records: Vec<Record>) -> Vec<Record> {
    let mut text_language: HashMap<&str, &str> = HashMap::new();
    let mut conflicting: HashSet<String> = HashSet::new();

    for record in &records {
        match text_language.get(record.text.as_str()) {
            Some(language) if *language != record.language => {
                conflicting.insert(record.text.clone());
            }
            None => {
                text_language.insert(record.text.as_str(), record.language.as_str());
            }
            _ => {}
        }
    }
    drop(text_language);

    if conflicting.is_empty() {
        return records;
    }
    let before = records.len();
    let kept: Vec<Record> = records
        .into_iter()
        .filter(|record| !conflicting.contains(&record.text))
        .collect();
    warn!(
        "dropped {} row(s) with contradictory labels across {} text(s)",
        before - kept.len(),
        conflicting.len()
    );
    kept
}

fn drop_duplicate_pairs(records: Vec<Record>) -> Vec<Record> {
    let mut text_ids: HashMap<&str, usize> = HashMap::new();
    let mut label_ids: HashMap<&str, usize> = HashMap::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut keep = Vec::with_capacity(records.len());

    for record in &records {
        let next_text = text_ids.len();
        let text_id = *text_ids.entry(record.text.as_str()).or_insert(next_text);
        let next_label = label_ids.len();
        let label_id = *label_ids
            .entry(record.language.as_str())
            .or_insert(next_label);
        keep.push(seen.insert((text_id, label_id)));
    }

    let before = records.len();
    let mut mask = keep.into_iter();
    let kept: Vec<Record> = records
        .into_iter()
        .filter(|_| mask.next().unwrap_or(false))
        .collect();
    if before != kept.len() {
        info!(
            "dropped {} exact duplicate (text, language) row(s)",
            before - kept.len()
        );
    }
    kept
}

fn drop_rare_languages(records: Vec<Record>, minimum: usize) -> Vec<Record> {
    if minimum == 0 {
        return records;
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for record in &records {
        *counts.entry(record.language.as_str()).or_insert(0) += 1;
    }
    let rare: HashSet<String> = counts
        .iter()
        .filter(|(_, count)| **count < minimum)
        .map(|(language, _)| (*language).to_string())
        .collect();
    let total_languages = counts.len();
    drop(counts);
    if rare.is_empty() {
        info!(
            "all {} languages have at least {} examples",
            total_languages, minimum
        );
        return records;
    }
    let before = records.len();
    let kept: Vec<Record> = records
        .into_iter()
        .filter(|record| !rare.contains(&record.language))
        .collect();
    info!(
        "dropping {} language(s) with fewer than {} examples; {} -> {} rows",
        rare.len(),
        minimum,
        before,
        kept.len()
    );
    kept
}

fn apply_group_cap(
    keep: &mut [bool],
    records: &[Record],
    indices: &[usize],
    limit: usize,
    seed: u64,
    label: &str,
) {
    if limit == 0 || indices.is_empty() {
        if limit == 0 {
            info!("{label}: keeping every row (limit=0)");
        }
        return;
    }
    let mut groups: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
    for &index in indices {
        let record = &records[index];
        groups
            .entry((record.language.as_str(), record.source.as_str()))
            .or_default()
            .push(index);
    }
    let before = indices.len();
    let mut rng = Rng::new(seed);
    for group in groups.values() {
        if group.len() <= limit {
            continue;
        }
        for &index in group {
            keep[index] = false;
        }
        for position in rng.choose_indices(group.len(), limit) {
            keep[group[position]] = true;
        }
    }
    let after = keep.iter().filter(|value| **value).count();
    info!("{label}: {before} -> {after} rows (limit={limit} per group)");
}

fn apply_language_cap(
    keep: &mut [bool],
    records: &[Record],
    indices: &[usize],
    limit: usize,
    seed: u64,
    label: &str,
) {
    if limit == 0 {
        info!("{label}: keeping every row (limit=0)");
        return;
    }
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for &index in indices {
        groups
            .entry(records[index].language.as_str())
            .or_default()
            .push(index);
    }
    let before = indices.len();
    let mut rng = Rng::new(seed);
    for group in groups.values() {
        if group.len() <= limit {
            continue;
        }
        for &index in group {
            keep[index] = false;
        }
        for position in rng.choose_indices(group.len(), limit) {
            keep[group[position]] = true;
        }
    }
    let after = keep.iter().filter(|value| **value).count();
    info!("{label}: {before} -> {after} rows (limit={limit} per language)");
}

fn apply_global_cap(keep: &mut [bool], indices: &[usize], limit: usize, seed: u64) {
    if indices.len() <= limit {
        return;
    }
    let mut rng = Rng::new(seed);
    let chosen: HashSet<usize> = rng
        .choose_indices(indices.len(), limit)
        .into_iter()
        .collect();
    for (position, &index) in indices.iter().enumerate() {
        keep[index] = chosen.contains(&position);
    }
    info!("global cap: {} -> {} rows", indices.len(), limit);
}

/// Deterministically split each language into train/validation/test.
pub fn split_records(
    mut records: Vec<Record>,
    config: &Config,
) -> Result<(Vec<Record>, SplitDetails)> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, record) in records.iter().enumerate() {
        groups
            .entry(record.language.clone())
            .or_default()
            .push(index);
    }

    let mut rng = Rng::new(config.seed);
    for (language, group) in &groups {
        let mut shuffled = group.clone();
        rng.shuffle(&mut shuffled);
        let count = shuffled.len();
        let n_val = round_half_even(count as f64 * config.val_fraction).max(1);
        let n_test = round_half_even(count as f64 * config.test_fraction).max(1);
        if count <= n_val + n_test {
            return Err(Error::Dataset(format!(
                "language {language:?} has only {count} unique text(s); at least 3 are needed"
            )));
        }
        let n_train = count - n_val - n_test;
        for (position, &index) in shuffled.iter().enumerate() {
            records[index].split = if position < n_train {
                Split::Train
            } else if position < n_train + n_val {
                Split::Validation
            } else {
                Split::Test
            };
        }
    }

    let train_languages: HashSet<&str> = records
        .iter()
        .filter(|record| record.split == Split::Train)
        .map(|record| record.language.as_str())
        .collect();
    let evaluation_languages: HashSet<&str> = records
        .iter()
        .filter(|record| record.split != Split::Train)
        .map(|record| record.language.as_str())
        .collect();
    let missing: Vec<&str> = evaluation_languages
        .difference(&train_languages)
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(Error::Dataset(format!(
            "evaluation languages missing from training: {missing:?}"
        )));
    }

    let details = split_details(&records, &groups);
    info!(
        "split: train={} validation={} test={}",
        details.rows_per_split.get("train").copied().unwrap_or(0),
        details
            .rows_per_split
            .get("validation")
            .copied()
            .unwrap_or(0),
        details.rows_per_split.get("test").copied().unwrap_or(0),
    );
    Ok((records, details))
}

fn split_details(records: &[Record], groups: &BTreeMap<String, Vec<usize>>) -> SplitDetails {
    let mut details = SplitDetails::default();
    for split in [Split::Train, Split::Validation, Split::Test] {
        let name = split.as_str().to_string();
        details.rows_per_split.insert(name.clone(), 0);
        details.languages_per_split.insert(name.clone(), 0);
        details
            .rows_per_source_split
            .insert(name.clone(), BTreeMap::new());
    }

    for record in records {
        let split = record.split.as_str().to_string();
        *details.rows_per_split.entry(split.clone()).or_insert(0) += 1;
        *details
            .rows_per_source
            .entry(record.source.clone())
            .or_insert(0) += 1;
        *details
            .rows_per_source_split
            .entry(split.clone())
            .or_default()
            .entry(record.source.clone())
            .or_insert(0) += 1;
    }

    for split in [Split::Train, Split::Validation, Split::Test] {
        let name = split.as_str().to_string();
        let languages: HashSet<&str> = records
            .iter()
            .filter(|record| record.split == split)
            .map(|record| record.language.as_str())
            .collect();
        details.languages_per_split.insert(name, languages.len());
    }

    for language in groups.keys() {
        let mut counts = BTreeMap::new();
        for split in [Split::Train, Split::Validation, Split::Test] {
            let name = split.as_str().to_string();
            let count = records
                .iter()
                .filter(|record| &record.language == language && record.split == split)
                .count();
            counts.insert(name, count);
        }
        details
            .examples_per_language_by_split
            .insert(language.clone(), counts);
    }

    for (left, right) in [
        (Split::Train, Split::Validation),
        (Split::Train, Split::Test),
        (Split::Validation, Split::Test),
    ] {
        let left_texts: HashSet<&str> = records
            .iter()
            .filter(|record| record.split == left)
            .map(|record| record.text.as_str())
            .collect();
        let right_texts: HashSet<&str> = records
            .iter()
            .filter(|record| record.split == right)
            .map(|record| record.text.as_str())
            .collect();
        details.text_overlap.insert(
            format!("{}|{}", left.as_str(), right.as_str()),
            left_texts.intersection(&right_texts).count(),
        );
    }

    details
}

/// Summarise label counts, source counts and character lengths.
pub fn dataset_stats(records: &[Record]) -> DatasetStats {
    let mut stats = DatasetStats::default();
    let mut lengths = Vec::with_capacity(records.len());
    let mut lengths_by_source: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    for record in records {
        *stats
            .label_counts
            .entry(record.language.clone())
            .or_insert(0) += 1;
        *stats
            .source_counts
            .entry(record.source.clone())
            .or_insert(0) += 1;
        let length = record.text.chars().count();
        lengths.push(length);
        lengths_by_source
            .entry(record.source.clone())
            .or_default()
            .push(length);
    }

    stats.lengths = length_stats(lengths);
    for (source, values) in lengths_by_source {
        stats.lengths_by_source.insert(source, length_stats(values));
    }
    stats.duplicate_text_rows = records.len().saturating_sub(
        records
            .iter()
            .map(|record| record.text.as_str())
            .collect::<HashSet<_>>()
            .len(),
    );
    stats
}

pub fn length_stats(mut lengths: Vec<usize>) -> LengthStats {
    if lengths.is_empty() {
        return LengthStats::default();
    }
    lengths.sort_unstable();
    LengthStats {
        min: lengths[0],
        median: percentile(&lengths, 0.5),
        p90: percentile(&lengths, 0.9),
        p99: percentile(&lengths, 0.99),
        max: *lengths.last().unwrap(),
        mean: lengths.iter().sum::<usize>() as f64 / lengths.len() as f64,
    }
}

pub fn percentile(sorted: &[usize], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower] as f64
    } else {
        let weight = position - lower as f64;
        sorted[lower] as f64 * (1.0 - weight) + sorted[upper] as f64 * weight
    }
}

fn round_half_even(value: f64) -> usize {
    let floor = value.floor();
    let fraction = value - floor;
    let rounded = if (fraction - 0.5).abs() < 1e-12 {
        if (floor as u64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    };
    rounded as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_like_python() {
        assert_eq!(round_half_even(2.5), 2);
        assert_eq!(round_half_even(3.5), 4);
        assert_eq!(round_half_even(2.4), 2);
        assert_eq!(round_half_even(2.6), 3);
    }

    #[test]
    fn percentile_uses_linear_interpolation() {
        assert_eq!(percentile(&[0, 10], 0.5), 5.0);
        assert_eq!(percentile(&[0, 10, 20], 0.9), 18.0);
    }
}
