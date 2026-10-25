use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{SecondsFormat, Utc};
use log::info;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::dataset::{DatasetStats, SplitDetails};
use crate::error::{Error, Result};
use crate::metrics::{
    classification_by_length, classification_metrics, confidence_summary, confusion_matrix,
    length_summary, length_summary_by_source, ClassificationMetrics, ConfidenceSummary,
    ConfusionMatrix, LengthBucket, LengthBucketBySource, PerLanguageMetrics, TestPrediction,
};
use crate::model::{CharCnn, ModelConfig};
use crate::source::{Record, ValidationReport};
use crate::train::{EpochRecord, TrainResult};
use crate::vocab::{Vocabulary, PAD_ID, UNK_ID};

/// Evaluation outputs derived from the held-out test split.
pub struct Evaluation {
    pub metrics: ClassificationMetrics,
    pub per_language: Vec<PerLanguageMetrics>,
    pub confusion: ConfusionMatrix,
    pub by_source: BTreeMap<String, SourceMetrics>,
    pub length_buckets: Vec<LengthBucket>,
    pub length_classification: Vec<crate::metrics::LengthClassificationMetrics>,
    pub length_buckets_by_source: Vec<LengthBucketBySource>,
    pub confidence: ConfidenceSummary,
    pub length_correlation: Option<f64>,
    pub misclassified: Vec<MisclassifiedRow>,
    pub leakage: LeakageChecks,
}

#[derive(Debug, Clone)]
pub struct SourceMetrics {
    pub accuracy: f64,
    pub examples: usize,
    pub languages: usize,
    pub predictions: BTreeMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct MisclassifiedRow {
    pub text: String,
    pub expected: String,
    pub predicted: String,
    pub source: String,
    pub role: String,
    pub confidence: f32,
    pub expected_idx: u32,
    pub predicted_idx: u32,
    pub char_length: usize,
}

#[derive(Debug, Clone)]
pub struct LeakageChecks {
    pub duplicate_texts_train_vs_validation: usize,
    pub duplicate_texts_train_vs_test: usize,
    pub duplicate_texts_validation_vs_test: usize,
    pub evaluation_languages_missing_from_train: usize,
    pub test_examples_per_language_min: usize,
    pub test_examples_per_language_max: usize,
    pub class_imbalance_ratio: Option<f64>,
    pub suspiciously_high_accuracy: bool,
}

impl LeakageChecks {
    pub fn leaked(&self) -> usize {
        self.duplicate_texts_train_vs_validation
            .max(self.duplicate_texts_train_vs_test)
            .max(self.duplicate_texts_validation_vs_test)
    }
}

/// Compute classification, source, length and confidence summaries.
pub fn evaluate(
    test_records: &[Record],
    all_records: &[Record],
    predictions: &[TestPrediction],
    labels: &[String],
) -> Result<Evaluation> {
    if test_records.len() != predictions.len() {
        return Err(Error::Training(format!(
            "prediction/truth length mismatch: {} != {}",
            predictions.len(),
            test_records.len()
        )));
    }

    let expected: Vec<u32> = predictions.iter().map(|item| item.expected).collect();
    let predicted: Vec<u32> = predictions.iter().map(|item| item.predicted).collect();
    let confusion = confusion_matrix(&expected, &predicted, labels.len());
    let (metrics, per_language) = classification_metrics(&confusion, labels);

    let mut by_source: BTreeMap<String, SourceMetrics> = BTreeMap::new();
    for (record, prediction) in test_records.iter().zip(predictions.iter()) {
        let entry = by_source
            .entry(record.source.clone())
            .or_insert_with(|| SourceMetrics {
                accuracy: 0.0,
                examples: 0,
                languages: 0,
                predictions: BTreeMap::new(),
            });
        entry.examples += 1;
        if prediction.expected == prediction.predicted {
            entry.accuracy += 1.0;
        }
        let label = labels
            .get(prediction.predicted as usize)
            .cloned()
            .unwrap_or_default();
        *entry.predictions.entry(label).or_insert(0) += 1;
    }
    for (source, entry) in &mut by_source {
        entry.accuracy /= entry.examples.max(1) as f64;
        entry.languages = test_records
            .iter()
            .filter(|record| &record.source == source)
            .map(|record| record.language.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len();
    }

    let length_buckets = length_summary(test_records, predictions);
    let length_classification = classification_by_length(test_records, predictions, labels);
    let length_buckets_by_source = length_summary_by_source(test_records, predictions);
    let confidence = confidence_summary(predictions);

    let correctness: Vec<f64> = predictions
        .iter()
        .map(|prediction| (prediction.expected == prediction.predicted) as u8 as f64)
        .collect();
    let lengths: Vec<f64> = test_records
        .iter()
        .map(|record| record.text.chars().count() as f64)
        .collect();
    let length_correlation = crate::metrics::pearson(&lengths, &correctness);

    let mut misclassified: Vec<MisclassifiedRow> = test_records
        .iter()
        .zip(predictions.iter())
        .filter(|(_, prediction)| prediction.expected != prediction.predicted)
        .map(|(record, prediction)| MisclassifiedRow {
            text: record.text.clone(),
            expected: labels
                .get(prediction.expected as usize)
                .cloned()
                .unwrap_or_default(),
            predicted: labels
                .get(prediction.predicted as usize)
                .cloned()
                .unwrap_or_default(),
            source: record.source.clone(),
            role: record.role.clone(),
            confidence: prediction.confidence,
            expected_idx: prediction.expected,
            predicted_idx: prediction.predicted,
            char_length: record.text.chars().count(),
        })
        .collect();
    misclassified.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.text.cmp(&right.text))
    });

    let leakage = leakage_checks(all_records, test_records, &metrics);

    Ok(Evaluation {
        metrics,
        per_language,
        confusion,
        by_source,
        length_buckets,
        length_classification,
        length_buckets_by_source,
        confidence,
        length_correlation,
        misclassified,
        leakage,
    })
}

fn leakage_checks(
    records: &[Record],
    test_records: &[Record],
    metrics: &ClassificationMetrics,
) -> LeakageChecks {
    let texts = |split: crate::dataset::Split| -> std::collections::HashSet<&str> {
        records
            .iter()
            .filter(|record| record.split == split)
            .map(|record| record.text.as_str())
            .collect()
    };
    let train = texts(crate::dataset::Split::Train);
    let validation = texts(crate::dataset::Split::Validation);
    let test = texts(crate::dataset::Split::Test);

    let mut per_language: BTreeMap<&str, usize> = BTreeMap::new();
    for record in test_records {
        *per_language.entry(record.language.as_str()).or_insert(0) += 1;
    }
    let min = per_language.values().copied().min().unwrap_or(0);
    let max = per_language.values().copied().max().unwrap_or(0);

    let train_languages: std::collections::HashSet<&str> = records
        .iter()
        .filter(|record| record.split == crate::dataset::Split::Train)
        .map(|record| record.language.as_str())
        .collect();
    let evaluation_languages: std::collections::HashSet<&str> = records
        .iter()
        .filter(|record| record.split != crate::dataset::Split::Train)
        .map(|record| record.language.as_str())
        .collect();

    LeakageChecks {
        duplicate_texts_train_vs_validation: train.intersection(&validation).count(),
        duplicate_texts_train_vs_test: train.intersection(&test).count(),
        duplicate_texts_validation_vs_test: validation.intersection(&test).count(),
        evaluation_languages_missing_from_train: evaluation_languages
            .difference(&train_languages)
            .count(),
        test_examples_per_language_min: min,
        test_examples_per_language_max: max,
        class_imbalance_ratio: if min > 0 {
            Some(max as f64 / min as f64)
        } else {
            None
        },
        suspiciously_high_accuracy: metrics.accuracy > 0.999,
    }
}

pub struct ArtifactReport<'a> {
    pub config: &'a Config,
    pub model_config: &'a ModelConfig,
    pub model: &'a CharCnn,
    pub vocabulary: &'a Vocabulary,
    pub labels: &'a [String],
    pub records: &'a [Record],
    pub validation_report: &'a ValidationReport,
    pub dataset_stats: &'a DatasetStats,
    pub split_details: &'a SplitDetails,
    pub history: &'a [EpochRecord],
    pub train_result: &'a TrainResult,
    pub evaluation: &'a Evaluation,
    pub device: &'a str,
    pub unknown_character_rate: BTreeMap<String, f64>,
}

/// Recreate the artifact directory so every training run exports a clean artifact.
pub fn prepare_artifact_dir(config: &Config) -> Result<PathBuf> {
    let directory = &config.artifact_dir;
    if directory.exists() {
        fs::remove_dir_all(directory).map_err(|source| Error::Write {
            path: directory.clone(),
            source,
        })?;
    }
    fs::create_dir_all(directory).map_err(|source| Error::Write {
        path: directory.clone(),
        source,
    })?;
    Ok(directory.clone())
}

/// Write the label mapping used by training to `<processed_dir>/labels.json`.
pub fn write_label_mapping(processed_dir: &Path, labels: &[String]) -> Result<()> {
    let label2idx: BTreeMap<&str, usize> = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect();
    let idx2label: BTreeMap<String, &str> = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (index.to_string(), label.as_str()))
        .collect();
    let path = processed_dir.join("labels.json");
    write_json(
        &path,
        &json!({ "label2idx": label2idx, "idx2label": idx2label }),
    )?;
    info!("label mapping written to {}", path.display());
    Ok(())
}

/// Write every artifact file, including the safetensors weights and manifest.
pub fn write_artifact(report: &ArtifactReport<'_>) -> Result<()> {
    let directory = prepare_artifact_dir(report.config)?;
    let created_at = Utc::now().to_rfc3339_opts(SecondsFormat::Micros, true);
    let model_config = report.model_config;

    let export_config = json!({
        "schema_version": 1,
        "model_type": "character_cnn",
        "framework": "candle",
        "artifact_schema": "linguae.language-detector/2",
        "embedding_dim": model_config.embedding_dim,
        "num_filters": model_config.num_filters,
        "kernel_sizes": model_config.kernel_sizes,
        "dropout": model_config.dropout,
        "vocab_size": model_config.vocab_size,
        "num_classes": model_config.num_classes,
        "max_length": report.config.max_length,
        "padding_idx": PAD_ID,
        "unknown_idx": UNK_ID,
        "special_tokens": ["<pad>", "<unk>"],
        "padding_side": "right",
        "pooling": if model_config.mean_pooling { "masked_max+masked_mean" } else { "masked_max" },
        "mean_pooling": model_config.mean_pooling,
        "length_embedding": model_config.length_embedding,
        "length_embedding_dim": model_config.length_embedding_dim,
        "length_bucket_boundaries": report.config.length_bucket_boundaries,
        "normalisation": "NFC + whitespace collapse",
        "lowercase": false,
        "prediction_scope": "single-label-per-string",
        "weights": "model.safetensors",
    });
    write_json(&directory.join("config.json"), &export_config)?;

    let vocabulary_value =
        serde_json::to_value(&report.vocabulary.token_to_id).map_err(|error| Error::Json {
            path: directory.join("vocabulary.json"),
            source: error,
        })?;
    write_json(&directory.join("vocabulary.json"), &vocabulary_value)?;

    let label2idx: BTreeMap<&str, usize> = report
        .labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.as_str(), index))
        .collect();
    let idx2label: BTreeMap<String, &str> = report
        .labels
        .iter()
        .enumerate()
        .map(|(index, label)| (index.to_string(), label.as_str()))
        .collect();
    write_json(
        &directory.join("labels.json"),
        &json!({ "label2idx": label2idx, "idx2label": idx2label }),
    )?;

    let model_path = directory.join("model.safetensors");
    report.model.save(&model_path)?;

    write_training_history(&directory.join("training_history.csv"), report.history)?;
    write_confusion_matrix(
        &directory.join("confusion_matrix.csv"),
        &report.evaluation.confusion,
        report.labels,
    )?;
    write_misclassified(
        &directory.join("misclassified_examples.csv"),
        &report.evaluation.misclassified,
        report.config.max_misclassified_examples,
    )?;

    let metrics = metrics_payload(report, &created_at);
    write_json(&directory.join("metrics.json"), &metrics)?;

    let mut files = BTreeMap::new();
    for name in [
        "config.json",
        "vocabulary.json",
        "labels.json",
        "model.safetensors",
        "metrics.json",
        "confusion_matrix.csv",
        "misclassified_examples.csv",
        "training_history.csv",
    ] {
        files.insert(name.to_string(), sha256_file(&directory.join(name))?);
    }

    let datasets: Vec<Value> = report
        .config
        .sources
        .iter()
        .filter(|source| source.path.is_file())
        .map(|source| {
            json!({
                "name": source.name,
                "path": source.path,
                "sha256": sha256_file(&source.path).unwrap_or_default(),
                "format": source.format.as_str(),
                "role": source.role,
                "text_column": source.text_column,
                "label_column": source.label_column,
                "rows_used": report.records.iter().filter(|record| record.source == source.name).count(),
            })
        })
        .collect();

    let manifest = json!({
        "schema_version": 2,
        "artifact_schema": "linguae.language-detector/2",
        "model_version": report.config.model_version,
        "created_at": created_at,
        "datasets": datasets,
        "dataset_rows": report.records.len(),
        "training": {
            "seed": report.config.seed,
            "train_fraction": report.config.train_fraction,
            "val_fraction": report.config.val_fraction,
            "test_fraction": report.config.test_fraction,
            "max_length": report.config.max_length,
            "embedding_dim": model_config.embedding_dim,
            "num_filters": model_config.num_filters,
            "kernel_sizes": model_config.kernel_sizes,
            "dropout": model_config.dropout,
            "batch_size": report.config.batch_size,
            "epochs_configured": report.config.epochs,
            "epochs_run": report.history.len(),
            "training_seconds": report.train_result.training_seconds,
            "best_epoch": report.train_result.best_epoch,
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss",
            "learning_rate": report.config.learning_rate,
            "weight_decay": report.config.weight_decay,
            "grad_clip_norm": report.config.grad_clip_norm,
            "early_stopping_patience": report.config.early_stopping_patience,
            "device": report.device,
            "min_examples_per_language": report.config.min_examples_per_language,
            "max_rows_per_language_per_source": report.config.max_rows_per_language_per_source,
            "max_words_per_language": report.config.max_words_per_language,
            "max_rows_per_language": report.config.max_rows_per_language,
        },
        "environment": {
            "linguae_train": env!("CARGO_PKG_VERSION"),
            "candle": "0.11.1",
            "platform": std::env::consts::OS,
        },
        "files": files,
        "prediction_scope": "single-language classification per string; mixed-language span detection is not supported",
    });
    write_json(&directory.join("manifest.json"), &manifest)?;

    info!("artifact directory: {}", directory.display());
    for (name, digest) in files {
        info!("  sha256 {digest}  {name}");
    }
    Ok(())
}

fn metrics_payload(report: &ArtifactReport<'_>, created_at: &str) -> Value {
    let evaluation = report.evaluation;
    let metrics = &evaluation.metrics;
    let test = json!({
        "accuracy": metrics.accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "examples": metrics.examples,
        "languages": metrics.languages,
        "by_text_length": evaluation.length_classification.iter().map(|b| json!({"bucket": b.bucket, "examples": b.examples, "accuracy": b.accuracy, "precision": b.precision, "recall": b.recall, "macro_f1": b.macro_f1, "average_confidence": b.average_confidence})).collect::<Vec<_>>(),
    });

    let by_source: BTreeMap<&str, Value> = evaluation
        .by_source
        .iter()
        .map(|(source, summary)| {
            (
                source.as_str(),
                json!({
                    "accuracy": summary.accuracy,
                    "examples": summary.examples,
                    "languages": summary.languages,
                    "predictions": summary.predictions,
                }),
            )
        })
        .collect();

    let per_language: BTreeMap<&str, Value> = evaluation
        .per_language
        .iter()
        .map(|item| {
            (
                item.language.as_str(),
                json!({
                    "precision": item.precision,
                    "recall": item.recall,
                    "f1": item.f1,
                    "support": item.support,
                }),
            )
        })
        .collect();

    let length_buckets: Vec<Value> = evaluation
        .length_buckets
        .iter()
        .map(|bucket| {
            json!({
                "length_bucket": bucket.bucket,
                "examples": bucket.examples,
                "accuracy": bucket.accuracy,
                "share": bucket.share,
            })
        })
        .collect();
    let length_buckets_by_source: Vec<Value> = evaluation
        .length_buckets_by_source
        .iter()
        .map(|bucket| {
            json!({
                "source": bucket.source,
                "length_bucket": bucket.bucket,
                "examples": bucket.examples,
                "accuracy": bucket.accuracy,
            })
        })
        .collect();

    let confidence = &evaluation.confidence;
    let confidence = json!({
        "mean_confidence": confidence.mean_confidence,
        "median_confidence": confidence.median_confidence,
        "share_confidence_at_least_0_9": confidence.share_confidence_at_least_0_9,
        "accuracy_when_confidence_at_least_0_9": confidence.accuracy_when_confidence_at_least_0_9,
        "accuracy_when_confidence_below_0_5": confidence.accuracy_when_confidence_below_0_5,
        "mean_confidence_correct": confidence.mean_confidence_correct,
        "mean_confidence_incorrect": confidence.mean_confidence_incorrect,
    });

    let leakage = &evaluation.leakage;
    let leakage_checks = json!({
        "duplicate_texts_train_vs_validation": leakage.duplicate_texts_train_vs_validation,
        "duplicate_texts_train_vs_test": leakage.duplicate_texts_train_vs_test,
        "duplicate_texts_validation_vs_test": leakage.duplicate_texts_validation_vs_test,
        "evaluation_languages_missing_from_train": leakage.evaluation_languages_missing_from_train,
        "test_examples_per_language_min": leakage.test_examples_per_language_min,
        "test_examples_per_language_max": leakage.test_examples_per_language_max,
        "class_imbalance_ratio": leakage.class_imbalance_ratio,
        "suspiciously_high_accuracy": leakage.suspiciously_high_accuracy,
    });

    let dataset_stats = &report.dataset_stats;
    let source_summary: Vec<Value> = dataset_stats
        .source_counts
        .iter()
        .map(|(source, rows)| {
            json!({
                "source": source,
                "rows": rows,
                "languages": report.records.iter().filter(|record| &record.source == source).map(|record| record.language.as_str()).collect::<std::collections::HashSet<_>>().len(),
            })
        })
        .collect();

    json!({
        "model_version": report.config.model_version,
        "created_at": created_at,
        "datasets": report.config.sources.iter().map(|source| json!({
            "name": source.name,
            "path": source.path,
            "format": source.format.as_str(),
            "role": source.role,
            "rows_used": report.records.iter().filter(|record| record.source == source.name).count(),
        })).collect::<Vec<_>>(),
        "dataset_rows": report.records.len(),
        "split_rows": report.split_details.rows_per_split,
        "split_rows_by_source": report.split_details.rows_per_source_split,
        "max_length": report.config.max_length,
        "vocabulary_size": report.vocabulary.len(),
        "num_classes": report.labels.len(),
        "best_epoch": report.train_result.best_epoch,
        "best_val_loss": report.train_result.best_val_loss,
        "best_val_accuracy": report.train_result.best_val_accuracy,
        "test": test,
        "test_by_source": by_source,
        "per_language": per_language,
        "length_buckets": length_buckets,
        "length_buckets_by_source": length_buckets_by_source,
        "source_summary": source_summary,
        "confidence": confidence,
        "leakage_checks": leakage_checks,
        "length_correlation": evaluation.length_correlation,
        "device": report.device,
        "unknown_character_rate": report.unknown_character_rate,
        "validation": report.validation_report,
        "dataset_stats": {
            "min_examples_per_language": dataset_stats.label_counts.values().copied().min().unwrap_or(0),
            "max_examples_per_language": dataset_stats.label_counts.values().copied().max().unwrap_or(0),
            "median_length": dataset_stats.lengths.median,
            "p90_length": dataset_stats.lengths.p90,
            "p99_length": dataset_stats.lengths.p99,
        },
    })
}

fn write_training_history(path: &Path, history: &[EpochRecord]) -> Result<()> {
    let mut output = String::from("epoch,train_loss,train_accuracy,val_loss,val_accuracy,seconds,improved,samples_per_second,characters_per_second,padding_ratio\n");
    for epoch in history {
        output.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.1},{},{:.2},{:.2},{:.6}\n",
            epoch.epoch,
            epoch.train_loss,
            epoch.train_accuracy,
            epoch.val_loss,
            epoch.val_accuracy,
            epoch.seconds,
            epoch.improved,
            epoch.samples_per_second,
            epoch.characters_per_second,
            epoch.padding_ratio
        ));
    }
    write_text(path, &output)
}

fn write_confusion_matrix(path: &Path, matrix: &ConfusionMatrix, labels: &[String]) -> Result<()> {
    let mut output = String::new();
    output.push(',');
    output.push_str(&labels.join(","));
    output.push('\n');
    for (row, label) in labels.iter().enumerate() {
        output.push_str(label);
        for column in 0..matrix.size {
            output.push(',');
            output.push_str(&matrix.get(row, column).to_string());
        }
        output.push('\n');
    }
    write_text(path, &output)
}

fn write_misclassified(path: &Path, rows: &[MisclassifiedRow], limit: usize) -> Result<()> {
    let mut output = String::from(
        "text,expected,source,dataset_role,predicted,confidence,expected_idx,predicted_idx,char_length\n",
    );
    for row in rows.iter().take(limit) {
        output.push_str(&csv_escape(&row.text));
        output.push(',');
        output.push_str(&csv_escape(&row.expected));
        output.push(',');
        output.push_str(&csv_escape(&row.source));
        output.push(',');
        output.push_str(&csv_escape(&row.role));
        output.push(',');
        output.push_str(&csv_escape(&row.predicted));
        output.push_str(&format!(
            ",{:.6},{},{},{}\n",
            row.confidence, row.expected_idx, row.predicted_idx, row.char_length
        ));
    }
    write_text(path, &output)
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut content = serde_json::to_string_pretty(value).map_err(|source| Error::Json {
        path: path.to_path_buf(),
        source,
    })?;
    content.push('\n');
    write_text(path, &content)
}

fn write_text(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| Error::Write {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let temporary = path.with_file_name(format!(
        "{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact")
    ));
    {
        let mut file = fs::File::create(&temporary).map_err(|source| Error::Write {
            path: temporary.clone(),
            source,
        })?;
        file.write_all(content.as_bytes())
            .map_err(|source| Error::Write {
                path: temporary.clone(),
                source,
            })?;
        file.sync_all().map_err(|source| Error::Write {
            path: temporary.clone(),
            source,
        })?;
    }
    fs::rename(&temporary, path).map_err(|source| Error::Write {
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
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect())
}
