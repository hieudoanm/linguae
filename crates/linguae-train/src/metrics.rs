use std::collections::BTreeMap;

use crate::source::Record;

#[derive(Debug, Clone, Default)]
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub weighted_f1: f64,
    pub examples: usize,
    pub languages: usize,
}

#[derive(Debug, Clone)]
pub struct PerLanguageMetrics {
    pub language: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    pub size: usize,
    pub counts: Vec<u64>,
}

impl ConfusionMatrix {
    pub fn get(&self, expected: usize, predicted: usize) -> u64 {
        self.counts[expected * self.size + predicted]
    }

    pub fn max(&self) -> u64 {
        self.counts.iter().copied().max().unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct TestPrediction {
    pub expected: u32,
    pub predicted: u32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct LengthBucket {
    pub bucket: String,
    pub examples: usize,
    pub accuracy: f64,
    pub share: f64,
}

#[derive(Debug, Clone)]
pub struct LengthBucketBySource {
    pub source: String,
    pub bucket: String,
    pub examples: usize,
    pub accuracy: f64,
}

#[derive(Debug, Clone)]
pub struct LengthClassificationMetrics {
    pub bucket: String,
    pub examples: usize,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub macro_f1: f64,
    pub average_confidence: f64,
}

pub fn classification_by_length(
    records: &[Record],
    predictions: &[TestPrediction],
    labels: &[String],
) -> Vec<LengthClassificationMetrics> {
    let ranges = [
        ("Single words", 1, 10),
        ("Short text", 11, 25),
        ("Medium text", 26, 50),
        ("Long text", 51, 100),
        ("Paragraphs", 101, 256),
    ];
    ranges
        .iter()
        .map(|(name, low, high)| {
            let selected: Vec<_> = records
                .iter()
                .zip(predictions)
                .filter(|(r, _)| {
                    (r.text.chars().count() >= *low) && (r.text.chars().count() <= *high)
                })
                .collect();
            let subset: Vec<TestPrediction> = selected.iter().map(|(_, p)| (*p).clone()).collect();
            let matrix = confusion_matrix(
                &subset.iter().map(|p| p.expected).collect::<Vec<_>>(),
                &subset.iter().map(|p| p.predicted).collect::<Vec<_>>(),
                labels.len(),
            );
            let (metrics, _) = classification_metrics(&matrix, labels);
            LengthClassificationMetrics {
                bucket: (*name).into(),
                examples: subset.len(),
                accuracy: metrics.accuracy,
                precision: metrics.macro_precision,
                recall: metrics.macro_recall,
                macro_f1: metrics.macro_f1,
                average_confidence: if subset.is_empty() {
                    0.0
                } else {
                    subset.iter().map(|p| p.confidence as f64).sum::<f64>() / subset.len() as f64
                },
            }
        })
        .collect()
}

#[derive(Debug, Clone, Default)]
pub struct ConfidenceSummary {
    pub mean_confidence: f64,
    pub median_confidence: f64,
    pub share_confidence_at_least_0_9: f64,
    pub accuracy_when_confidence_at_least_0_9: Option<f64>,
    pub accuracy_when_confidence_below_0_5: Option<f64>,
    pub mean_confidence_correct: Option<f64>,
    pub mean_confidence_incorrect: Option<f64>,
}

pub const LENGTH_BUCKETS: [&str; 6] = [
    "<100", "100-199", "200-399", "400-799", "800-1599", ">=1600",
];

pub fn confusion_matrix(
    expected: &[u32],
    predicted: &[u32],
    num_classes: usize,
) -> ConfusionMatrix {
    let mut counts = vec![0u64; num_classes * num_classes];
    for (expected, predicted) in expected.iter().zip(predicted.iter()) {
        let expected = *expected as usize;
        let predicted = *predicted as usize;
        if expected < num_classes && predicted < num_classes {
            counts[expected * num_classes + predicted] += 1;
        }
    }
    ConfusionMatrix {
        size: num_classes,
        counts,
    }
}

pub fn classification_metrics(
    matrix: &ConfusionMatrix,
    labels: &[String],
) -> (ClassificationMetrics, Vec<PerLanguageMetrics>) {
    let size = matrix.size;
    let mut per_language = Vec::with_capacity(size);
    let mut present = 0usize;
    let mut macro_precision = 0.0;
    let mut macro_recall = 0.0;
    let mut macro_f1 = 0.0;
    let mut weighted_f1 = 0.0;
    let mut total_support = 0usize;
    let mut correct = 0u64;

    for index in 0..size {
        let true_positive = matrix.get(index, index) as f64;
        let support: u64 = (0..size)
            .map(|predicted| matrix.get(index, predicted))
            .sum();
        let predicted: u64 = (0..size).map(|expected| matrix.get(expected, index)).sum();
        let precision = if predicted > 0 {
            true_positive / predicted as f64
        } else {
            0.0
        };
        let recall = if support > 0 {
            true_positive / support as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        correct += true_positive as u64;
        total_support += support as usize;
        if support > 0 {
            present += 1;
            macro_precision += precision;
            macro_recall += recall;
            macro_f1 += f1;
            weighted_f1 += f1 * support as f64;
        }

        per_language.push(PerLanguageMetrics {
            language: labels
                .get(index)
                .cloned()
                .unwrap_or_else(|| index.to_string()),
            precision,
            recall,
            f1,
            support: support as usize,
        });
    }

    let accuracy = if total_support > 0 {
        correct as f64 / total_support as f64
    } else {
        0.0
    };
    let divisor = present.max(1) as f64;
    let metrics = ClassificationMetrics {
        accuracy,
        macro_precision: macro_precision / divisor,
        macro_recall: macro_recall / divisor,
        macro_f1: macro_f1 / divisor,
        weighted_f1: if total_support > 0 {
            weighted_f1 / total_support as f64
        } else {
            0.0
        },
        examples: total_support,
        languages: present,
    };
    (metrics, per_language)
}

pub fn confidence_summary(predictions: &[TestPrediction]) -> ConfidenceSummary {
    if predictions.is_empty() {
        return ConfidenceSummary::default();
    }
    let confidences: Vec<f64> = predictions
        .iter()
        .map(|prediction| prediction.confidence as f64)
        .collect();
    let mean_confidence = confidences.iter().sum::<f64>() / confidences.len() as f64;
    let high: Vec<&TestPrediction> = predictions
        .iter()
        .filter(|prediction| prediction.confidence >= 0.9)
        .collect();
    let low: Vec<&TestPrediction> = predictions
        .iter()
        .filter(|prediction| prediction.confidence < 0.5)
        .collect();
    let correct: Vec<&TestPrediction> = predictions
        .iter()
        .filter(|prediction| prediction.expected == prediction.predicted)
        .collect();
    let incorrect: Vec<&TestPrediction> = predictions
        .iter()
        .filter(|prediction| prediction.expected != prediction.predicted)
        .collect();

    ConfidenceSummary {
        mean_confidence,
        median_confidence: median(&confidences),
        share_confidence_at_least_0_9: high.len() as f64 / predictions.len() as f64,
        accuracy_when_confidence_at_least_0_9: accuracy_of(&high),
        accuracy_when_confidence_below_0_5: accuracy_of(&low),
        mean_confidence_correct: mean_confidence_of(&correct),
        mean_confidence_incorrect: mean_confidence_of(&incorrect),
    }
}

fn accuracy_of(predictions: &[&TestPrediction]) -> Option<f64> {
    if predictions.is_empty() {
        return None;
    }
    let correct = predictions
        .iter()
        .filter(|prediction| prediction.expected == prediction.predicted)
        .count();
    Some(correct as f64 / predictions.len() as f64)
}

fn mean_confidence_of(predictions: &[&TestPrediction]) -> Option<f64> {
    if predictions.is_empty() {
        return None;
    }
    Some(
        predictions
            .iter()
            .map(|prediction| prediction.confidence as f64)
            .sum::<f64>()
            / predictions.len() as f64,
    )
}

pub fn length_summary(records: &[Record], predictions: &[TestPrediction]) -> Vec<LengthBucket> {
    let total = records.len().max(1);
    let mut buckets: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for (record, prediction) in records.iter().zip(predictions.iter()) {
        let bucket = length_bucket(record.text.chars().count());
        let entry = buckets.entry(bucket).or_insert((0, 0));
        entry.0 += 1;
        if prediction.expected == prediction.predicted {
            entry.1 += 1;
        }
    }
    LENGTH_BUCKETS
        .iter()
        .filter_map(|bucket| {
            buckets.get(bucket).map(|(examples, correct)| LengthBucket {
                bucket: (*bucket).to_string(),
                examples: *examples,
                accuracy: *correct as f64 / *examples as f64,
                share: *examples as f64 / total as f64,
            })
        })
        .collect()
}

pub fn length_summary_by_source(
    records: &[Record],
    predictions: &[TestPrediction],
) -> Vec<LengthBucketBySource> {
    let mut buckets: BTreeMap<(String, &str), (usize, usize)> = BTreeMap::new();
    for (record, prediction) in records.iter().zip(predictions.iter()) {
        let bucket = length_bucket(record.text.chars().count());
        let entry = buckets
            .entry((record.source.clone(), bucket))
            .or_insert((0, 0));
        entry.0 += 1;
        if prediction.expected == prediction.predicted {
            entry.1 += 1;
        }
    }
    buckets
        .into_iter()
        .map(
            |((source, bucket), (examples, correct))| LengthBucketBySource {
                source,
                bucket: bucket.to_string(),
                examples,
                accuracy: correct as f64 / examples as f64,
            },
        )
        .collect()
}

pub fn length_bucket(length: usize) -> &'static str {
    match length {
        0..=99 => "<100",
        100..=199 => "100-199",
        200..=399 => "200-399",
        400..=799 => "400-799",
        800..=1599 => "800-1599",
        _ => ">=1600",
    }
}

pub fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let n = left.len() as f64;
    let left_mean = left.iter().sum::<f64>() / n;
    let right_mean = right.iter().sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (left, right) in left.iter().zip(right.iter()) {
        let left_delta = left - left_mean;
        let right_delta = right - right_mean;
        covariance += left_delta * right_delta;
        left_variance += left_delta * left_delta;
        right_variance += right_delta * right_delta;
    }
    let denominator = (left_variance * right_variance).sqrt();
    if denominator == 0.0 {
        None
    } else {
        Some(covariance / denominator)
    }
}

pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_perfect_metrics() {
        let matrix = confusion_matrix(&[0, 1, 1], &[0, 1, 1], 2);
        let (metrics, per_language) = classification_metrics(&matrix, &["a".into(), "b".into()]);
        assert!((metrics.accuracy - 1.0).abs() < 1e-9);
        assert!((metrics.macro_f1 - 1.0).abs() < 1e-9);
        assert_eq!(per_language[1].support, 2);
    }

    #[test]
    fn buckets_lengths() {
        assert_eq!(length_bucket(99), "<100");
        assert_eq!(length_bucket(100), "100-199");
        assert_eq!(length_bucket(1600), ">=1600");
    }
}
