use std::collections::BTreeMap;
use std::path::Path;

use plotters::prelude::*;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::export::Evaluation;
use crate::metrics::{LengthBucketBySource, LENGTH_BUCKETS};
use crate::source::Record;
use crate::train::EpochRecord;

const PALETTE: [RGBColor; 8] = [
    RGBColor(38, 70, 83),
    RGBColor(42, 157, 143),
    RGBColor(233, 196, 106),
    RGBColor(244, 162, 97),
    RGBColor(231, 111, 81),
    RGBColor(69, 123, 157),
    RGBColor(168, 218, 220),
    RGBColor(141, 153, 174),
];

/// Write every core plot as an SVG, matching the Python pipeline's figures.
pub fn write_all(
    config: &Config,
    records: &[Record],
    history: &[EpochRecord],
    evaluation: &Evaluation,
    best_epoch: usize,
    labels: &[String],
) -> Result<()> {
    if !config.plots {
        return Ok(());
    }
    std::fs::create_dir_all(&config.plots_dir).map_err(|source| Error::Write {
        path: config.plots_dir.clone(),
        source,
    })?;

    dataset_quality(&config.plots_dir.join("dataset_quality.svg"), records)?;
    training_curves(
        &config.plots_dir.join("training_curves.svg"),
        history,
        best_epoch,
    )?;
    confusion_matrix_plot(
        &config.plots_dir.join("confusion_matrix.svg"),
        &evaluation.confusion,
        labels,
    )?;
    accuracy_by_length(
        &config.plots_dir.join("accuracy_by_length.svg"),
        &evaluation.length_buckets_by_source,
    )?;
    Ok(())
}

fn dataset_quality(path: &Path, records: &[Record]) -> Result<()> {
    let root = SVGBackend::new(path, (1800, 520)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|error| Error::Plot(error.to_string()))?;
    let (left, rest) = root.split_horizontally(600);

    let mut lengths: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    let mut label_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut source_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for record in records {
        let length = record.text.chars().count().min(1500);
        lengths
            .entry(record.source.as_str())
            .or_default()
            .push(length);
        *label_counts.entry(record.language.as_str()).or_insert(0) += 1;
        *source_counts.entry(record.source.as_str()).or_insert(0) += 1;
    }

    // Character length histogram by source (clipped at 1500 characters).
    let bins = 60usize;
    let mut histograms: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (source, values) in &lengths {
        let mut histogram = vec![0usize; bins];
        for value in values {
            let index = ((*value as f64 / 1500.0) * bins as f64).floor() as usize;
            histogram[index.min(bins - 1)] += 1;
        }
        histograms.insert(source, histogram);
    }
    let max_count = histograms
        .values()
        .flat_map(|histogram| histogram.iter().copied())
        .max()
        .unwrap_or(1)
        .max(1);
    let mut chart = ChartBuilder::on(&left)
        .caption(
            "Character length by source (clipped at 1500)",
            ("sans-serif", 20),
        )
        .margin(16)
        .x_label_area_size(36)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..1500f64, 0f64..max_count as f64)
        .map_err(|error| Error::Plot(error.to_string()))?;
    chart
        .configure_mesh()
        .x_desc("characters")
        .y_desc("examples")
        .draw()
        .map_err(|error| Error::Plot(error.to_string()))?;
    for (index, (source, histogram)) in histograms.iter().enumerate() {
        let color = PALETTE[index % PALETTE.len()];
        let points: Vec<(f64, f64)> = histogram
            .iter()
            .enumerate()
            .map(|(bin, count)| ((bin as f64 + 0.5) * (1500.0 / bins as f64), *count as f64))
            .collect();
        chart
            .draw_series(LineSeries::new(points, color.stroke_width(2)))
            .map_err(|error| Error::Plot(error.to_string()))?
            .label(*source)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|error| Error::Plot(error.to_string()))?;

    // Examples per language, sorted descending.
    let (middle, right) = rest.split_horizontally(600);
    let mut counts: Vec<usize> = label_counts.values().copied().collect();
    counts.sort_unstable_by(|left, right| right.cmp(left));
    let max_label = counts.first().copied().unwrap_or(1).max(1);
    let mut chart = ChartBuilder::on(&middle)
        .caption("Examples per language (sorted)", ("sans-serif", 20))
        .margin(16)
        .x_label_area_size(36)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..counts.len().max(1) as f64, 0f64..max_label as f64)
        .map_err(|error| Error::Plot(error.to_string()))?;
    chart
        .configure_mesh()
        .x_desc("language rank")
        .y_desc("examples")
        .draw()
        .map_err(|error| Error::Plot(error.to_string()))?;
    chart
        .draw_series(counts.iter().enumerate().map(|(index, count)| {
            Rectangle::new(
                [(index as f64, 0.0), (index as f64 + 1.0, *count as f64)],
                RGBColor(255, 140, 0).filled(),
            )
        }))
        .map_err(|error| Error::Plot(error.to_string()))?;

    // Rows by dataset source.
    let mut sources: Vec<(&str, usize)> = source_counts.into_iter().collect();
    sources.sort_by_key(|source| std::cmp::Reverse(source.1));
    let max_source = sources
        .iter()
        .map(|(_, count)| *count)
        .max()
        .unwrap_or(1)
        .max(1);
    let mut chart = ChartBuilder::on(&right)
        .caption("Rows by dataset source", ("sans-serif", 20))
        .margin(16)
        .x_label_area_size(48)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..sources.len().max(1) as f64, 0f64..max_source as f64)
        .map_err(|error| Error::Plot(error.to_string()))?;
    chart
        .configure_mesh()
        .x_desc("source")
        .y_desc("examples")
        .x_labels(sources.len())
        .x_label_formatter(&|value| {
            sources
                .get(value.round() as usize)
                .map(|(source, _)| (*source).to_string())
                .unwrap_or_default()
        })
        .draw()
        .map_err(|error| Error::Plot(error.to_string()))?;
    chart
        .draw_series(sources.iter().enumerate().map(|(index, (_, count))| {
            Rectangle::new(
                [
                    (index as f64 + 0.1, 0.0),
                    (index as f64 + 0.9, *count as f64),
                ],
                RGBColor(46, 139, 87).filled(),
            )
        }))
        .map_err(|error| Error::Plot(error.to_string()))?;

    root.present()
        .map_err(|error| Error::Plot(error.to_string()))?;
    Ok(())
}

fn training_curves(path: &Path, history: &[EpochRecord], best_epoch: usize) -> Result<()> {
    let root = SVGBackend::new(path, (1300, 480)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|error| Error::Plot(error.to_string()))?;
    let (left, right) = root.split_horizontally(650);

    let max_epoch = history.iter().map(|row| row.epoch).max().unwrap_or(1) as f64;
    let max_loss = history
        .iter()
        .flat_map(|row| [row.train_loss, row.val_loss])
        .fold(0.0f64, f64::max)
        .max(1e-6);

    let mut chart = ChartBuilder::on(&left)
        .caption("Cross-entropy loss", ("sans-serif", 20))
        .margin(16)
        .x_label_area_size(36)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..max_epoch.max(1.0), 0f64..max_loss)
        .map_err(|error| Error::Plot(error.to_string()))?;
    chart
        .configure_mesh()
        .x_desc("epoch")
        .y_desc("loss")
        .draw()
        .map_err(|error| Error::Plot(error.to_string()))?;
    draw_curve(
        &mut chart,
        history,
        |row| row.train_loss,
        PALETTE[1],
        "train",
    )?;
    draw_curve(
        &mut chart,
        history,
        |row| row.val_loss,
        PALETTE[4],
        "validation",
    )?;
    draw_best_epoch(&mut chart, best_epoch, max_loss)?;
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|error| Error::Plot(error.to_string()))?;

    let mut chart = ChartBuilder::on(&right)
        .caption("Accuracy", ("sans-serif", 20))
        .margin(16)
        .x_label_area_size(36)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..max_epoch.max(1.0), 0f64..1.0f64)
        .map_err(|error| Error::Plot(error.to_string()))?;
    chart
        .configure_mesh()
        .x_desc("epoch")
        .y_desc("accuracy")
        .draw()
        .map_err(|error| Error::Plot(error.to_string()))?;
    draw_curve(
        &mut chart,
        history,
        |row| row.train_accuracy,
        PALETTE[1],
        "train",
    )?;
    draw_curve(
        &mut chart,
        history,
        |row| row.val_accuracy,
        PALETTE[4],
        "validation",
    )?;
    draw_best_epoch(&mut chart, best_epoch, 1.0)?;
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|error| Error::Plot(error.to_string()))?;

    root.present()
        .map_err(|error| Error::Plot(error.to_string()))?;
    Ok(())
}

fn draw_curve<DB, X, Y>(
    chart: &mut ChartContext<'_, DB, Cartesian2d<X, Y>>,
    history: &[EpochRecord],
    value: impl Fn(&EpochRecord) -> f64,
    color: RGBColor,
    label: &str,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
    X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
{
    let points: Vec<(f64, f64)> = history
        .iter()
        .map(|row| (row.epoch as f64, value(row)))
        .collect();
    chart
        .draw_series(LineSeries::new(points, color.stroke_width(2)))
        .map_err(|error| Error::Plot(error.to_string()))?
        .label(label)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    Ok(())
}

fn draw_best_epoch<DB, X, Y>(
    chart: &mut ChartContext<'_, DB, Cartesian2d<X, Y>>,
    best_epoch: usize,
    height: f64,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
    X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
{
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(best_epoch as f64, 0.0), (best_epoch as f64, height)],
            RGBColor(128, 128, 128).stroke_width(1),
        )))
        .map_err(|error| Error::Plot(error.to_string()))?;
    Ok(())
}

fn confusion_matrix_plot(
    path: &Path,
    matrix: &crate::metrics::ConfusionMatrix,
    labels: &[String],
) -> Result<()> {
    let size = matrix.size.max(1);
    let root = SVGBackend::new(path, (1400, 1400)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|error| Error::Plot(error.to_string()))?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Confusion matrix — test set (row = expected, column = predicted)",
            ("sans-serif", 18),
        )
        .margin(24)
        .x_label_area_size(70)
        .y_label_area_size(90)
        .build_cartesian_2d(0f64..size as f64, 0f64..size as f64)
        .map_err(|error| Error::Plot(error.to_string()))?;

    let label_step = (size / 24).max(1);
    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_desc("predicted language")
        .y_desc("expected language")
        .x_labels(24)
        .y_labels(24)
        .x_label_formatter(&|value| {
            let index = value.round() as usize;
            if index % label_step == 0 {
                labels.get(index).cloned().unwrap_or_default()
            } else {
                String::new()
            }
        })
        .y_label_formatter(&|value| {
            let index = value.round() as usize;
            if index % label_step == 0 {
                labels.get(index).cloned().unwrap_or_default()
            } else {
                String::new()
            }
        })
        .draw()
        .map_err(|error| Error::Plot(error.to_string()))?;

    let max = matrix.max().max(1) as f64;
    let mut cells = Vec::new();
    for expected in 0..size {
        for predicted in 0..size {
            let count = matrix.get(expected, predicted);
            if count == 0 {
                continue;
            }
            let intensity = (1.0 + count as f64).ln() / (1.0 + max).ln();
            let blue = (40.0 + 215.0 * intensity) as u8;
            let red = (255.0 * (1.0 - intensity)) as u8;
            let green = (255.0 * (1.0 - intensity)) as u8;
            cells.push(Rectangle::new(
                [
                    (predicted as f64, expected as f64),
                    (predicted as f64 + 1.0, expected as f64 + 1.0),
                ],
                RGBColor(red, green, blue).filled(),
            ));
        }
    }
    chart
        .draw_series(cells)
        .map_err(|error| Error::Plot(error.to_string()))?;
    root.present()
        .map_err(|error| Error::Plot(error.to_string()))?;
    Ok(())
}

fn accuracy_by_length(path: &Path, rows: &[LengthBucketBySource]) -> Result<()> {
    let root = SVGBackend::new(path, (1000, 480)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|error| Error::Plot(error.to_string()))?;

    let sources: Vec<&str> = rows
        .iter()
        .map(|row| row.source.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Test accuracy by input length and source",
            ("sans-serif", 20),
        )
        .margin(16)
        .x_label_area_size(56)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..LENGTH_BUCKETS.len() as f64, 0f64..1.02f64)
        .map_err(|error| Error::Plot(error.to_string()))?;
    chart
        .configure_mesh()
        .x_desc("characters")
        .y_desc("accuracy")
        .x_labels(LENGTH_BUCKETS.len())
        .x_label_formatter(&|value| {
            LENGTH_BUCKETS
                .get(value.round() as usize)
                .map(|label| (*label).to_string())
                .unwrap_or_default()
        })
        .draw()
        .map_err(|error| Error::Plot(error.to_string()))?;

    let width = 0.8 / sources.len().max(1) as f64;
    for (offset, source) in sources.iter().enumerate() {
        let color = PALETTE[offset % PALETTE.len()];
        let series: Vec<(f64, f64)> = rows
            .iter()
            .filter(|row| row.source == *source)
            .filter_map(|row| {
                LENGTH_BUCKETS
                    .iter()
                    .position(|bucket| *bucket == row.bucket)
                    .map(|index| {
                        (
                            index as f64
                                + (offset as f64 - (sources.len() as f64 - 1.0) / 2.0) * width,
                            row.accuracy,
                        )
                    })
            })
            .collect();
        chart
            .draw_series(
                series
                    .into_iter()
                    .map(|(x, y)| Rectangle::new([(x, 0.0), (x + width, y)], color.filled())),
            )
            .map_err(|error| Error::Plot(error.to_string()))?
            .label(*source)
            .legend(move |(x, y)| Rectangle::new([(x, y - 4), (x + 20, y + 4)], color.filled()));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|error| Error::Plot(error.to_string()))?;

    root.present()
        .map_err(|error| Error::Plot(error.to_string()))?;
    Ok(())
}
