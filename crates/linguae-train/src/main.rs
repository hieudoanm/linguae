use std::path::PathBuf;

use candle_core::Device;
use clap::{Args, Parser, Subcommand};
use log::{info, warn};

use linguae_train::config::Config;
use linguae_train::data::EncodedSplit;
use linguae_train::dataset::{clean_records, dataset_stats, split_records, Split};
use linguae_train::error::{Error, Result};
use linguae_train::export::{
    evaluate, prepare_artifact_dir, write_artifact, write_label_mapping, ArtifactReport,
};
use linguae_train::model::{CharCnn, ModelConfig};
use linguae_train::source::{load_sources, Record};
use linguae_train::train::{predict, predict_probabilities, select_device, train};
use linguae_train::vocab::Vocabulary;

#[derive(Debug, Parser)]
#[command(
    name = "linguae-train",
    version,
    about = "Train, evaluate and export the Linguae language detector"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Train a model and export the artifact.
    Train(TrainArgs),
    /// Convert a legacy PyTorch artifact to safetensors.
    Convert(ConvertArgs),
    /// Load an artifact and print sample predictions.
    Verify(VerifyArgs),
}

#[derive(Debug, Default, Args)]
struct TrainArgs {
    /// Data root containing `csv/wili-2018.csv` and `csv/tatoeba.tsv`.
    #[arg(long)]
    data_dir: Option<PathBuf>,
    /// Directory for the exported artifact.
    #[arg(long)]
    artifact_dir: Option<PathBuf>,
    /// Directory for SVG plots.
    #[arg(long)]
    plots_dir: Option<PathBuf>,
    /// Device preference: auto, cpu, cuda or metal.
    #[arg(long)]
    device: Option<String>,
    /// Maximum number of epochs.
    #[arg(long)]
    epochs: Option<usize>,
    /// Mini-batch size.
    #[arg(long)]
    batch_size: Option<usize>,
    /// Maximum characters per example.
    #[arg(long)]
    max_length: Option<usize>,
    /// Random seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Skip plot generation.
    #[arg(long)]
    no_plots: bool,
}

#[derive(Debug, Args)]
struct ConvertArgs {
    /// Legacy artifact directory containing `model.pt`.
    #[arg(long)]
    source: PathBuf,
    /// Destination directory for the safetensors artifact.
    #[arg(long)]
    destination: PathBuf,
    /// Replace a non-empty destination directory.
    #[arg(long)]
    overwrite: bool,
}

#[derive(Debug, Args)]
struct VerifyArgs {
    /// Artifact directory. Defaults to repository discovery.
    #[arg(long)]
    artifact: Option<PathBuf>,
    /// Device preference: cpu, cuda or metal.
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Strings to classify. Defaults to a small multilingual sample.
    texts: Vec<String>,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    match Cli::parse()
        .command
        .unwrap_or(Command::Train(TrainArgs::default()))
    {
        Command::Train(args) => run_train(args),
        Command::Convert(args) => {
            linguae_train::convert::convert_artifact(
                &args.source,
                &args.destination,
                args.overwrite,
            )?;
            info!("conversion complete: {}", args.destination.display());
            Ok(())
        }
        Command::Verify(args) => {
            let directory = linguae_core::resolve(args.artifact.as_deref())?;
            linguae_train::verify::verify_artifact(&directory, &args.device, &args.texts)
        }
    }
}

fn run_train(args: TrainArgs) -> Result<()> {
    let mut config = Config::from_env()?;
    if let Some(data_dir) = &args.data_dir {
        config.set_sources(Some(data_dir), None)?;
    }
    if let Some(artifact_dir) = args.artifact_dir {
        config.artifact_dir = artifact_dir;
    }
    if let Some(plots_dir) = args.plots_dir {
        config.plots_dir = plots_dir;
    }
    if let Some(device) = args.device {
        config.device = device;
    }
    if let Some(epochs) = args.epochs {
        config.epochs = epochs;
    }
    if let Some(batch_size) = args.batch_size {
        config.batch_size = batch_size;
    }
    if let Some(max_length) = args.max_length {
        config.max_length = max_length;
    }
    if let Some(seed) = args.seed {
        config.seed = seed;
    }
    if args.no_plots {
        config.plots = false;
    }
    config.validate()?;

    info!("configuration:\n{config:#?}");

    let (records, invalid, validation_report) = load_sources(&config.sources)?;
    info!(
        "loaded {} valid rows and {} invalid rows",
        records.len(),
        invalid.len()
    );
    let records = clean_records(records, &config)?;
    let dataset_stats = dataset_stats(&records);
    let (records, split_details) = split_records(records, &config)?;

    let train_records = records_for(&records, Split::Train);
    let validation_records = records_for(&records, Split::Validation);
    let test_records = records_for(&records, Split::Test);

    let vocabulary = Vocabulary::from_texts(train_records.iter().map(|row| row.text.as_str()));
    info!(
        "vocabulary: {} ids ({} characters)",
        vocabulary.len(),
        vocabulary.len().saturating_sub(2)
    );

    let mut unknown_character_rate = std::collections::BTreeMap::new();
    for (name, split) in [
        ("train", &train_records),
        ("validation", &validation_records),
        ("test", &test_records),
    ] {
        unknown_character_rate.insert(name.to_string(), vocabulary.unknown_character_rate(split));
    }
    info!("unknown character rate: {unknown_character_rate:?}");

    let labels: Vec<String> = records
        .iter()
        .map(|record| record.language.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let label_to_idx: std::collections::HashMap<String, u32> = labels
        .iter()
        .enumerate()
        .map(|(index, label)| (label.clone(), index as u32))
        .collect();
    write_label_mapping(&config.processed_dir, &labels)?;

    let train_data = EncodedSplit::from_records(
        &train_records,
        &vocabulary,
        &label_to_idx,
        config.max_length,
    )?;
    let validation_data = EncodedSplit::from_records(
        &validation_records,
        &vocabulary,
        &label_to_idx,
        config.max_length,
    )?;
    let test_data =
        EncodedSplit::from_records(&test_records, &vocabulary, &label_to_idx, config.max_length)?;
    for (name, split) in [
        ("train", &train_data),
        ("validation", &validation_data),
        ("test", &test_data),
    ] {
        let truncated = split
            .char_lengths
            .iter()
            .filter(|length| **length > config.max_length)
            .count();
        if truncated > 0 {
            info!(
                "{name} split: {truncated} sequences exceed max length {}; they are truncated",
                config.max_length
            );
        }
    }

    let model_config = ModelConfig {
        vocab_size: vocabulary.len(),
        num_classes: labels.len(),
        embedding_dim: config.embedding_dim,
        num_filters: config.num_filters.clone(),
        kernel_sizes: config.kernel_sizes.clone(),
        dropout: config.dropout,
        length_embedding: config.length_embedding,
        length_embedding_dim: 8,
        mean_pooling: config.mean_pooling,
    };

    let device = select_device(&config.device)?;
    let device_label = device_label(&device);
    info!("device: {device_label}");
    let mut model = CharCnn::new(&model_config, &device)?;

    let artifact_dir = prepare_artifact_dir(&config)?;
    let checkpoint = artifact_dir.join("checkpoint.safetensors");
    let train_result = train(
        &mut model,
        &train_data,
        &validation_data,
        &config,
        &checkpoint,
    )?;

    let predictions = predict(&model, &test_data, config.batch_size)?;
    let evaluation = evaluate(&test_records, &records, &predictions, &labels)?;
    if evaluation.leakage.leaked() > 0 {
        return Err(Error::Dataset(format!(
            "text leakage detected between splits: {:?}",
            evaluation.leakage
        )));
    }
    if evaluation.leakage.evaluation_languages_missing_from_train > 0 {
        return Err(Error::Dataset(
            "some evaluation languages never appear in training".into(),
        ));
    }
    if evaluation.leakage.suspiciously_high_accuracy {
        warn!("near-perfect accuracy is unusual; check for duplicated documents or label leakage");
    }

    let report = ArtifactReport {
        config: &config,
        model_config: &model_config,
        model: &model,
        vocabulary: &vocabulary,
        labels: &labels,
        records: &records,
        validation_report: &validation_report,
        dataset_stats: &dataset_stats,
        split_details: &split_details,
        history: &train_result.history,
        train_result: &train_result,
        evaluation: &evaluation,
        device: &device_label,
        unknown_character_rate,
    };

    linguae_train::plots::write_all(
        &config,
        &records,
        &train_result.history,
        &evaluation,
        train_result.best_epoch,
        &labels,
    )?;
    write_artifact(&report)?;
    verify_export(
        &model,
        &model_config,
        &device,
        &test_records,
        &vocabulary,
        &config,
    )?;

    let _ = std::fs::remove_file(&checkpoint);
    info!(
        "training complete: best epoch {}, test accuracy {:.4}, artifact {}",
        train_result.best_epoch,
        evaluation.metrics.accuracy,
        artifact_dir.display()
    );
    Ok(())
}

fn records_for(records: &[Record], split: Split) -> Vec<Record> {
    records
        .iter()
        .filter(|record| record.split == split)
        .cloned()
        .collect()
}

fn device_label(device: &Device) -> String {
    match device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(_) => "cuda".to_string(),
        Device::Metal(_) => "metal".to_string(),
    }
}

fn verify_export(
    model: &CharCnn,
    model_config: &ModelConfig,
    device: &Device,
    test_records: &[Record],
    vocabulary: &Vocabulary,
    config: &Config,
) -> Result<()> {
    let path = config.artifact_dir.join("model.safetensors");
    let reloaded = CharCnn::load(model_config, &path, device)?;

    let mut texts: Vec<String> = test_records
        .iter()
        .take(3)
        .map(|record| record.text.clone())
        .collect();
    texts.push("Bonjour tout le monde".into());
    texts.push("The quick brown fox jumps over the lazy dog.".into());
    let sequences: Vec<Vec<u32>> = texts
        .iter()
        .map(|text| vocabulary.encode(text, config.max_length))
        .collect();

    let reference = predict_probabilities(model, &sequences, config.batch_size)?;
    let reloaded_probabilities = predict_probabilities(&reloaded, &sequences, config.batch_size)?;

    let mut maximum_difference = 0.0f32;
    for (left, right) in reference.iter().zip(reloaded_probabilities.iter()) {
        for (left, right) in left.iter().zip(right.iter()) {
            maximum_difference = maximum_difference.max((left - right).abs());
        }
        let left_index = argmax(left);
        let right_index = argmax(right);
        if left_index != right_index {
            return Err(Error::Training(
                "artifact predictions differ from the trained model".into(),
            ));
        }
    }
    if maximum_difference > 1e-5 {
        return Err(Error::Training(format!(
            "artifact probabilities differ by {maximum_difference:.2e}"
        )));
    }
    info!(
        "verification: {} fixed strings, max probability difference {maximum_difference:.2e}",
        texts.len()
    );
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}
