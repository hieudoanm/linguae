use std::path::Path;
use std::time::{Duration, Instant};

use candle_core::backprop::GradStore;
use candle_core::{Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::{loss, ops};
use log::{info, warn};

use crate::config::Config;
use crate::data::{prepare_batch, EncodedSplit};
use crate::error::{Error, Result};
use crate::metrics::TestPrediction;
use crate::model::CharCnn;
use crate::rng::Rng;

#[derive(Debug, Clone)]
pub struct EpochRecord {
    pub epoch: usize,
    pub train_loss: f64,
    pub train_accuracy: f64,
    pub val_loss: f64,
    pub val_accuracy: f64,
    pub seconds: f64,
    pub improved: bool,
    pub samples_per_second: f64,
    pub characters_per_second: f64,
    pub padding_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct TrainResult {
    pub history: Vec<EpochRecord>,
    pub best_epoch: usize,
    pub best_val_loss: f64,
    pub best_val_accuracy: f64,
    pub training_seconds: f64,
}

/// Choose CUDA, Metal or CPU according to the requested preference.
pub fn select_device(preference: &str) -> Result<Device> {
    if preference.eq_ignore_ascii_case("auto") {
        #[cfg(feature = "cuda")]
        if let Ok(device) = Device::new_cuda(0) {
            return Ok(device);
        }
        #[cfg(feature = "metal")]
        if let Ok(device) = Device::new_metal(0) {
            return Ok(device);
        }
        return Ok(Device::Cpu);
    }
    Ok(linguae_core::parse_device(preference)?)
}

pub fn train(
    model: &mut CharCnn,
    train_data: &EncodedSplit,
    validation_data: &EncodedSplit,
    config: &Config,
    checkpoint_path: &Path,
) -> Result<TrainResult> {
    if train_data.is_empty() {
        return Err(Error::Training("training split is empty".into()));
    }
    if validation_data.is_empty() {
        return Err(Error::Training("validation split is empty".into()));
    }

    let mut optimizer = AdamW::new(
        model.vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: config.weight_decay,
        },
    )?;

    let mut rng = Rng::new(config.seed);
    let indices: Vec<usize> = (0..train_data.len()).collect();
    let mut history = Vec::new();
    let mut best_val_loss = f64::INFINITY;
    let mut best_val_accuracy = 0.0;
    let mut best_epoch = 0usize;
    let training_started = Instant::now();

    info!(
        "training plan: epochs={}, batch_size={}, lr={}, weight_decay={}, patience={}",
        config.epochs,
        config.batch_size,
        config.learning_rate,
        config.weight_decay,
        config.early_stopping_patience
    );

    for epoch in 1..=config.epochs {
        let started = Instant::now();
        model.set_training(true);
        let batches = training_batches(train_data, &indices, config, &mut rng);

        let mut total_loss = 0.0f64;
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut valid_characters = 0usize;
        let mut allocated_positions = 0usize;
        let mut total_characters = 0usize;
        let batch_count = batches.len();

        for (batch_index, chunk) in batches.iter().enumerate() {
            let sequences: Vec<&[u32]> = chunk
                .iter()
                .map(|index| train_data.sequences[*index].as_slice())
                .collect();
            let targets: Vec<u32> = chunk
                .iter()
                .map(|index| train_data.labels[*index])
                .collect();
            let batch = prepare_batch(&sequences, &targets, model.device())?;
            let batch_length = batch.input_ids.dims()[1];
            allocated_positions += batch.size * batch_length;
            valid_characters += sequences
                .iter()
                .map(|sequence| sequence.len())
                .sum::<usize>();
            total_characters += sequences
                .iter()
                .map(|sequence| sequence.len())
                .sum::<usize>();
            let logits = model.forward(&batch.input_ids, &batch.mask, &mut rng)?;
            let loss = loss::cross_entropy(&logits, &batch.targets)?;
            let loss_value = loss.to_scalar::<f32>()? as f64;
            if !loss_value.is_finite() {
                return Err(Error::Training(format!(
                    "non-finite training loss at epoch {epoch}, batch {}",
                    batch_index + 1
                )));
            }

            let gradients = loss.backward()?;
            let gradients = clip_gradients(model, gradients, config.grad_clip_norm)?;
            optimizer.step(&gradients)?;

            let predicted = logits.argmax(1)?.to_vec1::<u32>()?;
            correct += predicted
                .iter()
                .zip(targets.iter())
                .filter(|(predicted, expected)| predicted == expected)
                .count();
            total += batch.size;
            total_loss += loss_value * batch.size as f64;

            if (batch_index + 1) % config.progress_every == 0 || batch_index + 1 == batch_count {
                let elapsed = started.elapsed().as_secs_f64().max(1e-9);
                info!(
                    "  batch {}/{} ({:.1}%) | loss {:.4} | accuracy {:.4} | {:.0} samples/s",
                    batch_index + 1,
                    batch_count,
                    100.0 * (batch_index + 1) as f64 / batch_count as f64,
                    total_loss / total as f64,
                    correct as f64 / total as f64,
                    total as f64 / elapsed,
                );
            }
        }

        let train_loss = total_loss / total as f64;
        let train_accuracy = correct as f64 / total as f64;
        model.set_training(false);
        let (val_loss, val_accuracy) = evaluate(model, validation_data, config.batch_size)?;
        let improved = val_loss < best_val_loss;
        if improved {
            best_val_loss = val_loss;
            best_val_accuracy = val_accuracy;
            best_epoch = epoch;
            model.save(checkpoint_path)?;
        }

        let seconds = started.elapsed().as_secs_f64();
        history.push(EpochRecord {
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            seconds,
            improved,
            samples_per_second: total as f64 / seconds.max(1e-9),
            characters_per_second: total_characters as f64 / seconds.max(1e-9),
            padding_ratio: 1.0 - valid_characters as f64 / allocated_positions.max(1) as f64,
        });
        info!(
            "epoch {:>2}/{} | train loss {:.4} acc {:.4} | val loss {:.4} acc {:.4} | {:.1}s {}",
            epoch,
            config.epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            seconds,
            if improved { "*" } else { "" },
        );

        if epoch - best_epoch >= config.early_stopping_patience {
            info!(
                "early stopping at epoch {epoch}: validation loss has not improved for {} epoch(s)",
                config.early_stopping_patience
            );
            break;
        }
    }

    if best_epoch == 0 {
        return Err(Error::Training(
            "training finished without a checkpoint".into(),
        ));
    }
    model.load_weights(checkpoint_path)?;
    info!("restored best checkpoint from epoch {best_epoch}");

    Ok(TrainResult {
        history,
        best_epoch,
        best_val_loss,
        best_val_accuracy,
        training_seconds: training_started.elapsed().as_secs_f64(),
    })
}

pub fn evaluate(model: &mut CharCnn, data: &EncodedSplit, batch_size: usize) -> Result<(f64, f64)> {
    model.set_training(false);
    let mut total_loss = 0.0f64;
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut rng = Rng::new(0);

    for start in (0..data.len()).step_by(batch_size) {
        let end = (start + batch_size).min(data.len());
        let sequences: Vec<&[u32]> = data.sequences[start..end]
            .iter()
            .map(Vec::as_slice)
            .collect();
        let targets = &data.labels[start..end];
        let batch = prepare_batch(&sequences, targets, model.device())?;
        let logits = model.forward(&batch.input_ids, &batch.mask, &mut rng)?;
        let loss = loss::cross_entropy(&logits, &batch.targets)?;
        total_loss += loss.to_scalar::<f32>()? as f64 * batch.size as f64;
        let predicted = logits.argmax(1)?.to_vec1::<u32>()?;
        correct += predicted
            .iter()
            .zip(targets.iter())
            .filter(|(predicted, expected)| predicted == expected)
            .count();
        total += batch.size;
    }

    if total == 0 {
        return Err(Error::Training("evaluation split is empty".into()));
    }
    Ok((total_loss / total as f64, correct as f64 / total as f64))
}

fn training_batches(
    data: &EncodedSplit,
    indices: &[usize],
    config: &Config,
    rng: &mut Rng,
) -> Vec<Vec<usize>> {
    let mut groups = vec![Vec::new(); config.length_bucket_boundaries.len() + 1];
    for &index in indices {
        let length = data.sequences[index].len();
        let bucket = config
            .length_bucket_boundaries
            .iter()
            .position(|bound| length <= *bound)
            .unwrap_or(groups.len() - 1);
        groups[bucket].push(index);
    }
    let mut batches = Vec::new();
    let mut pools = Vec::new();
    for group in &mut groups {
        rng.shuffle(group);
        pools.extend(group.chunks(config.batch_size).map(|chunk| chunk.to_vec()));
    }
    rng.shuffle(&mut pools);
    if !config.length_bucketing {
        let mut all = indices.to_vec();
        rng.shuffle(&mut all);
        return all
            .chunks(config.batch_size)
            .map(<[usize]>::to_vec)
            .collect();
    }
    batches.extend(pools);
    batches
}

pub fn predict(
    model: &CharCnn,
    data: &EncodedSplit,
    batch_size: usize,
) -> Result<Vec<TestPrediction>> {
    let mut predictions = Vec::with_capacity(data.len());
    let mut rng = Rng::new(0);

    for start in (0..data.len()).step_by(batch_size) {
        let end = (start + batch_size).min(data.len());
        let sequences: Vec<&[u32]> = data.sequences[start..end]
            .iter()
            .map(Vec::as_slice)
            .collect();
        let targets = &data.labels[start..end];
        let batch = prepare_batch(&sequences, targets, model.device())?;
        let logits = model.forward(&batch.input_ids, &batch.mask, &mut rng)?;
        let probabilities = ops::softmax(&logits, 1)?.to_vec2::<f32>()?;

        for (row, expected) in probabilities.iter().zip(targets.iter()) {
            let (predicted, confidence) = row
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, score)| (index as u32, *score))
                .unwrap_or((0, 0.0));
            predictions.push(TestPrediction {
                expected: *expected,
                predicted,
                confidence,
            });
        }
    }
    Ok(predictions)
}

pub fn predict_probabilities(
    model: &CharCnn,
    sequences: &[Vec<u32>],
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut result = Vec::with_capacity(sequences.len());
    let mut rng = Rng::new(0);
    for start in (0..sequences.len()).step_by(batch_size) {
        let end = (start + batch_size).min(sequences.len());
        let batch_sequences: Vec<&[u32]> =
            sequences[start..end].iter().map(Vec::as_slice).collect();
        let targets = vec![0u32; batch_sequences.len()];
        let batch = prepare_batch(&batch_sequences, &targets, model.device())?;
        let logits = model.forward(&batch.input_ids, &batch.mask, &mut rng)?;
        result.extend(ops::softmax(&logits, 1)?.to_vec2::<f32>()?);
    }
    Ok(result)
}

fn clip_gradients(model: &CharCnn, mut gradients: GradStore, clip: f64) -> Result<GradStore> {
    if clip <= 0.0 {
        return Ok(gradients);
    }
    let vars = model.vars();
    let mut norm_squared = 0.0f64;
    for var in &vars {
        if let Some(gradient) = gradients.get(var) {
            norm_squared += gradient.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
    }
    let norm = norm_squared.sqrt();
    if norm > clip {
        let scale = clip / norm;
        let mut updates: Vec<(Var, Tensor)> = Vec::new();
        for var in &vars {
            if let Some(gradient) = gradients.get(var) {
                updates.push((var.clone(), (gradient * scale)?));
            }
        }
        for (var, gradient) in updates {
            gradients.insert(&var, gradient);
        }
        warn!("clipped gradients: norm {norm:.3} -> {clip:.3}");
    }
    Ok(gradients)
}

#[allow(dead_code)]
fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else {
        format!("{:.1}min", seconds / 60.0)
    }
}
