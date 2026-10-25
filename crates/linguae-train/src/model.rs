use std::path::Path;

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor, Var};
use candle_nn::{
    conv1d, embedding, linear, Conv1d, Conv1dConfig, Embedding, Linear, VarBuilder, VarMap,
};

use crate::rng::Rng;

/// Architecture parameters shared by training and inference.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub num_classes: usize,
    pub embedding_dim: usize,
    pub num_filters: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    pub dropout: f32,
    pub length_embedding: bool,
    pub length_embedding_dim: usize,
    pub mean_pooling: bool,
}

/// Character-level CNN matching the exported `CharCNN` module.
pub struct CharCnn {
    varmap: VarMap,
    embedding: Embedding,
    convolutions: Vec<Conv1d>,
    classifier: Linear,
    length_embedding: Option<Embedding>,
    mean_pooling: bool,
    dropout: f32,
    training: bool,
    device: Device,
}

impl CharCnn {
    pub fn new(config: &ModelConfig, device: &Device) -> CandleResult<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let embedding = embedding(config.vocab_size, config.embedding_dim, vb.pp("embedding"))?;

        let mut convolutions = Vec::with_capacity(config.kernel_sizes.len());
        for (index, (filters, kernel_size)) in config
            .num_filters
            .iter()
            .zip(config.kernel_sizes.iter())
            .enumerate()
        {
            convolutions.push(conv1d(
                config.embedding_dim,
                *filters,
                *kernel_size,
                Conv1dConfig {
                    padding: kernel_size / 2,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    ..Default::default()
                },
                vb.pp("convolutions").pp(index),
            )?);
        }

        let classifier = linear(
            config.num_filters.iter().sum::<usize>() * if config.mean_pooling { 2 } else { 1 }
                + if config.length_embedding {
                    config.length_embedding_dim
                } else {
                    0
                },
            config.num_classes,
            vb.pp("classifier"),
        )?;
        let length_embedding = if config.length_embedding {
            Some(candle_nn::embedding(
                6,
                config.length_embedding_dim,
                vb.pp("length_embedding"),
            )?)
        } else {
            None
        };

        Ok(Self {
            varmap,
            embedding,
            convolutions,
            classifier,
            length_embedding,
            mean_pooling: config.mean_pooling,
            dropout: config.dropout,
            training: false,
            device: device.clone(),
        })
    }

    /// Build the model and load weights from a safetensors checkpoint.
    pub fn load(config: &ModelConfig, path: &Path, device: &Device) -> CandleResult<Self> {
        let mut model = Self::new(config, device)?;
        model.varmap.load(path)?;
        Ok(model)
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn vars(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }

    pub fn save(&self, path: &Path) -> CandleResult<()> {
        self.varmap.save(path)
    }

    pub fn load_weights(&mut self, path: &Path) -> CandleResult<()> {
        self.varmap.load(path)
    }

    /// Map character IDs `[batch, length]` to logits `[batch, num_classes]`.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        mask: &Tensor,
        rng: &mut Rng,
    ) -> CandleResult<Tensor> {
        let embedded = self.embedding.forward(input_ids)?.transpose(1, 2)?;
        let embedded = embedded.broadcast_mul(mask)?;
        let lengths = mask.sum(2)?.squeeze(1)?;
        let mut pooled = Vec::with_capacity(self.convolutions.len());

        for convolution in &self.convolutions {
            let features = convolution.forward(&embedded)?.relu()?;
            // Output positions are valid only at original character positions. This
            // prevents convolution responses beyond a sequence's right edge pooling.
            let valid = mask.broadcast_as(features.dims())?;
            let masked = features.broadcast_mul(&valid)?;
            let max = masked.max(2)?;
            let sum = masked.sum(2)?;
            let count = valid.sum(2)?.clamp(1.0, f64::INFINITY)?;
            let mean = sum.broadcast_div(&count)?;
            pooled.push(max);
            if self.mean_pooling {
                pooled.push(mean);
            }
        }

        if let Some(length_embedding) = &self.length_embedding {
            let categories = lengths
                .to_vec1::<f32>()?
                .into_iter()
                .map(|length| match length as usize {
                    0..=10 => 0u32,
                    11..=25 => 1,
                    26..=50 => 2,
                    51..=100 => 3,
                    101..=160 => 4,
                    _ => 5,
                })
                .collect::<Vec<_>>();
            pooled.push(length_embedding.forward(&Tensor::from_vec(
                categories,
                lengths.dims(),
                lengths.device(),
            )?)?);
        }
        let representation = Tensor::cat(&pooled, 1)?;
        let representation = self.apply_dropout(&representation, rng)?;
        self.classifier.forward(&representation)
    }

    fn apply_dropout(&self, values: &Tensor, rng: &mut Rng) -> CandleResult<Tensor> {
        if !self.training || self.dropout <= 0.0 {
            return Ok(values.clone());
        }
        let keep_scale = 1.0 / (1.0 - self.dropout as f64);
        let mask: Vec<f32> = (0..values.elem_count())
            .map(|_| {
                if rng.next_f32() < self.dropout {
                    0.0
                } else {
                    keep_scale as f32
                }
            })
            .collect();
        let mask = Tensor::from_vec(mask, values.dims().to_vec(), values.device())?;
        values.mul(&mask)
    }
}
