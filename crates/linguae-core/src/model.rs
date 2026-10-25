use candle_core::{Module, Result, Tensor};
use candle_nn::{conv1d, embedding, linear, Conv1d, Conv1dConfig, Embedding, Linear, VarBuilder};

use crate::config::Config;

/// Character-level CNN matching the exported `CharCNN` module.
pub struct CharCnn {
    embedding: Embedding,
    convolutions: Vec<Conv1d>,
    classifier: Linear,
    length_embedding: Option<Embedding>,
    mean_pooling: bool,
}

impl CharCnn {
    /// Build the model from an exported safetensors or legacy `model.pt` state dict.
    pub fn load(config: &Config, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(config.vocab_size, config.embedding_dim, vb.pp("embedding"))?;

        let mut convolutions = Vec::with_capacity(config.kernel_sizes.len());
        for (index, (filters, kernel_size)) in config
            .num_filters
            .iter()
            .zip(config.kernel_sizes.iter())
            .enumerate()
        {
            let convolution = conv1d(
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
            )?;
            convolutions.push(convolution);
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
            embedding,
            convolutions,
            classifier,
            length_embedding,
            mean_pooling: config.mean_pooling,
        })
    }

    /// Map character IDs `[batch, length]` to logits `[batch, num_classes]`.
    ///
    /// `mask` is a float mask of shape `[batch, 1, length]` where padding
    /// positions are `0` and real positions are `1`. PyTorch uses
    /// `masked_fill(-inf) + max`; because every convolution output passes
    /// through ReLU (so real values are non-negative), zeroing padding before
    /// the max is numerically identical.
    pub fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let embedded = self
            .embedding
            .forward(input_ids)?
            .transpose(1, 2)?
            .broadcast_mul(mask)?;
        let lengths = mask.sum(2)?.squeeze(1)?;
        let mut pooled = Vec::with_capacity(self.convolutions.len());

        for convolution in &self.convolutions {
            let features = convolution.forward(&embedded)?.relu()?;
            let valid = mask.broadcast_as(features.dims())?;
            let masked = features.broadcast_mul(&valid)?;
            pooled.push(masked.max(2)?);
            if self.mean_pooling {
                let sum = masked.sum(2)?;
                let count = valid.sum(2)?.clamp(1.0, f64::INFINITY)?;
                pooled.push(sum.broadcast_div(&count)?);
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
        self.classifier.forward(&representation)
    }
}
