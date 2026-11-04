use candle_core::{Module, Result, Tensor};
use candle_nn::{conv1d, embedding, linear, Conv1d, Conv1dConfig, Embedding, Linear, VarBuilder};

use crate::config::Config;

/// Character-level CNN matching the exported PyTorch `CharCNN` module.
pub struct CharCnn {
    embedding: Embedding,
    convolutions: Vec<Conv1d>,
    classifier: Linear,
}

impl CharCnn {
    /// Build the model from an exported `model.pt` state dict.
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
            config.num_filters.iter().sum(),
            config.num_classes,
            vb.pp("classifier"),
        )?;

        Ok(Self {
            embedding,
            convolutions,
            classifier,
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
        let embedded = self.embedding.forward(input_ids)?.transpose(1, 2)?;
        let mut pooled = Vec::with_capacity(self.convolutions.len());

        for convolution in &self.convolutions {
            let features = convolution.forward(&embedded)?.relu()?;
            let masked = features.broadcast_mul(mask)?;
            pooled.push(masked.max(2)?);
        }

        let representation = Tensor::cat(&pooled, 1)?;
        self.classifier.forward(&representation)
    }
}
