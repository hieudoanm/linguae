use std::collections::HashMap;

use candle_core::{Device, Tensor};

use crate::error::Result;
use crate::source::Record;
use crate::vocab::{Vocabulary, PAD_ID};

/// Character-id sequences and integer labels for one split.
#[derive(Debug, Clone, Default)]
pub struct EncodedSplit {
    pub sequences: Vec<Vec<u32>>,
    pub labels: Vec<u32>,
    pub char_lengths: Vec<usize>,
}

impl EncodedSplit {
    pub fn from_records(
        records: &[Record],
        vocabulary: &Vocabulary,
        label_to_idx: &HashMap<String, u32>,
        max_length: usize,
    ) -> Result<Self> {
        let mut split = Self {
            sequences: Vec::with_capacity(records.len()),
            labels: Vec::with_capacity(records.len()),
            char_lengths: Vec::with_capacity(records.len()),
        };

        for record in records {
            let label = *label_to_idx.get(&record.language).ok_or_else(|| {
                crate::Error::Dataset(format!(
                    "label {:?} is missing from the label mapping",
                    record.language
                ))
            })?;
            split
                .sequences
                .push(vocabulary.encode(&record.text, max_length));
            split.labels.push(label);
            split.char_lengths.push(record.text.chars().count());
        }
        Ok(split)
    }

    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

/// A padded batch ready for the model.
pub struct Batch {
    pub input_ids: Tensor,
    pub mask: Tensor,
    pub targets: Tensor,
    pub size: usize,
}

/// Right-pad a batch of sequences and build the padding mask.
pub fn prepare_batch(sequences: &[&[u32]], targets: &[u32], device: &Device) -> Result<Batch> {
    if sequences.is_empty() {
        return Err(crate::Error::Training("empty batch".into()));
    }
    if sequences.len() != targets.len() {
        return Err(crate::Error::Training(format!(
            "batch size mismatch: {} sequences and {} targets",
            sequences.len(),
            targets.len()
        )));
    }

    let length = sequences
        .iter()
        .map(|sequence| sequence.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let batch = sequences.len();
    let mut ids = Vec::with_capacity(batch * length);
    let mut mask: Vec<f32> = Vec::with_capacity(batch * length);

    for sequence in sequences {
        for position in 0..length {
            let id = sequence.get(position).copied().unwrap_or(PAD_ID);
            ids.push(id);
            mask.push(if id == PAD_ID { 0.0 } else { 1.0 });
        }
    }

    Ok(Batch {
        input_ids: Tensor::from_vec(ids, (batch, length), device)?,
        mask: Tensor::from_vec(mask, (batch, 1, length), device)?,
        targets: Tensor::from_vec(targets.to_vec(), (batch,), device)?,
        size: batch,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_sequences_and_builds_a_mask() {
        let sequences: [&[u32]; 2] = [&[2, 3], &[4]];
        let batch = prepare_batch(&sequences, &[0, 1], &Device::Cpu).unwrap();
        assert_eq!(batch.input_ids.dims(), &[2, 2]);
        assert_eq!(batch.mask.dims(), &[2, 1, 2]);
        assert_eq!(
            batch.input_ids.to_vec2::<u32>().unwrap(),
            vec![vec![2, 3], vec![4, 0]]
        );
        assert_eq!(
            batch.mask.to_vec3::<f32>().unwrap(),
            vec![vec![vec![1.0, 1.0]], vec![vec![1.0, 0.0]]]
        );
    }
}
