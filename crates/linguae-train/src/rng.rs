//! Small deterministic random number generator used for splits and dropout.
//!
//! The pipeline must be reproducible without pulling in a large RNG
//! dependency. SplitMix64 is more than adequate for shuffling indices and
//! generating dropout masks.

#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Uniform float in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform float in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    /// In-place Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            let swap = (self.next_u64() % (index as u64 + 1)) as usize;
            items.swap(index, swap);
        }
    }

    /// Pick `count` distinct indices from `0..len`, returned in ascending order.
    pub fn choose_indices(&mut self, len: usize, count: usize) -> Vec<usize> {
        if count >= len {
            return (0..len).collect();
        }
        let mut indices: Vec<usize> = (0..len).collect();
        self.shuffle(&mut indices);
        indices.truncate(count);
        indices.sort_unstable();
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_is_deterministic() {
        let mut left = vec![0, 1, 2, 3, 4, 5];
        let mut right = left.clone();
        Rng::new(42).shuffle(&mut left);
        Rng::new(42).shuffle(&mut right);
        assert_eq!(left, right);
        assert_ne!(left, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn choose_indices_returns_sorted_unique_values() {
        let mut rng = Rng::new(7);
        let chosen = rng.choose_indices(10, 4);
        assert_eq!(chosen.len(), 4);
        assert!(chosen.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(chosen.iter().all(|index| *index < 10));
    }
}
