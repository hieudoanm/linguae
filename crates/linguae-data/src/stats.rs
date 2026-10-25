use std::collections::BTreeMap;

/// Minimum, median and maximum of a stream of lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthStats {
    pub min: usize,
    pub median: usize,
    pub max: usize,
}

/// Counts lengths in a compact histogram so large datasets can report
/// percentiles without keeping every value in memory.
#[derive(Debug, Default, Clone)]
pub struct LengthHistogram {
    counts: BTreeMap<usize, u64>,
    total: u64,
}

impl LengthHistogram {
    pub fn push(&mut self, length: usize) {
        *self.counts.entry(length).or_default() += 1;
        self.total += 1;
    }

    pub fn finish(&self) -> Option<LengthStats> {
        if self.total == 0 {
            return None;
        }

        let min = *self.counts.keys().next().expect("a non-empty histogram");
        let max = *self
            .counts
            .keys()
            .next_back()
            .expect("a non-empty histogram");
        let low_rank = self.total.div_ceil(2);
        let high_rank = (self.total + 2) / 2;

        let mut cumulative = 0u64;
        let mut low = None;
        let mut high = None;
        for (length, count) in &self.counts {
            cumulative += count;
            if low.is_none() && low_rank <= cumulative {
                low = Some(*length);
            }
            if high_rank <= cumulative {
                high = Some(*length);
                break;
            }
        }

        let low = low.expect("the median rank is covered by the histogram");
        let high = high.expect("the median rank is covered by the histogram");
        let median = ((low + high) as f64 / 2.0).round() as usize;
        Some(LengthStats { min, median, max })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(lengths: &[usize]) -> LengthStats {
        let mut histogram = LengthHistogram::default();
        for length in lengths {
            histogram.push(*length);
        }
        histogram.finish().expect("non-empty")
    }

    #[test]
    fn reports_min_median_and_max() {
        assert_eq!(
            stats(&[3, 1, 2]),
            LengthStats {
                min: 1,
                median: 2,
                max: 3
            }
        );
    }

    #[test]
    fn averages_the_two_middle_values_for_even_counts() {
        assert_eq!(stats(&[1, 2, 3, 10]).median, 3);
    }

    #[test]
    fn returns_none_for_an_empty_histogram() {
        assert!(LengthHistogram::default().finish().is_none());
    }
}
