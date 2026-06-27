//! Maximum entropy partitioning
//!
//! Implements maximum entropy partitioning for optimal symbolic encoding.

use crate::Result;
use super::types::Partition;

/// Maximum entropy partitioner
#[derive(Debug, Clone)]
pub struct MaximumEntropyPartitioner {
    /// Number of symbols (partition regions)
    num_symbols: usize,
}

impl MaximumEntropyPartitioner {
    /// Create a new maximum entropy partitioner
    pub fn new(num_symbols: usize) -> Result<Self> {
        if num_symbols < 2 {
            return Err(crate::SdfError::InvalidParameter(
                "Number of symbols must be at least 2".to_string(),
            ));
        }

        Ok(MaximumEntropyPartitioner { num_symbols })
    }

    /// Compute partition boundaries from signal data
    ///
    /// Uses information theory to determine optimal partition boundaries
    /// that maximize entropy.
    pub fn compute_partition(&self, signal: &[f64]) -> Result<Partition> {
        if signal.is_empty() {
            return Err(crate::SdfError::InvalidParameter(
                "Signal cannot be empty".to_string(),
            ));
        }

        // Find min and max of signal
        let min = signal
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = signal
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < 1e-10 {
            return Err(crate::SdfError::ComputationError(
                "Signal has no variance".to_string(),
            ));
        }

        // Compute equiprobable boundaries
        // Sort signal values and divide into quantiles
        let mut sorted = signal.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut boundaries = Vec::new();
        let samples_per_region = signal.len() as f64 / self.num_symbols as f64;

        for i in 1..self.num_symbols {
            let idx = ((i as f64) * samples_per_region) as usize;
            if idx < sorted.len() {
                boundaries.push(sorted[idx]);
            }
        }

        // Ensure no duplicates
        boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        boundaries.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

        if boundaries.len() >= self.num_symbols {
            boundaries.truncate(self.num_symbols - 1);
        }

        Partition::new(boundaries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maximum_entropy_partitioner() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let partitioner = MaximumEntropyPartitioner::new(2).unwrap();
        let partition = partitioner.compute_partition(&signal).unwrap();
        assert!(partition.boundaries.len() >= 1);
    }
}
