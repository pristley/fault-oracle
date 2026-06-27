//! Uniform partitioning
//!
//! Implements simple uniform partitioning strategies.

use crate::Result;
use super::types::Partition;

/// Uniform partitioner
#[derive(Debug, Clone)]
pub struct UniformPartitioner {
    /// Number of symbols (partition regions)
    num_symbols: usize,
}

impl UniformPartitioner {
    /// Create a new uniform partitioner
    pub fn new(num_symbols: usize) -> Result<Self> {
        if num_symbols < 2 {
            return Err(crate::SdfError::InvalidParameter(
                "Number of symbols must be at least 2".to_string(),
            ));
        }

        Ok(UniformPartitioner { num_symbols })
    }

    /// Compute uniform partition boundaries
    ///
    /// Divides the signal range into equal-width regions.
    pub fn compute_partition(&self, signal: &[f64]) -> Result<Partition> {
        if signal.is_empty() {
            return Err(crate::SdfError::InvalidParameter(
                "Signal cannot be empty".to_string(),
            ));
        }

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

        let width = (max - min) / (self.num_symbols as f64);

        let boundaries: Vec<f64> = (1..self.num_symbols)
            .map(|i| min + (i as f64) * width)
            .collect();

        Partition::new(boundaries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_partitioner() {
        let signal = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let partitioner = UniformPartitioner::new(2).unwrap();
        let partition = partitioner.compute_partition(&signal).unwrap();
        assert_eq!(partition.boundaries.len(), 1);
    }
}
