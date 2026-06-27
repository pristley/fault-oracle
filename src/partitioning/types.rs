//! Partition types and traits
//!
//! Defines the core partition structures used for symbolic dynamics.

use serde::{Deserialize, Serialize};
use crate::Result;

/// Partition boundaries for symbolic encoding
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Partition {
    /// Partition boundaries (edges between symbols)
    pub boundaries: Vec<f64>,
    /// Number of partition regions
    pub num_regions: usize,
}

impl Partition {
    /// Create a new partition with the given boundaries
    ///
    /// Boundaries should be sorted in ascending order.
    pub fn new(boundaries: Vec<f64>) -> Result<Self> {
        if boundaries.is_empty() {
            return Err(crate::SdfError::InvalidParameter(
                "Boundaries cannot be empty".to_string(),
            ));
        }

        // Check if sorted
        for i in 1..boundaries.len() {
            if boundaries[i] <= boundaries[i - 1] {
                return Err(crate::SdfError::InvalidParameter(
                    "Boundaries must be strictly increasing".to_string(),
                ));
            }
        }

        let num_regions = boundaries.len() + 1;

        Ok(Partition {
            boundaries,
            num_regions,
        })
    }

    /// Encode a value to a symbol based on this partition
    ///
    /// Returns the index of the region the value falls into.
    pub fn encode(&self, value: f64) -> usize {
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if value < boundary {
                return i;
            }
        }
        self.boundaries.len()
    }

    /// Get the boundaries for this partition
    pub fn get_boundaries(&self) -> &[f64] {
        &self.boundaries
    }
}

/// Partition scheme definitions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PartitionScheme {
    /// Uniform quantization
    Uniform,
    /// Maximum entropy partitioning
    MaximumEntropy,
    /// Equiprobable partitioning
    Equiprobable,
}

impl std::fmt::Display for PartitionScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PartitionScheme::Uniform => write!(f, "Uniform"),
            PartitionScheme::MaximumEntropy => write!(f, "MaximumEntropy"),
            PartitionScheme::Equiprobable => write!(f, "Equiprobable"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_creation() {
        let boundaries = vec![-1.0, 0.0, 1.0];
        let partition = Partition::new(boundaries).unwrap();
        assert_eq!(partition.num_regions, 4);
    }

    #[test]
    fn test_partition_encoding() {
        let boundaries = vec![-1.0, 0.0, 1.0];
        let partition = Partition::new(boundaries).unwrap();

        assert_eq!(partition.encode(-2.0), 0);
        assert_eq!(partition.encode(-0.5), 1);
        assert_eq!(partition.encode(0.5), 2);
        assert_eq!(partition.encode(2.0), 3);
    }

    #[test]
    fn test_invalid_boundaries() {
        let boundaries = vec![1.0, 0.0]; // Not sorted
        assert!(Partition::new(boundaries).is_err());
    }
}
