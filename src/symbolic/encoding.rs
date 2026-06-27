//! Symbolic encoding from continuous signals
//!
//! Converts continuous signal values to symbolic sequences.
//! Implements Phase 4.1: Symbol Sequence Generation

use crate::Result;
use crate::partitioning::types::Partition;
use super::alphabet::Alphabet;
use std::collections::HashMap;

/// Symbolic encoder for converting continuous signals to symbol sequences
#[derive(Debug, Clone)]
pub struct SymbolicEncoder {
    alphabet: Alphabet,
    partition: Partition,
}

impl SymbolicEncoder {
    /// Create a new symbolic encoder
    pub fn new(alphabet: Alphabet, partition: Partition) -> Result<Self> {
        if alphabet.size() != partition.num_regions {
            return Err(crate::SdfError::InvalidParameter(
                "Alphabet size must match number of partition regions".to_string(),
            ));
        }

        Ok(SymbolicEncoder {
            alphabet,
            partition,
        })
    }

    /// Encode a time series to a symbolic sequence
    ///
    /// Implements Phase 4.1, Step 1: encode_timeseries
    /// For each data point, finds which partition it belongs to and maps to corresponding symbol
    ///
    /// # Arguments
    /// * `data` - Continuous time series
    /// * `partition` - Partition boundaries for discretization
    ///
    /// # Returns
    /// Symbol sequence
    pub fn encode_timeseries(&self, data: &[f64]) -> Result<Vec<char>> {
        if data.is_empty() {
            return Err(crate::SdfError::InvalidParameter(
                "Data cannot be empty".to_string(),
            ));
        }

        let sequence = data
            .iter()
            .map(|&value| {
                let region = self.partition.encode(value);
                self.alphabet.symbol(region).unwrap_or('?')
            })
            .collect();

        Ok(sequence)
    }

    /// Encode wavelet coefficients to a symbolic sequence
    ///
    /// Implements Phase 4.1, Step 2: encode_wavelet_coefficients
    /// More robust to noise than direct time-domain encoding (Section 7)
    ///
    /// # Arguments
    /// * `scale_series` - Wavelet domain features
    /// * `partition` - Partition boundaries
    ///
    /// # Returns
    /// Symbol sequence
    pub fn encode_wavelet_coefficients(&self, scale_series: &[f64]) -> Result<Vec<char>> {
        if scale_series.is_empty() {
            return Err(crate::SdfError::InvalidParameter(
                "Scale series cannot be empty".to_string(),
            ));
        }

        // Apply smoothing for robustness (wavelet-specific preprocessing)
        let smoothed = Self::smooth_series(scale_series, 3);

        // Encode smoothed series
        let sequence = smoothed
            .iter()
            .map(|&value| {
                let region = self.partition.encode(value);
                self.alphabet.symbol(region).unwrap_or('?')
            })
            .collect();

        Ok(sequence)
    }

    /// Compute probability distribution of symbols in sequence
    ///
    /// Implements Phase 4.1, Step 3: symbol_probabilities
    /// Count symbol occurrences and return probability distribution
    ///
    /// # Arguments
    /// * `sequence` - Symbol sequence
    ///
    /// # Returns
    /// HashMap mapping symbols to probabilities
    pub fn symbol_probabilities(&self, sequence: &[char]) -> HashMap<char, f64> {
        if sequence.is_empty() {
            return HashMap::new();
        }

        let mut counts: HashMap<char, usize> = HashMap::new();

        for &symbol in sequence {
            *counts.entry(symbol).or_insert(0) += 1;
        }

        let total = sequence.len() as f64;
        let mut probabilities: HashMap<char, f64> = HashMap::new();

        for (symbol, count) in counts {
            probabilities.insert(symbol, count as f64 / total);
        }

        probabilities
    }

    /// Encode with a specific word length (for n-grams)
    pub fn encode_words(&self, signal: &[f64], word_length: usize) -> Result<Vec<String>> {
        if word_length == 0 {
            return Err(crate::SdfError::InvalidParameter(
                "Word length must be > 0".to_string(),
            ));
        }

        let sequence = self.encode_timeseries(signal)?;

        let words = sequence
            .windows(word_length)
            .map(|window| window.iter().collect::<String>())
            .collect();

        Ok(words)
    }

    /// Get the alphabet
    pub fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }

    /// Get the partition
    pub fn partition(&self) -> &Partition {
        &self.partition
    }

    /// Smooth a series using a simple moving average
    fn smooth_series(data: &[f64], window_size: usize) -> Vec<f64> {
        if data.is_empty() || window_size == 0 {
            return data.to_vec();
        }

        let window_size = window_size.min(data.len());
        let mut smoothed = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            let start = if i >= window_size / 2 {
                i - window_size / 2
            } else {
                0
            };

            let end = (i + window_size / 2 + 1).min(data.len());
            let mean: f64 = data[start..end].iter().sum::<f64>() / (end - start) as f64;
            smoothed.push(mean);
        }

        smoothed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_timeseries() {
        let alphabet = Alphabet::new(2).unwrap();
        let partition = Partition::new(vec![0.0]).unwrap();
        let encoder = SymbolicEncoder::new(alphabet, partition).unwrap();

        let data = vec![-1.0, 1.0, -0.5, 0.5];
        let sequence = encoder.encode_timeseries(&data).unwrap();
        assert_eq!(sequence.len(), 4);
        assert_eq!(sequence[0], 'a'); // Below 0.0
        assert_eq!(sequence[1], 'b'); // Above 0.0
    }

    #[test]
    fn test_encode_wavelet_coefficients() {
        let alphabet = Alphabet::new(2).unwrap();
        let partition = Partition::new(vec![0.5]).unwrap();
        let encoder = SymbolicEncoder::new(alphabet, partition).unwrap();

        let coeffs = vec![0.1, 0.2, 0.8, 0.9];
        let sequence = encoder.encode_wavelet_coefficients(&coeffs).unwrap();
        assert_eq!(sequence.len(), 4);
    }

    #[test]
    fn test_symbol_probabilities() {
        let alphabet = Alphabet::new(3).unwrap();
        let partition = Partition::new(vec![-1.0, 1.0]).unwrap();
        let encoder = SymbolicEncoder::new(alphabet, partition).unwrap();

        let sequence = vec!['a', 'a', 'b', 'b', 'b', 'c'];
        let probs = encoder.symbol_probabilities(&sequence);

        assert!((probs.get(&'a').unwrap_or(&0.0) - 1.0 / 3.0).abs() < 0.01);
        assert!((probs.get(&'b').unwrap_or(&0.0) - 0.5).abs() < 0.01);
    }
}
