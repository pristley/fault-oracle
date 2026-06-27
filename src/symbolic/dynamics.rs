//! Symbolic dynamics analysis
//!
//! Analyzes the properties of symbolic sequences.
//! Implements Phase 4.2: Symbolic Dynamics with phase space reconstruction

use std::collections::HashMap;
use ndarray::Array2;
use crate::Result;

/// Symbolic dynamics analyzer with phase space reconstruction
#[derive(Debug, Clone)]
pub struct SymbolicDynamics {
    sequence: Vec<char>,
}

impl SymbolicDynamics {
    /// Create from a symbolic sequence
    pub fn new(sequence: Vec<char>) -> Result<Self> {
        if sequence.is_empty() {
            return Err(crate::SdfError::InvalidParameter(
                "Sequence cannot be empty".to_string(),
            ));
        }

        Ok(SymbolicDynamics { sequence })
    }

    /// Phase space reconstruction using Takens' embedding theorem (Eq. 5)
    ///
    /// Implements Phase 4.2, Step 1: phase_space_reconstruction
    /// x(n) = [y(n), y(n+T), ..., y(n+(dE-1)T)]
    ///
    /// # Arguments
    /// * `signal` - Input signal
    /// * `time_lag` - Embedding time lag (T)
    /// * `embedding_dimension` - Embedding dimension (dE)
    ///
    /// # Returns
    /// Phase space matrix of shape (N-lag*(dE-1), dE)
    pub fn phase_space_reconstruction(
        signal: &[f64],
        time_lag: usize,
        embedding_dimension: usize,
    ) -> Result<Array2<f64>> {
        if signal.is_empty() {
            return Err(crate::SdfError::InvalidParameter(
                "Signal cannot be empty".to_string(),
            ));
        }

        if embedding_dimension == 0 {
            return Err(crate::SdfError::InvalidParameter(
                "Embedding dimension must be > 0".to_string(),
            ));
        }

        if time_lag == 0 {
            return Err(crate::SdfError::InvalidParameter(
                "Time lag must be > 0".to_string(),
            ));
        }

        // Calculate number of rows
        let max_index = embedding_dimension * time_lag;
        if signal.len() < max_index {
            return Err(crate::SdfError::InvalidParameter(
                format!(
                    "Signal length {} is too short for dE={}, tau={}",
                    signal.len(),
                    embedding_dimension,
                    time_lag
                ),
            ));
        }

        let n_rows = signal.len() - (embedding_dimension - 1) * time_lag;
        let mut phase_space = Array2::<f64>::zeros((n_rows, embedding_dimension));

        // Fill phase space matrix
        for i in 0..n_rows {
            for j in 0..embedding_dimension {
                let idx = i + j * time_lag;
                phase_space[[i, j]] = signal[idx];
            }
        }

        Ok(phase_space)
    }

    /// Determine optimal time lag using mutual information
    ///
    /// Implements Phase 4.2, Step 2: mutual_information
    /// Finds the first zero crossing of mutual information
    ///
    /// # Arguments
    /// * `signal` - Input signal
    ///
    /// # Returns
    /// Optimal time lag T
    pub fn mutual_information(signal: &[f64]) -> usize {
        if signal.is_empty() {
            return 1;
        }

        let mut max_lag = signal.len() / 2;
        if max_lag > 100 {
            max_lag = 100; // Limit for performance
        }

        let mut best_lag = 1;
        let mut best_mi = Self::compute_mutual_info_for_lag(signal, 1);

        for lag in 2..=max_lag {
            let mi = Self::compute_mutual_info_for_lag(signal, lag);

            // Find first minimum after first significant correlation
            if mi < best_mi * 0.5 {
                best_lag = lag;
                break;
            }

            if mi < best_mi {
                best_mi = mi;
                best_lag = lag;
            }
        }

        best_lag
    }

    /// Determine optimal embedding dimension using false nearest neighbors
    ///
    /// Implements Phase 4.2, Step 3: false_nearest_neighbors
    /// Uses FNN algorithm to find minimum sufficient embedding dimension
    ///
    /// # Arguments
    /// * `phase_space` - Phase space matrix
    ///
    /// # Returns
    /// Tuple of (optimal_embedding_dimension, percentage_false_neighbors)
    pub fn false_nearest_neighbors(phase_space: &Array2<f64>) -> (usize, f64) {
        if phase_space.nrows() < 2 || phase_space.ncols() == 0 {
            return (1, 100.0);
        }

        let n_points = phase_space.nrows();
        let dE = phase_space.ncols();

        // Compute threshold (10 * avg distance)
        let mut avg_dist = 0.0;
        let mut count = 0;

        for i in 0..n_points {
            for j in (i + 1)..n_points {
                let dist = Self::euclidean_distance(
                    phase_space.row(i).to_slice().unwrap(),
                    phase_space.row(j).to_slice().unwrap(),
                );
                avg_dist += dist;
                count += 1;
            }
        }

        if count == 0 {
            return (dE, 0.0);
        }

        avg_dist /= count as f64;
        let threshold = 10.0 * avg_dist;

        // Count false nearest neighbors
        let mut false_neighbors = 0;

        for i in 0..n_points {
            if let Some(nearest_idx) = Self::find_nearest_neighbor(phase_space, i) {
                let full_dist = Self::euclidean_distance(
                    phase_space.row(i).to_slice().unwrap(),
                    phase_space.row(nearest_idx).to_slice().unwrap(),
                );

                if full_dist > threshold {
                    false_neighbors += 1;
                }
            }
        }

        let fnn_percent = (false_neighbors as f64 / n_points as f64) * 100.0;
        (dE, fnn_percent)
    }

    /// Compute Shannon entropy of the sequence
    pub fn shannon_entropy(&self) -> f64 {
        let len = self.sequence.len() as f64;
        let mut counts: HashMap<char, f64> = HashMap::new();

        for &symbol in &self.sequence {
            *counts.entry(symbol).or_insert(0.0) += 1.0;
        }

        let mut entropy = 0.0;
        for count in counts.values() {
            let p = count / len;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Compute transition probabilities
    pub fn transition_matrix(&self) -> Result<(Vec<char>, Vec<Vec<f64>>)> {
        let mut symbols: Vec<char> = self.sequence.iter().cloned().collect();
        symbols.sort();
        symbols.dedup();

        let n = symbols.len();
        let mut matrix = vec![vec![0.0; n]; n];
        let mut counts = vec![0usize; n];

        // Build transition matrix
        for i in 0..self.sequence.len() - 1 {
            let from_idx = symbols.iter().position(|&s| s == self.sequence[i]).unwrap();
            let to_idx = symbols
                .iter()
                .position(|&s| s == self.sequence[i + 1])
                .unwrap();

            matrix[from_idx][to_idx] += 1.0;
            counts[from_idx] += 1;
        }

        // Normalize
        for i in 0..n {
            if counts[i] > 0 {
                for j in 0..n {
                    matrix[i][j] /= counts[i] as f64;
                }
            }
        }

        Ok((symbols, matrix))
    }

    /// Get sequence length
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if sequence is empty
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Get the sequence
    pub fn sequence(&self) -> &[char] {
        &self.sequence
    }

    /// Compute mutual information for a specific lag
    fn compute_mutual_info_for_lag(signal: &[f64], lag: usize) -> f64 {
        if lag >= signal.len() {
            return 0.0;
        }

        // Discretize signal into 10 bins
        let min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < 1e-10 {
            return 0.0;
        }

        let bin_width = (max - min) / 10.0;

        // Count joint occurrences
        let mut joint_counts: HashMap<(usize, usize), usize> = HashMap::new();
        let mut x_counts = vec![0usize; 10];
        let mut y_counts = vec![0usize; 10];

        for i in 0..(signal.len() - lag) {
            let x_bin = ((signal[i] - min) / bin_width).floor() as usize;
            let y_bin = ((signal[i + lag] - min) / bin_width).floor() as usize;

            let x_bin = x_bin.min(9);
            let y_bin = y_bin.min(9);

            *joint_counts.entry((x_bin, y_bin)).or_insert(0) += 1;
            x_counts[x_bin] += 1;
            y_counts[y_bin] += 1;
        }

        // Compute mutual information
        let n = (signal.len() - lag) as f64;
        let mut mi = 0.0;

        for ((x, y), count) in joint_counts {
            let p_xy = count as f64 / n;
            let p_x = x_counts[x] as f64 / n;
            let p_y = y_counts[y] as f64 / n;

            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).log2();
            }
        }

        mi
    }

    /// Compute Euclidean distance between vectors
    fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return f64::NAN;
        }

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Find nearest neighbor index (excluding self)
    fn find_nearest_neighbor(phase_space: &Array2<f64>, point_idx: usize) -> Option<usize> {
        let n_points = phase_space.nrows();
        let mut min_dist = f64::INFINITY;
        let mut nearest_idx = None;

        for j in 0..n_points {
            if j == point_idx {
                continue;
            }

            let dist = Self::euclidean_distance(
                phase_space.row(point_idx).to_slice().unwrap(),
                phase_space.row(j).to_slice().unwrap(),
            );

            if dist < min_dist {
                min_dist = dist;
                nearest_idx = Some(j);
            }
        }

        nearest_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_space_reconstruction() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let phase_space = SymbolicDynamics::phase_space_reconstruction(&signal, 1, 2).unwrap();

        assert_eq!(phase_space.nrows(), 5);
        assert_eq!(phase_space.ncols(), 2);
        assert_eq!(phase_space[[0, 0]], 1.0);
        assert_eq!(phase_space[[0, 1]], 2.0);
    }

    #[test]
    fn test_mutual_information() {
        let signal = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let lag = SymbolicDynamics::mutual_information(&signal);
        assert!(lag >= 1);
    }

    #[test]
    fn test_entropy() {
        let sequence = vec!['a', 'b', 'a', 'b', 'a', 'b'];
        let dynamics = SymbolicDynamics::new(sequence).unwrap();
        let entropy = dynamics.shannon_entropy();
        assert!((entropy - 1.0).abs() < 0.01);
    }
}
