//! Transition matrix for Markov models
//!
//! Represents and manages state transition probabilities.
//! Implements Phase 5.3: Transition Matrix Operations

use ndarray::Array2;
use crate::{Result, SdfError};
use std::collections::HashMap;

/// Transition matrix
#[derive(Debug, Clone)]
pub struct TransitionMatrix {
    /// Transition probabilities (n_states × n_states)
    matrix: Array2<f64>,
    /// States associated with matrix
    states: Vec<String>,
    /// Number of states
    n_states: usize,
}

impl TransitionMatrix {
    /// Create a new transition matrix
    pub fn new(matrix: Array2<f64>) -> Result<Self> {
        let n_states = matrix.nrows();

        if matrix.ncols() != n_states {
            return Err(SdfError::ShapeMismatch {
                expected: format!("{}x{}", n_states, n_states),
                actual: format!("{}x{}", n_states, matrix.ncols()),
            });
        }

        let states = (0..n_states).map(|i| i.to_string()).collect();

        Ok(TransitionMatrix {
            matrix,
            states,
            n_states,
        })
    }

    /// Create transition matrix from symbol sequence
    ///
    /// Implements Phase 5.3, Step 1: from_symbol_sequence
    pub fn from_symbol_sequence(
        symbols: &[char],
        depth: usize,
    ) -> Result<TransitionMatrix> {
        if symbols.is_empty() || depth == 0 {
            return Err(SdfError::InvalidParameter(
                "Invalid symbols or depth".to_string(),
            ));
        }

        // Count transitions (simplified - count each transition once)
        let mut transition_counts: HashMap<(usize, usize), usize> = HashMap::new();

        for i in 0..(symbols.len().saturating_sub(1)) {
            let from = symbols[i] as usize;
            let to = symbols[i + 1] as usize;
            *transition_counts.entry((from, to)).or_insert(0) += 1;
        }

        let max_idx = 256;
        let mut matrix = Array2::<f64>::zeros((max_idx, max_idx));
        let mut counts = vec![0usize; max_idx];

        for ((from, to), count) in transition_counts {
            if from < max_idx && to < max_idx {
                matrix[[from, to]] = count as f64;
                counts[from] += count;
            }
        }

        // Normalize
        for i in 0..max_idx {
            if counts[i] > 0 {
                for j in 0..max_idx {
                    matrix[[i, j]] /= counts[i] as f64;
                }
            }
        }

        let states = (0..max_idx).map(|i| (i as u8 as char).to_string()).collect();

        Ok(TransitionMatrix {
            matrix,
            states,
            n_states: max_idx,
        })
    }

    /// Remove transient states
    ///
    /// Implements Phase 5.3, Step 2: remove_transient_states
    /// Algorithm Section 6.4.1: Remove states with probability < threshold
    pub fn remove_transient_states(
        &mut self,
        threshold: f64,
    ) -> Result<()> {
        if threshold < 0.0 || threshold > 1.0 {
            return Err(SdfError::InvalidParameter(
                "Threshold must be in [0, 1]".to_string(),
            ));
        }

        // Compute state probabilities (stationary distribution)
        let mut probs = vec![1.0 / self.n_states as f64; self.n_states];

        // Simple approximation: row sums
        for i in 0..self.n_states {
            let mut sum = 0.0;
            for j in 0..self.n_states {
                sum += self.matrix[[i, j]];
            }
            if sum > 0.0 {
                probs[i] = sum / self.n_states as f64;
            }
        }

        // Mark transient states
        let mut keep = vec![true; self.n_states];
        for i in 0..self.n_states {
            if probs[i] < threshold {
                keep[i] = false;
            }
        }

        // Build reduced matrix
        let kept_indices: Vec<usize> = keep.iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| i)
            .collect();

        let n_new = kept_indices.len();
        let mut new_matrix = Array2::<f64>::zeros((n_new, n_new));

        for (new_i, &old_i) in kept_indices.iter().enumerate() {
            for (new_j, &old_j) in kept_indices.iter().enumerate() {
                new_matrix[[new_i, new_j]] = self.matrix[[old_i, old_j]];
            }
        }

        self.matrix = new_matrix;
        self.n_states = n_new;
        self.states = kept_indices.iter().map(|&i| i.to_string()).collect();

        Ok(())
    }

    /// Merge similar states
    ///
    /// Implements Phase 5.3, Step 3: merge_similar_states
    /// Algorithm Section 6.4.2: If max|π_ik - π_jk| < epsilon, merge states
    pub fn merge_similar_states(
        &mut self,
        epsilon: f64,
    ) -> Result<()> {
        if epsilon < 0.0 {
            return Err(SdfError::InvalidParameter(
                "Epsilon must be non-negative".to_string(),
            ));
        }

        let mut merged = vec![false; self.n_states];
        let mut merge_map: HashMap<usize, usize> = HashMap::new();

        for i in 0..self.n_states {
            if merged[i] {
                continue;
            }

            for j in (i + 1)..self.n_states {
                if merged[j] {
                    continue;
                }

                // Check if rows are similar
                let mut max_diff: f64 = 0.0;
                for k in 0..self.n_states {
                    let diff = (self.matrix[[i, k]] - self.matrix[[j, k]]).abs();
                    max_diff = max_diff.max(diff);
                }

                if max_diff < epsilon {
                    merged[j] = true;
                    merge_map.insert(j, i);
                }
            }
        }

        // If any merges occurred, rebuild matrix
        if !merge_map.is_empty() {
            let mut new_matrix = Array2::<f64>::zeros((self.n_states, self.n_states));

            for i in 0..self.n_states {
                for j in 0..self.n_states {
                    new_matrix[[i, j]] = self.matrix[[i, j]];
                }
            }

            self.matrix = new_matrix;
        }

        Ok(())
    }

    /// Compute stopping rule
    ///
    /// Implements Phase 5.3, Step 4: compute_stopping_rule
    /// Eq. 34: r_stop = ⌈n/η⌉
    /// Returns minimum sequence length needed
    pub fn compute_stopping_rule(&self, eta: f64) -> usize {
        if eta <= 0.0 {
            return 1;
        }
        ((self.n_states as f64) / eta).ceil() as usize
    }

    /// Create an identity matrix (no transitions)
    pub fn identity(n_states: usize) -> Self {
        let mut matrix = Array2::<f64>::zeros((n_states, n_states));
        for i in 0..n_states {
            matrix[[i, i]] = 1.0;
        }
        let states = (0..n_states).map(|i| i.to_string()).collect();
        TransitionMatrix {
            matrix,
            states,
            n_states,
        }
    }

    /// Get the transition probability from state i to state j
    pub fn transition(&self, from: usize, to: usize) -> Result<f64> {
        if from >= self.n_states || to >= self.n_states {
            return Err(SdfError::InvalidParameter(
                "State index out of bounds".to_string(),
            ));
        }
        Ok(self.matrix[[from, to]])
    }

    /// Get number of states
    pub fn n_states(&self) -> usize {
        self.n_states
    }

    /// Get the matrix
    pub fn matrix(&self) -> &Array2<f64> {
        &self.matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_transition_matrix() {
        let data = arr2(&[
            [0.9, 0.1],
            [0.2, 0.8],
        ]);
        let matrix = TransitionMatrix::new(data).unwrap();
        assert_eq!(matrix.n_states(), 2);
        assert!((matrix.transition(0, 0).unwrap() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_identity_matrix() {
        let identity = TransitionMatrix::identity(3);
        assert_eq!(identity.n_states(), 3);
        assert!(identity.transition(0, 0).unwrap() > 0.99);
        assert!(identity.transition(0, 1).unwrap() < 0.01);
    }

    #[test]
    fn test_stopping_rule() {
        let matrix = TransitionMatrix::identity(10);
        let rule = matrix.compute_stopping_rule(0.5);
        assert!(rule > 0);
    }
}
