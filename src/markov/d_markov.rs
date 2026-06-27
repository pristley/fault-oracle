//! D-Markov machine implementation
//!
//! Implements D-Markov machines for order-dependent state analysis.
//! Implements Phase 5.1: Core D-Markov Implementation

use ndarray::Array2;
use std::collections::HashMap;
use crate::{Result, SdfError};

/// D-Markov machine for order-dependent state analysis
#[derive(Debug, Clone)]
pub struct DMarkovMachine {
    /// Machine order (state memory depth)
    order: usize,
    /// Alphabet size
    alphabet_size: usize,
    /// All possible D-length sequences (states)
    states: Vec<String>,
    /// Transition probability matrix (Π)
    transition_matrix: Array2<f64>,
    /// State probability vector (p)
    state_probabilities: Vec<f64>,
}

impl DMarkovMachine {
    /// Create a new D-Markov machine from symbol sequence
    ///
    /// Implements Phase 5.1, Step 1: new
    /// Initializes states and builds transition matrix
    ///
    /// # Arguments
    /// * `symbol_sequence` - Symbol sequence from encoding
    /// * `depth` - D parameter (order/memory depth)
    /// * `alphabet_size` - Size of the alphabet
    ///
    /// # Returns
    /// D-Markov machine
    pub fn new(
        symbol_sequence: &[char],
        depth: usize,
        alphabet_size: usize,
    ) -> Result<DMarkovMachine> {
        if symbol_sequence.is_empty() {
            return Err(SdfError::InvalidParameter(
                "Symbol sequence cannot be empty".to_string(),
            ));
        }

        if depth == 0 {
            return Err(SdfError::InvalidParameter(
                "Depth must be > 0".to_string(),
            ));
        }

        if alphabet_size == 0 {
            return Err(SdfError::InvalidParameter(
                "Alphabet size must be > 0".to_string(),
            ));
        }

        // Generate all possible D-length states
        let states = Self::generate_states(depth, alphabet_size);
        let _n_states = states.len();

        // Build transition matrix
        let transition_matrix = Self::build_transition_matrix(symbol_sequence, &states, depth)?;

        // Compute state probabilities
        let state_probabilities = Self::compute_state_probabilities_vector(&transition_matrix)?;

        Ok(DMarkovMachine {
            order: depth,
            alphabet_size,
            states,
            transition_matrix,
            state_probabilities,
        })
    }

    /// Generate all possible D-length word states
    fn generate_states(depth: usize, alphabet_size: usize) -> Vec<String> {
        let mut states = Vec::new();

        fn generate_recursive(
            depth: usize,
            alphabet_size: usize,
            current: String,
            states: &mut Vec<String>,
        ) {
            if depth == 0 {
                states.push(current);
                return;
            }

            for i in 0..alphabet_size {
                let symbol = (('a' as u8 + i as u8) as char).to_string();
                generate_recursive(depth - 1, alphabet_size, current.clone() + &symbol, states);
            }
        }

        generate_recursive(depth, alphabet_size, String::new(), &mut states);
        states
    }

    /// Build transition matrix from symbol sequence
    ///
    /// Implements Phase 5.1, Step 2: build_transition_matrix
    /// Uses sliding block codes to count transitions (Eqs. 27-28)
    pub fn build_transition_matrix(
        symbol_sequence: &[char],
        states: &[String],
        depth: usize,
    ) -> Result<Array2<f64>> {
        let n_states = states.len();
        let mut matrix = Array2::<f64>::zeros((n_states, n_states));

        // Create state index map
        let mut state_index: HashMap<String, usize> = HashMap::new();
        for (i, state) in states.iter().enumerate() {
            state_index.insert(state.clone(), i);
        }

        // Count transitions using sliding windows
        if symbol_sequence.len() < depth + 1 {
            return Err(SdfError::ComputationError(
                "Sequence too short for given depth".to_string(),
            ));
        }

        let mut counts = vec![0usize; n_states];

        for i in 0..(symbol_sequence.len() - depth) {
            // Current state: (D-length word)
            let current_state: String = symbol_sequence[i..i + depth]
                .iter()
                .collect();

            // Next symbol
            let next_symbol = symbol_sequence[i + depth];

            // Next state: (shift window and add new symbol)
            let next_state = if depth > 1 {
                format!(
                    "{}{}",
                    &current_state[1..],
                    next_symbol
                )
            } else {
                next_symbol.to_string()
            };

            if let (Some(&from_idx), Some(&to_idx)) = (
                state_index.get(&current_state),
                state_index.get(&next_state),
            ) {
                matrix[[from_idx, to_idx]] += 1.0;
                counts[from_idx] += 1;
            }
        }

        // Normalize to get probabilities (Eq. 27-28)
        for i in 0..n_states {
            if counts[i] > 0 {
                let norm = counts[i] as f64;
                for j in 0..n_states {
                    matrix[[i, j]] /= norm;
                }
            }
        }

        Ok(matrix)
    }

    /// Compute state probability vector using power iteration
    ///
    /// Implements Phase 5.1, Step 3: compute_state_probabilities
    /// Finds left eigenvector for eigenvalue λ=1
    fn compute_state_probabilities_vector(matrix: &Array2<f64>) -> Result<Vec<f64>> {
        let n = matrix.nrows();

        // Ensure matrix is square
        if matrix.ncols() != n {
            return Err(SdfError::ShapeMismatch {
                expected: format!("{}x{}", n, n),
                actual: format!("{}x{}", n, matrix.ncols()),
            });
        }

        // Power iteration for stationary distribution
        let mut p = vec![1.0 / n as f64; n];
        let max_iterations = 1000;
        let tolerance = 1e-6;

        for _ in 0..max_iterations {
            let mut p_new = vec![0.0; n];

            // p_new = p * matrix (left multiplication)
            for i in 0..n {
                for j in 0..n {
                    p_new[i] += p[j] * matrix[[j, i]];
                }
            }

            // Check convergence
            let diff: f64 = p_new
                .iter()
                .zip(&p)
                .map(|(a, b)| (a - b).abs())
                .sum();

            p = p_new;

            if diff < tolerance {
                break;
            }
        }

        // Normalize
        let sum: f64 = p.iter().sum();
        if sum > 0.0 {
            for pi in &mut p {
                *pi /= sum;
            }
        }

        Ok(p)
    }

    /// Compute entropy rate (Eq. 29)
    ///
    /// Implements Phase 5.1, Step 4: entropy_rate
    /// hµ = -Σ pi Σ π_ij log(π_ij)
    pub fn entropy_rate(&self) -> f64 {
        let mut entropy = 0.0;

        for i in 0..self.states.len() {
            let pi = self.state_probabilities[i];

            if pi > 0.0 {
                for j in 0..self.states.len() {
                    let pi_ij = self.transition_matrix[[i, j]];

                    if pi_ij > 0.0 {
                        entropy -= pi * pi_ij * pi_ij.log2();
                    }
                }
            }
        }

        entropy
    }

    /// Select optimal depth automatically
    ///
    /// Implements Phase 5.1, Step 5: select_optimal_depth
    /// Algorithm Section 6.3: entropy stabilizes when D is sufficient
    pub fn select_optimal_depth(
        symbol_sequence: &[char],
        alphabet_size: usize,
        max_depth: usize,
        epsilon: f64,
    ) -> Result<usize> {
        if symbol_sequence.is_empty() {
            return Err(SdfError::InvalidParameter(
                "Symbol sequence cannot be empty".to_string(),
            ));
        }

        if max_depth == 0 {
            return Err(SdfError::InvalidParameter(
                "Max depth must be > 0".to_string(),
            ));
        }

        let mut previous_entropy = 0.0;
        let mut stable_count = 0;
        const STABILITY_THRESHOLD: usize = 2;

        for depth in 1..=max_depth {
            let machine = DMarkovMachine::new(symbol_sequence, depth, alphabet_size)?;
            let entropy = machine.entropy_rate();

            // Check if entropy stabilized
            let entropy_diff = (entropy - previous_entropy).abs();

            if entropy_diff < epsilon {
                stable_count += 1;
                if stable_count >= STABILITY_THRESHOLD {
                    return Ok(depth);
                }
            } else {
                stable_count = 0;
            }

            previous_entropy = entropy;
        }

        // If not stabilized, return max_depth
        Ok(max_depth)
    }

    /// Get the order of the machine
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the number of states
    pub fn n_states(&self) -> usize {
        self.states.len()
    }

    /// Get a state by index
    pub fn state(&self, index: usize) -> Result<&String> {
        self.states.get(index).ok_or(SdfError::InvalidParameter(
            "State index out of bounds".to_string(),
        ))
    }

    /// Get all states
    pub fn states(&self) -> &[String] {
        &self.states
    }

    /// Get transition matrix
    pub fn transition_matrix(&self) -> &Array2<f64> {
        &self.transition_matrix
    }

    /// Get state probabilities
    pub fn state_probabilities(&self) -> &[f64] {
        &self.state_probabilities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d_markov_creation() {
        let sequence = vec!['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'];
        let machine = DMarkovMachine::new(&sequence, 1, 2).unwrap();

        assert_eq!(machine.order(), 1);
        assert_eq!(machine.n_states(), 2);
    }

    #[test]
    fn test_entropy_rate() {
        let sequence = vec!['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'];
        let machine = DMarkovMachine::new(&sequence, 1, 2).unwrap();
        let entropy = machine.entropy_rate();

        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_state_probabilities() {
        let sequence = vec!['a', 'a', 'a', 'b', 'b', 'b'];
        let machine = DMarkovMachine::new(&sequence, 1, 2).unwrap();
        let probs = machine.state_probabilities();

        assert_eq!(probs.len(), 2);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_select_optimal_depth() {
        let sequence = vec!['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'];
        let depth = DMarkovMachine::select_optimal_depth(&sequence, 3, 5, 0.01).unwrap();

        assert!(depth >= 1 && depth <= 5);
    }
}
