//! State definitions for Markov models
//!
//! Defines the state structure used in D-Markov machines.
//! Implements Phase 5.2: State Management

use crate::Result;

/// State in a Markov model
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    /// State identifier
    id: usize,
    /// State label (D-length symbol word)
    word: String,
    /// Number of visits to this state
    visit_count: usize,
    /// State probability
    probability: f64,
}

impl State {
    /// Create a new state
    pub fn new(id: usize, word: String) -> Result<Self> {
        Ok(State {
            id,
            word,
            visit_count: 0,
            probability: 0.0,
        })
    }

    /// Create a state from a symbol sequence (word)
    pub fn from_word(id: usize, word: &str) -> Result<Self> {
        Ok(State {
            id,
            word: word.to_string(),
            visit_count: 0,
            probability: 0.0,
        })
    }

    /// Get the state ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the state label
    pub fn word(&self) -> &str {
        &self.word
    }

    /// Get visit count
    pub fn visit_count(&self) -> usize {
        self.visit_count
    }

    /// Get state probability
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Set state probability
    pub fn set_probability(&mut self, prob: f64) -> Result<()> {
        if prob < 0.0 || prob > 1.0 {
            return Err(crate::SdfError::InvalidParameter(
                "Probability must be in [0, 1]".to_string(),
            ));
        }
        self.probability = prob;
        Ok(())
    }

    /// Check if state is transient
    ///
    /// Implements Phase 5.2: is_transient
    /// Remove states with probability < threshold (usually 1/N)
    pub fn is_transient(&self, threshold: f64) -> bool {
        self.probability < threshold
    }

    /// Check if states can merge
    ///
    /// Implements Phase 5.2: can_merge
    /// States can merge if they have similar transition probabilities
    pub fn can_merge(&self, _other: &State, _epsilon: f64) -> bool {
        // In full implementation, would compare transition distributions
        false
    }

    /// Merge with another state
    ///
    /// Implements Phase 5.2: merge_with
    /// Combine two similar states, sum probabilities
    pub fn merge_with(&self, other: &State) -> State {
        State {
            id: self.id,
            word: format!("{}+{}", self.word, other.word),
            visit_count: self.visit_count + other.visit_count,
            probability: self.probability + other.probability,
        }
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State(id={}, word={}, p={})", self.id, self.word, self.probability)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = State::new(0, "a".to_string()).unwrap();
        assert_eq!(state.id(), 0);
        assert_eq!(state.word(), "a");
    }

    #[test]
    fn test_state_from_word() {
        let state = State::from_word(1, "abc").unwrap();
        assert_eq!(state.word(), "abc");
    }

    #[test]
    fn test_state_probability() {
        let mut state = State::new(0, "a".to_string()).unwrap();
        state.set_probability(0.5).unwrap();
        assert_eq!(state.probability(), 0.5);
    }

    #[test]
    fn test_is_transient() {
        let mut state = State::new(0, "a".to_string()).unwrap();
        state.set_probability(0.05).unwrap();
        assert!(state.is_transient(0.1));
    }
}
