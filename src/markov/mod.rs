//! Markov model module
//!
//! Implements D-Markov machines and transition matrices for state analysis
//! and behavior modeling.

pub mod d_markov;
pub mod state;
pub mod transitions;

pub use d_markov::DMarkovMachine;
pub use state::State;
pub use transitions::TransitionMatrix;
