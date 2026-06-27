//! Symbolic dynamics module
//!
//! Handles conversion of continuous signals to symbolic sequences,
//! alphabet generation, and symbolic dynamics analysis.

pub mod encoding;
pub mod dynamics;
pub mod alphabet;

pub use alphabet::Alphabet;
pub use encoding::SymbolicEncoder;
pub use dynamics::SymbolicDynamics;
