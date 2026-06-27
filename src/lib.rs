//! Symbolic Dynamic Filtering (SDF) Library
//! 
//! A comprehensive library for real-time anomaly detection and fault prognosis
//! in complex systems using wavelet transforms, symbolic dynamics, and Markov models.

pub mod wavelets;
pub mod partitioning;
pub mod symbolic;
pub mod markov;
pub mod anomaly;
pub mod utils;
pub mod python_bindings;

use thiserror::Error;

/// Main error type for the SDF library
#[derive(Error, Debug)]
pub enum SdfError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Signal processing error: {0}")]
    SignalProcessingError(String),

    #[error("Invalid wavelet: {0}")]
    InvalidWavelet(String),

    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Result type for SDF operations
pub type Result<T> = std::result::Result<T, SdfError>;

// Re-exports for convenience
pub use wavelets::{WaveletBasis, WaveletTransform};
pub use anomaly::{AnomalyDetector, AnomalyMeasure};
