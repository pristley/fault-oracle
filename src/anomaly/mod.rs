//! Anomaly detection module
//!
//! Provides anomaly measures and detection algorithms including
//! norm-based, angle-based, and KL-divergence measures.

pub mod measures;
pub mod detection;

pub use measures::AnomalyMeasure;
pub use detection::AnomalyDetector;
