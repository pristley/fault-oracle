//! Wavelet transform module
//!
//! Implements continuous wavelet transform (CWT) with various wavelet families,
//! scale selection strategies, and wavelet basis handling.

pub mod transform;
pub mod basis;
pub mod scales;

pub use basis::WaveletBasis;
pub use transform::WaveletTransform;
pub use scales::ScaleSelector;
