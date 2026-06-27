//! Wavelet basis selection and management
//!
//! Provides different wavelet families with their properties and computation methods.

use std::f64::consts::PI;
use serde::{Deserialize, Serialize};
use crate::Result;

/// Wavelet basis types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WaveletBasis {
    /// Haar wavelet
    Haar,
    /// Daubechies wavelets (db1 to db10)
    Daubechies(usize),
    /// Gaussian wavelets (gaus1 to gaus10)
    Gaussian(usize),
    /// Morlet (Gabor) wavelet
    Morlet,
    /// Mexican Hat wavelet
    MexicanHat,
}

impl WaveletBasis {
    /// Get the center frequency of the wavelet
    pub fn center_frequency(&self) -> f64 {
        match self {
            WaveletBasis::Haar => 0.5,
            WaveletBasis::Daubechies(n) => {
                // Center frequencies for Daubechies wavelets
                match n {
                    1 => 0.5,
                    2 => 0.629,
                    3 => 0.656,
                    4 => 0.670,
                    5 => 0.677,
                    6 => 0.682,
                    7 => 0.686,
                    8 => 0.688,
                    9 => 0.690,
                    10 => 0.691,
                    _ => 0.5,
                }
            }
            WaveletBasis::Gaussian(_) => 0.84,
            WaveletBasis::Morlet => 0.85,
            WaveletBasis::MexicanHat => 0.6,
        }
    }

    /// Get the number of vanishing moments
    pub fn vanishing_moments(&self) -> usize {
        match self {
            WaveletBasis::Haar => 1,
            WaveletBasis::Daubechies(n) => *n,
            WaveletBasis::Gaussian(n) => *n,
            WaveletBasis::Morlet => 1,
            WaveletBasis::MexicanHat => 2,
        }
    }

    /// Get the support size of the wavelet
    pub fn support_size(&self) -> usize {
        match self {
            WaveletBasis::Haar => 1,
            WaveletBasis::Daubechies(n) => 2 * n,
            WaveletBasis::Gaussian(_) => 10,
            WaveletBasis::Morlet => 8,
            WaveletBasis::MexicanHat => 8,
        }
    }

    /// Compute the wavelet value at a given point
    /// 
    /// This is a simplified computation for demonstration.
    /// Real implementations would use detailed wavelet coefficients.
    pub fn compute_wavelet(&self, t: f64) -> f64 {
        match self {
            WaveletBasis::Haar => {
                if t >= -0.5 && t < 0.0 {
                    1.0
                } else if t >= 0.0 && t < 0.5 {
                    -1.0
                } else {
                    0.0
                }
            }
            WaveletBasis::Gaussian(n) => {
                let n = *n as i32;
                // Hermite polynomial derivative of Gaussian
                let exp_term = (-t * t / 2.0).exp();
                // Simplified computation
                exp_term * (t * t - 1.0).powi(n)
            }
            WaveletBasis::Morlet => {
                // Morlet wavelet: exp(-t²/2) * cos(5t)
                (-t * t / 2.0).exp() * (5.0 * t).cos()
            }
            WaveletBasis::MexicanHat => {
                // Mexican hat: (1 - t²) * exp(-t²/2)
                (1.0 - t * t) * (-t * t / 2.0).exp()
            }
            WaveletBasis::Daubechies(_) => {
                // Simplified Daubechies computation
                (-t * t / 2.0).exp() * (t).cos()
            }
        }
    }

    /// Compute coherence between the wavelet and signal
    /// 
    /// Implements the cross-correlation measure from Eq. 11 of the paper.
    /// Returns a value between 0 and 1 indicating how well the wavelet
    /// matches the signal characteristics.
    pub fn coherence_with_signal(&self, signal: &[f64]) -> Result<f64> {
        if signal.is_empty() {
            return Err(crate::SdfError::InvalidParameter(
                "Signal cannot be empty".to_string(),
            ));
        }

        // Compute signal energy
        let signal_energy: f64 = signal.iter().map(|x| x * x).sum();
        if signal_energy == 0.0 {
            return Ok(0.0);
        }

        // Compute cross-correlation with wavelet at central frequency
        let center_freq = self.center_frequency();
        let mut cross_corr = 0.0;
        let dt = 1.0 / signal.len() as f64;

        for (i, &s) in signal.iter().enumerate() {
            let t = i as f64 * dt - 0.5;
            let wavelet_val = self.compute_wavelet(t);
            cross_corr += s * wavelet_val;
        }

        // Normalize by signal energy
        let coherence = (cross_corr.abs() / signal_energy).min(1.0);
        Ok(coherence)
    }
}

impl std::fmt::Display for WaveletBasis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaveletBasis::Haar => write!(f, "Haar"),
            WaveletBasis::Daubechies(n) => write!(f, "db{}", n),
            WaveletBasis::Gaussian(n) => write!(f, "gaus{}", n),
            WaveletBasis::Morlet => write!(f, "Morlet"),
            WaveletBasis::MexicanHat => write!(f, "MexicanHat"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_center_frequency() {
        let db2 = WaveletBasis::Daubechies(2);
        assert!((db2.center_frequency() - 0.629).abs() < 0.001);
    }

    #[test]
    fn test_wavelet_computation() {
        let haar = WaveletBasis::Haar;
        assert_eq!(haar.compute_wavelet(-0.25), 1.0);
        assert_eq!(haar.compute_wavelet(0.25), -1.0);
        assert_eq!(haar.compute_wavelet(1.0), 0.0);
    }

    #[test]
    fn test_vanishing_moments() {
        let db3 = WaveletBasis::Daubechies(3);
        assert_eq!(db3.vanishing_moments(), 3);
    }
}
