//! Continuous wavelet transform implementation
//!
//! Provides the core wavelet transform functionality including CWT computation,
//! norm calculation, and scale series arrangement.

use ndarray::Array2;
use crate::{Result, SdfError};
use super::basis::WaveletBasis;

/// Wavelet transform container
#[derive(Debug, Clone)]
pub struct WaveletTransform {
    /// Wavelet coefficients (time × scale)
    pub coefficients: Array2<f64>,
    /// Scales used in the transform
    pub scales: Vec<usize>,
    /// Wavelet basis used
    pub basis: WaveletBasis,
}

impl WaveletTransform {
    /// Perform a continuous wavelet transform on a signal
    ///
    /// # Arguments
    /// * `signal` - Input time series
    /// * `scales` - Scale values to use in the transform
    /// * `wavelet` - Wavelet basis to use
    ///
    /// # Returns
    /// Wavelet transform result with coefficients and metadata
    ///
    /// Implements the CWT formula from Eq. 9:
    /// CWT(a,b) = (1/√a) ∫ ψ((t-b)/a) x(t) dt
    pub fn continuous(
        signal: &[f64],
        scales: &[usize],
        wavelet: WaveletBasis,
    ) -> Result<WaveletTransform> {
        if signal.is_empty() {
            return Err(SdfError::InvalidParameter(
                "Signal cannot be empty".to_string(),
            ));
        }

        if scales.is_empty() {
            return Err(SdfError::InvalidParameter(
                "Scales cannot be empty".to_string(),
            ));
        }

        let n_samples = signal.len();
        let n_scales = scales.len();

        // Initialize coefficient matrix
        let mut coefficients = Array2::<f64>::zeros((n_samples, n_scales));

        // Compute CWT coefficients
        for (scale_idx, &scale) in scales.iter().enumerate() {
            let scale_f64 = scale as f64;
            let norm_factor = 1.0 / scale_f64.sqrt();

            for time_idx in 0..n_samples {
                let mut coeff = 0.0;

                // Integrate over time
                for offset in 0..n_samples {
                    let t_normalized = (offset as f64 - time_idx as f64) / scale_f64;
                    
                    // Only compute where wavelet has significant support
                    if t_normalized.abs() < 5.0 {
                        let wavelet_val = wavelet.compute_wavelet(t_normalized);
                        coeff += signal[offset] * wavelet_val;
                    }
                }

                coefficients[[time_idx, scale_idx]] = coeff * norm_factor;
            }
        }

        Ok(WaveletTransform {
            coefficients,
            scales: scales.to_vec(),
            basis: wavelet,
        })
    }

    /// Compute the norm (magnitude) of coefficients for each scale
    ///
    /// Returns a vector of L2 norms, one for each scale.
    /// Used for scale selection and noise suppression analysis.
    pub fn compute_scale_norms(&self) -> Vec<f64> {
        self.scales
            .iter()
            .enumerate()
            .map(|(scale_idx, _)| {
                let col = self.coefficients.column(scale_idx);
                col.iter().map(|x| x * x).sum::<f64>().sqrt()
            })
            .collect()
    }

    /// Compute the norm across all coefficients
    pub fn compute_total_norm(&self) -> f64 {
        self.coefficients
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt()
    }

    /// Compute pseudo-frequency for each scale
    ///
    /// Implements Eq. 15 from the paper:
    /// f_p = F_c / (α * Δt)
    ///
    /// Where:
    /// - F_c is the center frequency of the wavelet
    /// - α is the scale
    /// - Δt is the sampling interval
    pub fn compute_pseudo_frequencies(
        &self,
        sampling_interval: f64,
    ) -> Vec<f64> {
        let center_freq = self.basis.center_frequency();

        self.scales
            .iter()
            .map(|&scale| {
                center_freq / ((scale as f64) * sampling_interval)
            })
            .collect()
    }

    /// Arrange wavelet coefficients into a scale series
    ///
    /// Stacks coefficients from smallest to largest scale and back down,
    /// creating a continuous series useful for subsequent analysis.
    ///
    /// This creates a feature vector that captures multi-scale information
    /// in a single continuous sequence.
    pub fn arrange_scale_series(&self) -> Vec<f64> {
        let mut series = Vec::new();

        // Get all scales sorted by scale value
        let mut scale_indices: Vec<(usize, usize)> = self.scales
            .iter()
            .enumerate()
            .map(|(idx, &val)| (idx, val))
            .collect();

        // Sort by scale value (ascending)
        scale_indices.sort_by_key(|(_, scale)| *scale);

        // Add coefficients from smallest to largest scale
        for (idx, _) in &scale_indices {
            let col = self.coefficients.column(*idx);
            series.extend(col.iter().copied());
        }

        // Add coefficients from largest to smallest scale (reverse)
        for (idx, _) in scale_indices.iter().rev() {
            let col = self.coefficients.column(*idx);
            series.extend(col.iter().copied());
        }

        series
    }

    /// Get a specific scale's coefficients
    pub fn get_scale_coefficients(&self, scale_idx: usize) -> Result<Vec<f64>> {
        if scale_idx >= self.scales.len() {
            return Err(SdfError::InvalidParameter(
                format!("Scale index {} out of bounds", scale_idx),
            ));
        }

        Ok(self.coefficients.column(scale_idx).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cwt_basic() {
        let signal = vec![1.0, 2.0, 1.5, 1.0, 0.5];
        let scales = vec![1, 2];
        let wavelet = WaveletBasis::Haar;

        let transform = WaveletTransform::continuous(&signal, &scales, wavelet)
            .expect("CWT should succeed");

        assert_eq!(transform.coefficients.nrows(), 5);
        assert_eq!(transform.coefficients.ncols(), 2);
    }

    #[test]
    fn test_scale_norms() {
        let signal = vec![1.0, 2.0, 1.0];
        let scales = vec![1, 2, 4];
        let wavelet = WaveletBasis::Haar;

        let transform = WaveletTransform::continuous(&signal, &scales, wavelet)
            .expect("CWT should succeed");

        let norms = transform.compute_scale_norms();
        assert_eq!(norms.len(), 3);
        assert!(norms.iter().all(|&n| n >= 0.0));
    }

    #[test]
    fn test_pseudo_frequencies() {
        let signal = vec![1.0, 2.0, 1.0];
        let scales = vec![1, 2];
        let wavelet = WaveletBasis::Haar;

        let transform = WaveletTransform::continuous(&signal, &scales, wavelet)
            .expect("CWT should succeed");

        let freqs = transform.compute_pseudo_frequencies(0.01);
        assert_eq!(freqs.len(), 2);
        assert!(freqs[0] > freqs[1]); // Smaller scale = higher frequency
    }
}
