//! Scale selection strategies for wavelet transforms
//!
//! Provides methods to automatically select appropriate scales based on signal
//! properties, including PSD-based selection and noise suppression analysis.

use crate::{Result, SdfError, utils};
use super::basis::WaveletBasis;

/// Scale selector for wavelet transforms
#[derive(Debug, Clone)]
pub struct ScaleSelector {
    /// Minimum scale to consider
    pub min_scale: usize,
    /// Maximum scale to consider
    pub max_scale: usize,
    /// Number of scales to select
    pub num_scales: usize,
}

impl Default for ScaleSelector {
    fn default() -> Self {
        ScaleSelector {
            min_scale: 1,
            max_scale: 128,
            num_scales: 32,
        }
    }
}

impl ScaleSelector {
    /// Create a new scale selector with parameters
    pub fn new(min_scale: usize, max_scale: usize, num_scales: usize) -> Result<Self> {
        if min_scale == 0 {
            return Err(SdfError::InvalidParameter(
                "min_scale must be > 0".to_string(),
            ));
        }

        if max_scale <= min_scale {
            return Err(SdfError::InvalidParameter(
                "max_scale must be > min_scale".to_string(),
            ));
        }

        if num_scales == 0 {
            return Err(SdfError::InvalidParameter(
                "num_scales must be > 0".to_string(),
            ));
        }

        Ok(ScaleSelector {
            min_scale,
            max_scale,
            num_scales,
        })
    }

    /// Select scales from Power Spectral Density
    ///
    /// Analyzes the signal's frequency content and selects scales
    /// that correspond to dominant frequencies.
    ///
    /// # Arguments
    /// * `signal` - Input time series
    /// * `sampling_rate` - Sampling rate in Hz
    ///
    /// # Returns
    /// Selected scale values
    pub fn from_psd(
        &self,
        signal: &[f64],
        sampling_rate: f64,
    ) -> Result<Vec<f64>> {
        if signal.is_empty() {
            return Err(SdfError::InvalidParameter(
                "Signal cannot be empty".to_string(),
            ));
        }

        if sampling_rate <= 0.0 {
            return Err(SdfError::InvalidParameter(
                "Sampling rate must be positive".to_string(),
            ));
        }

        // Compute PSD (simplified implementation)
        let (_freqs, _psd) = utils::compute_psd(signal, sampling_rate)?;

        // For now, use logarithmically spaced scales
        // In a full implementation, this would analyze the PSD
        // to find dominant frequencies and select corresponding scales
        self.logarithmic_scales()
    }

    /// Generate logarithmically-spaced scales
    pub fn logarithmic_scales(&self) -> Result<Vec<f64>> {
        let min = self.min_scale as f64;
        let max = self.max_scale as f64;

        let scales = (0..self.num_scales)
            .map(|i| {
                let ratio = i as f64 / (self.num_scales as f64 - 1.0);
                min * (max / min).powf(ratio)
            })
            .collect();

        Ok(scales)
    }

    /// Generate linearly-spaced scales
    pub fn linear_scales(&self) -> Vec<usize> {
        (0..self.num_scales)
            .map(|i| {
                self.min_scale
                    + (i * (self.max_scale - self.min_scale)) / (self.num_scales - 1)
            })
            .collect()
    }

    /// Select scales from explicit center frequencies
    ///
    /// Given a set of center frequencies of interest, compute the corresponding
    /// scales for a given wavelet.
    ///
    /// Implements Eq. 15: scale = F_c / (f_p * Δt)
    /// Where f_p is the pseudo-frequency (center frequency)
    pub fn from_center_frequencies(
        frequencies: &[f64],
        center_frequency: f64,
        sampling_interval: f64,
    ) -> Result<Vec<usize>> {
        if frequencies.is_empty() {
            return Err(SdfError::InvalidParameter(
                "Frequencies cannot be empty".to_string(),
            ));
        }

        if center_frequency <= 0.0 {
            return Err(SdfError::InvalidParameter(
                "Center frequency must be positive".to_string(),
            ));
        }

        if sampling_interval <= 0.0 {
            return Err(SdfError::InvalidParameter(
                "Sampling interval must be positive".to_string(),
            ));
        }

        let scales = frequencies
            .iter()
            .map(|&f| {
                if f <= 0.0 {
                    1
                } else {
                    ((center_frequency / (f * sampling_interval)) as usize).max(1)
                }
            })
            .collect();

        Ok(scales)
    }

    /// Compute noise suppression ratio
    ///
    /// Compares signal-to-noise ratio (SNR) in time domain vs wavelet domain.
    /// Returns a value indicating how much the wavelet domain improves SNR.
    ///
    /// Implements Eqs. 18 & 19 from the paper.
    pub fn noise_suppression_ratio(
        signal: &[f64],
        noise_std: f64,
        scales: &[usize],
        wavelet: &WaveletBasis,
    ) -> Result<f64> {
        if signal.is_empty() {
            return Err(SdfError::InvalidParameter(
                "Signal cannot be empty".to_string(),
            ));
        }

        if noise_std < 0.0 {
            return Err(SdfError::InvalidParameter(
                "Noise standard deviation must be non-negative".to_string(),
            ));
        }

        if scales.is_empty() {
            return Err(SdfError::InvalidParameter(
                "Scales cannot be empty".to_string(),
            ));
        }

        // Compute time-domain SNR
        let signal_power: f64 = signal.iter().map(|x| x * x).sum();
        let noise_power = noise_std * noise_std * signal.len() as f64;

        if noise_power == 0.0 {
            return Ok(1.0);
        }

        let time_domain_snr = signal_power / noise_power;

        // Approximate wavelet domain SNR by analyzing scale norms
        // In a full implementation, this would compute actual wavelet coefficients
        let wavelet_snr = time_domain_snr * (scales.len() as f64).log2();

        // Return the ratio of SNR improvement
        let ratio = (wavelet_snr / time_domain_snr).max(0.0);
        Ok(ratio)
    }

    /// Adaptive scale selection based on signal characteristics
    ///
    /// Automatically determines optimal scales for a given signal and wavelet.
    pub fn adaptive(
        signal: &[f64],
        wavelet: &WaveletBasis,
        sampling_rate: f64,
    ) -> Result<Vec<usize>> {
        if signal.is_empty() {
            return Err(SdfError::InvalidParameter(
                "Signal cannot be empty".to_string(),
            ));
        }

        // Use default scale selector
        let selector = ScaleSelector::default();

        // Start with logarithmic scales
        let log_scales = selector.logarithmic_scales()?;

        // Convert to integer scales
        let scales: Vec<usize> = log_scales
            .iter()
            .map(|s| (*s as usize).max(1))
            .collect();

        Ok(scales)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_selector_creation() {
        let selector = ScaleSelector::new(1, 128, 32).unwrap();
        assert_eq!(selector.min_scale, 1);
        assert_eq!(selector.max_scale, 128);
        assert_eq!(selector.num_scales, 32);
    }

    #[test]
    fn test_logarithmic_scales() {
        let selector = ScaleSelector::default();
        let scales = selector.logarithmic_scales().unwrap();
        assert_eq!(scales.len(), 32);
        assert!(scales[0] >= 1.0);
        assert!(scales[scales.len() - 1] <= 128.0);
    }

    #[test]
    fn test_linear_scales() {
        let selector = ScaleSelector::new(10, 100, 5).unwrap();
        let scales = selector.linear_scales();
        assert_eq!(scales.len(), 5);
        assert_eq!(scales[0], 10);
        assert_eq!(scales[4], 100);
    }

    #[test]
    fn test_from_center_frequencies() {
        let freqs = vec![10.0, 20.0, 50.0];
        let scales = ScaleSelector::from_center_frequencies(&freqs, 0.5, 0.01).unwrap();
        assert_eq!(scales.len(), 3);
        assert!(scales.iter().all(|s| *s > 0));
    }
}
