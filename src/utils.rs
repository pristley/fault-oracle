//! Utility functions and helpers

use crate::Result;

/// Compute the Power Spectral Density (PSD) of a signal
pub fn compute_psd(signal: &[f64], sampling_rate: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if signal.is_empty() {
        return Err(crate::SdfError::InvalidParameter(
            "Signal cannot be empty".to_string(),
        ));
    }

    // Simple FFT-based PSD computation
    let n = signal.len();
    let freqs: Vec<f64> = (0..n / 2)
        .map(|k| (k as f64) * sampling_rate / (n as f64))
        .collect();

    // Placeholder for actual FFT computation
    let psd = vec![0.0; freqs.len()];

    Ok((freqs, psd))
}

/// Normalize a vector
pub fn normalize(data: &mut [f64]) -> Result<()> {
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max).abs();
    if max == 0.0 {
        return Err(crate::SdfError::ComputationError(
            "Cannot normalize zero vector".to_string(),
        ));
    }

    for val in data.iter_mut() {
        *val /= max;
    }

    Ok(())
}

/// Compute standard deviation
pub fn compute_std(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let variance: f64 = data
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / data.len() as f64;

    variance.sqrt()
}

/// Compute mean
pub fn compute_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}
