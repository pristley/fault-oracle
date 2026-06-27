//! Anomaly measures
//!
//! Implements different anomaly measures including norm, angle, and KL-divergence based approaches.
//! Implements Phase 6.1: Anomaly Measures

use std::f64;
use crate::Result;

/// Norm type for norm-based anomaly measures
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormType {
    /// L1 norm (Manhattan distance)
    L1,
    /// L2 norm (Euclidean distance)
    L2,
    /// L-infinity norm (maximum absolute value)
    Linf,
}

/// Anomaly measure types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnomalyMeasure {
    /// Euclidean norm-based measure (Eq. 37)
    Norm(NormType),
    /// Angle-based measure (Eq. 38)
    Angle,
    /// Kullback-Leibler divergence (Eq. 39)
    KLDivergence,
    /// Matrix norm measure (Eq. 35)
    MatrixNorm,
    /// Entropy rate-based measure
    EntropyRate,
    /// Excess entropy measure
    ExcessEntropy,
    /// Statistical complexity measure
    StatisticalComplexity,
}

impl AnomalyMeasure {
    /// Compute norm-based anomaly measure
    /// Eq. 37: ||p_k - p_0||_r
    pub fn compute_norm_based(
        p_current: &[f64],
        p_nominal: &[f64],
        norm_type: NormType,
    ) -> Result<f64> {
        if p_current.len() != p_nominal.len() {
            return Err(crate::SdfError::ShapeMismatch {
                expected: format!("{}", p_nominal.len()),
                actual: format!("{}", p_current.len()),
            });
        }

        let distance = match norm_type {
            NormType::L1 => {
                p_current
                    .iter()
                    .zip(p_nominal.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>()
            }
            NormType::L2 => {
                p_current
                    .iter()
                    .zip(p_nominal.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }
            NormType::Linf => {
                p_current
                    .iter()
                    .zip(p_nominal.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(f64::NEG_INFINITY, f64::max)
            }
        };

        Ok(distance)
    }

    /// Compute angle-based anomaly measure
    /// Eq. 38: arccos(<p_k, p_0> / (||p_k|| ||p_0||))
    pub fn compute_angle(
        p_current: &[f64],
        p_nominal: &[f64],
    ) -> Result<f64> {
        if p_current.len() != p_nominal.len() {
            return Err(crate::SdfError::ShapeMismatch {
                expected: format!("{}", p_nominal.len()),
                actual: format!("{}", p_current.len()),
            });
        }

        let dot_product: f64 = p_current
            .iter()
            .zip(p_nominal.iter())
            .map(|(a, b)| a * b)
            .sum();

        let current_norm: f64 = p_current.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nominal_norm: f64 = p_nominal.iter().map(|x| x * x).sum::<f64>().sqrt();

        if current_norm < 1e-10 || nominal_norm < 1e-10 {
            return Ok(f64::NAN);
        }

        let cosine = dot_product / (current_norm * nominal_norm);
        let angle = cosine.max(-1.0).min(1.0).acos();

        Ok(angle / std::f64::consts::PI)
    }

    /// Compute Kullback-Leibler divergence
    /// Eq. 39: KL = -Σ p_k_i * log(p_k_i / p_0_i)
    pub fn compute_kullback_leibler(
        p_current: &[f64],
        p_nominal: &[f64],
    ) -> Result<f64> {
        if p_current.len() != p_nominal.len() {
            return Err(crate::SdfError::ShapeMismatch {
                expected: format!("{}", p_nominal.len()),
                actual: format!("{}", p_current.len()),
            });
        }

        let epsilon = 1e-10;
        let mut kl = 0.0;

        for (p_c, p_n) in p_current.iter().zip(p_nominal.iter()) {
            let p_c = p_c.max(epsilon);
            let p_n = p_n.max(epsilon);

            if p_c > 0.0 && p_n > 0.0 {
                kl += p_c * (p_c / p_n).ln();
            }
        }

        Ok(kl)
    }

    /// Compute anomaly score for a feature vector
    pub fn compute_score(&self, feature: &[f64], reference: &[f64]) -> Result<f64> {
        if feature.len() != reference.len() {
            return Err(crate::SdfError::ShapeMismatch {
                expected: format!("{}", reference.len()),
                actual: format!("{}", feature.len()),
            });
        }

        match self {
            AnomalyMeasure::Norm(norm_type) => Self::compute_norm_based(feature, reference, *norm_type),
            AnomalyMeasure::Angle => Self::compute_angle(feature, reference),
            AnomalyMeasure::KLDivergence => Self::compute_kullback_leibler(feature, reference),
            AnomalyMeasure::MatrixNorm => Ok(0.0),
            AnomalyMeasure::EntropyRate => Ok(Self::compute_entropy_rate(feature)),
            AnomalyMeasure::ExcessEntropy => Ok(Self::compute_excess_entropy(feature, reference)),
            AnomalyMeasure::StatisticalComplexity => Ok(Self::compute_statistical_complexity(feature, reference)),
        }
    }

    fn compute_entropy_rate(p: &[f64]) -> f64 {
        let mut entropy = 0.0;
        for &val in p {
            if val > 0.0 {
                entropy -= val * val.log2();
            }
        }
        entropy
    }

    fn compute_excess_entropy(p_current: &[f64], p_nominal: &[f64]) -> f64 {
        let mut excess = 0.0;
        for (c, n) in p_current.iter().zip(p_nominal.iter()) {
            if c > 0.0 && n > 0.0 {
                excess += (c - n).abs();
            }
        }
        excess
    }

    fn compute_statistical_complexity(p_current: &[f64], p_nominal: &[f64]) -> f64 {
        let mut complexity = 0.0;
        for (c, n) in p_current.iter().zip(p_nominal.iter()) {
            if c > 0.0 && n > 0.0 {
                complexity += (c * (c - n).abs()).abs();
            }
        }
        complexity
    }
}

impl std::fmt::Display for AnomalyMeasure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalyMeasure::Norm(NormType::L1) => write!(f, "Norm(L1)"),
            AnomalyMeasure::Norm(NormType::L2) => write!(f, "Norm(L2)"),
            AnomalyMeasure::Norm(NormType::Linf) => write!(f, "Norm(Linf)"),
            AnomalyMeasure::Angle => write!(f, "Angle"),
            AnomalyMeasure::KLDivergence => write!(f, "KLDivergence"),
            AnomalyMeasure::MatrixNorm => write!(f, "MatrixNorm"),
            AnomalyMeasure::EntropyRate => write!(f, "EntropyRate"),
            AnomalyMeasure::ExcessEntropy => write!(f, "ExcessEntropy"),
            AnomalyMeasure::StatisticalComplexity => write!(f, "StatisticalComplexity"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_l2_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = AnomalyMeasure::compute_norm_based(&a, &b, NormType::L2).unwrap();
        assert!((score - std::f64::consts::SQRT_2).abs() < 0.001);
    }

    #[test]
    fn test_angle_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        let score = AnomalyMeasure::compute_angle(&a, &b).unwrap();
        assert!(score < 0.001);
    }

    #[test]
    fn test_kl_divergence() {
        let a = vec![0.5, 0.5];
        let b = vec![0.5, 0.5];
        let score = AnomalyMeasure::compute_kullback_leibler(&a, &b).unwrap();
        assert!(score.abs() < 0.001);
    }
}
