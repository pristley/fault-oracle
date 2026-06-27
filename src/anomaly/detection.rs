//! Anomaly detection algorithms
//!
//! Implements detection algorithms and thresholding strategies.
//! Implements Phase 6.2: Anomaly Detection Algorithm

use crate::Result;
use super::measures::AnomalyMeasure;
use ndarray::Array2;

/// Pattern vector for anomaly detection
#[derive(Debug, Clone)]
pub struct PatternVector {
    /// State probability vector
    pub state_probabilities: Vec<f64>,
    /// Timestamp of measurement
    pub timestamp: usize,
    /// Transition matrix (optional)
    pub transition_matrix: Option<Array2<f64>>,
}

impl PatternVector {
    /// Create a new pattern vector
    pub fn new(state_probabilities: Vec<f64>, timestamp: usize) -> Result<Self> {
        if state_probabilities.is_empty() {
            return Err(crate::SdfError::InvalidParameter(
                "State probabilities cannot be empty".to_string(),
            ));
        }

        Ok(PatternVector {
            state_probabilities,
            timestamp,
            transition_matrix: None,
        })
    }

    /// Create with transition matrix
    pub fn with_matrix(
        state_probabilities: Vec<f64>,
        timestamp: usize,
        transition_matrix: Array2<f64>,
    ) -> Result<Self> {
        if state_probabilities.is_empty() {
            return Err(crate::SdfError::InvalidParameter(
                "State probabilities cannot be empty".to_string(),
            ));
        }

        Ok(PatternVector {
            state_probabilities,
            timestamp,
            transition_matrix: Some(transition_matrix),
        })
    }
}

/// Anomaly record
#[derive(Debug, Clone)]
pub struct AnomalyRecord {
    /// Measurement timestamp
    pub timestamp: usize,
    /// Anomaly measure value
    pub measure_value: f64,
    /// State probabilities at this time
    pub state_probabilities: Vec<f64>,
    /// Whether marked as anomalous
    pub is_anomalous: bool,
}

/// Anomaly detector
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Reference pattern from nominal data
    nominal_pattern: PatternVector,
    /// Anomaly measure to use
    anomaly_measure: AnomalyMeasure,
    /// Detection threshold
    threshold: f64,
    /// History of detections
    history: Vec<AnomalyRecord>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    ///
    /// Implements Phase 6.2, Step 1: new
    /// Forward problem: characterize nominal behavior
    pub fn new(
        nominal_pattern: PatternVector,
        anomaly_measure: AnomalyMeasure,
        threshold: f64,
    ) -> Result<Self> {
        if threshold < 0.0 {
            return Err(crate::SdfError::InvalidParameter(
                "Threshold must be non-negative".to_string(),
            ));
        }

        Ok(AnomalyDetector {
            nominal_pattern,
            anomaly_measure,
            threshold,
            history: Vec::new(),
        })
    }

    /// Detect anomaly at a single epoch
    ///
    /// Implements Phase 6.2, Step 2: detect_at_epoch
    /// Inverse problem: detect anomaly in new data
    /// Returns anomaly measure value
    pub fn detect_at_epoch(
        &mut self,
        current_pattern: &PatternVector,
    ) -> Result<f64> {
        // Compute anomaly measure value
        let measure_value = self.anomaly_measure.compute_score(
            &current_pattern.state_probabilities,
            &self.nominal_pattern.state_probabilities,
        )?;

        // Record detection
        let is_anomalous = measure_value > self.threshold;
        let record = AnomalyRecord {
            timestamp: current_pattern.timestamp,
            measure_value,
            state_probabilities: current_pattern.state_probabilities.clone(),
            is_anomalous,
        };

        self.history.push(record);

        Ok(measure_value)
    }

    /// Detect bifurcation (phase transition)
    ///
    /// Implements Phase 6.2, Step 3: detect_bifurcation
    /// Detects change in anomaly measure slope
    /// Indicates transition between phases (Section 10.2)
    pub fn detect_bifurcation(&self) -> Option<usize> {
        if self.history.len() < 3 {
            return None;
        }

        // Compute slopes between consecutive measurements
        let mut max_slope_change = 0.0;
        let mut bifurcation_time = None;

        for i in 1..(self.history.len() - 1) {
            let slope1 = self.history[i].measure_value - self.history[i - 1].measure_value;
            let slope2 = self.history[i + 1].measure_value - self.history[i].measure_value;

            let slope_change = (slope2 - slope1).abs();

            if slope_change > max_slope_change {
                max_slope_change = slope_change;
                bifurcation_time = Some(self.history[i].timestamp);
            }
        }

        // Return bifurcation only if significant slope change detected
        if max_slope_change > 0.1 {
            bifurcation_time
        } else {
            None
        }
    }

    /// Check if latest measurement is anomalous
    ///
    /// Implements Phase 6.2, Step 4: is_anomalous
    /// Compare latest anomaly measure to threshold
    pub fn is_anomalous(&self) -> bool {
        if let Some(last_record) = self.history.last() {
            last_record.is_anomalous
        } else {
            false
        }
    }

    /// Get the latest anomaly measure value
    pub fn latest_score(&self) -> Option<f64> {
        self.history.last().map(|r| r.measure_value)
    }

    /// Set a new threshold
    pub fn set_threshold(&mut self, threshold: f64) -> Result<()> {
        if threshold < 0.0 {
            return Err(crate::SdfError::InvalidParameter(
                "Threshold must be non-negative".to_string(),
            ));
        }
        self.threshold = threshold;
        Ok(())
    }

    /// Get the current threshold
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get the anomaly measure
    pub fn measure(&self) -> AnomalyMeasure {
        self.anomaly_measure
    }

    /// Get detection history
    pub fn history(&self) -> &[AnomalyRecord] {
        &self.history
    }

    /// Adaptive threshold based on statistics
    ///
    /// Sets threshold to mean + k * std_dev of training scores
    pub fn set_adaptive_threshold(
        &mut self,
        training_patterns: &[PatternVector],
        k: f64,
    ) -> Result<()> {
        if training_patterns.is_empty() {
            return Err(crate::SdfError::InvalidParameter(
                "Training patterns cannot be empty".to_string(),
            ));
        }

        // Compute scores for training data
        let scores: Vec<f64> = training_patterns
            .iter()
            .filter_map(|pattern| {
                self.anomaly_measure
                    .compute_score(
                        &pattern.state_probabilities,
                        &self.nominal_pattern.state_probabilities,
                    )
                    .ok()
            })
            .collect();

        if scores.is_empty() {
            return Err(crate::SdfError::ComputationError(
                "Could not compute scores for training data".to_string(),
            ));
        }

        // Compute mean
        let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;

        // Compute std dev
        let variance: f64 = scores
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>()
            / scores.len() as f64;
        let std_dev = variance.sqrt();

        self.threshold = mean + k * std_dev;
        Ok(())
    }

    /// Get anomaly detection summary
    pub fn summary(&self) -> AnomalyDetectionSummary {
        let total = self.history.len();
        let anomalies = self.history.iter().filter(|r| r.is_anomalous).count();
        let anomaly_rate = if total > 0 {
            anomalies as f64 / total as f64
        } else {
            0.0
        };

        let scores: Vec<f64> = self.history.iter().map(|r| r.measure_value).collect();
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_score = if !scores.is_empty() {
            scores.iter().sum::<f64>() / scores.len() as f64
        } else {
            0.0
        };

        AnomalyDetectionSummary {
            total_measurements: total,
            anomaly_count: anomalies,
            anomaly_rate,
            min_score,
            max_score,
            mean_score,
            threshold: self.threshold,
        }
    }
}

/// Summary statistics for anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyDetectionSummary {
    pub total_measurements: usize,
    pub anomaly_count: usize,
    pub anomaly_rate: f64,
    pub min_score: f64,
    pub max_score: f64,
    pub mean_score: f64,
    pub threshold: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detector_creation() {
        let pattern = PatternVector::new(vec![0.5, 0.5], 0).unwrap();
        let detector = AnomalyDetector::new(
            pattern,
            AnomalyMeasure::Norm(super::super::measures::NormType::L2),
            0.5,
        )
        .unwrap();

        assert_eq!(detector.threshold(), 0.5);
    }

    #[test]
    fn test_anomaly_detection() {
        let nominal = PatternVector::new(vec![0.5, 0.5], 0).unwrap();
        let mut detector = AnomalyDetector::new(
            nominal,
            AnomalyMeasure::Norm(super::super::measures::NormType::L2),
            0.3,
        )
        .unwrap();

        let normal = PatternVector::new(vec![0.45, 0.55], 1).unwrap();
        let anomalous = PatternVector::new(vec![1.0, 1.0], 2).unwrap();

        let _ = detector.detect_at_epoch(&normal);
        let _ = detector.detect_at_epoch(&anomalous);

        let summary = detector.summary();
        assert_eq!(summary.total_measurements, 2);
    }
}
