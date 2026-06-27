//! Integration tests
//!
//! Comprehensive tests for the complete SDF pipeline including:
//! - Phase 8.1: Wavelet transforms, partitioning, D-Markov machines, anomaly detection
//! - Validation of paper equations and algorithms

#[cfg(test)]
mod tests {
    use symbolic_dynamic_filtering::wavelets::{WaveletBasis, WaveletTransform, ScaleSelector};
    use symbolic_dynamic_filtering::partitioning::types::Partition;
    use symbolic_dynamic_filtering::symbolic::{Alphabet, SymbolicEncoder, SymbolicDynamics, encoding::SymbolicEncoder as Encoder};
    use symbolic_dynamic_filtering::markov::DMarkovMachine;
    use symbolic_dynamic_filtering::anomaly::{AnomalyDetector, AnomalyMeasure, detection::{PatternVector, AnomalyDetector as NewAnomalyDetector}};
    use symbolic_dynamic_filtering::anomaly::measures::NormType;

    /// Generate synthetic test signal with known properties
    fn generate_test_signal(size: usize) -> Vec<f64> {
        (0..size)
            .map(|i| {
                let t = i as f64 / 100.0;
                (t * std::f64::consts::PI * 2.0).sin() + 0.1 * t
            })
            .collect()
    }

    /// Generate a Duffing oscillator signal (electronic circuit example)
    fn generate_duffing_signal(beta: f64, a: f64, omega: f64, steps: usize) -> Vec<f64> {
        let mut signal = Vec::with_capacity(steps);
        let dt = 0.01;
        let mut y = 0.0;
        let mut dy = 0.0;

        for i in 0..steps {
            let t = i as f64 * dt;
            let forcing = a * (omega * t).cos();
            let ddy = forcing - beta * dy - y - y * y * y;
            dy += ddy * dt;
            y += dy * dt;
            signal.push(y);
        }
        signal
    }

    /// Generate fatigue damage signal with increasing amplitude
    fn generate_fatigue_signal(cycles: usize, damage_level: f64) -> Vec<f64> {
        let mut signal = Vec::new();
        let base_freq = 1000.0;
        let dt = 1.0 / (10.0 * base_freq);

        for i in 0..(cycles * 100) {
            let t = i as f64 * dt;
            let amplitude = 1.0 + damage_level * (t / (cycles as f64 * 0.1));
            let sample = amplitude * (2.0 * std::f64::consts::PI * base_freq * t).sin();
            signal.push(sample);
        }
        signal
    }

    /// Phase 8.1: Test 1 - Complete SDF Pipeline
    /// Validates all stages from signal to anomaly detection
    #[test]
    fn test_complete_sdf_pipeline() {
        // Create test signal
        let signal = generate_test_signal(50);

        // Stage 1: Wavelet Transform (Eq. 9)
        let wavelet = WaveletBasis::Morlet;
        let scales = vec![1, 2, 4, 8];

        let cwt = WaveletTransform::continuous(&signal, &scales, wavelet)
            .expect("CWT should succeed");

        // Validate coefficient matrix dimensions
        assert_eq!(cwt.coefficients.nrows(), signal.len());
        assert_eq!(cwt.coefficients.ncols(), scales.len());

        // Stage 2: Feature extraction (Eq. 15 - pseudo-frequencies)
        let freqs = cwt.compute_pseudo_frequencies(0.01);
        assert_eq!(freqs.len(), scales.len());
        // Smaller scales should correspond to higher frequencies
        for i in 0..freqs.len() - 1 {
            assert!(freqs[i] > freqs[i + 1]);
        }

        // Stage 3: Signal quantization
        let partition = Partition::new(vec![0.0, 0.5, 1.0])
            .expect("Partition creation should succeed");
        let alphabet = Alphabet::new(4).expect("Alphabet creation should succeed");

        // Stage 4: Symbolic encoding
        let scale_series = cwt.arrange_scale_series();
        let encoder = SymbolicEncoder::new(alphabet.clone(), partition)
            .expect("Encoder creation should succeed");

        let encoded: Vec<char> = scale_series
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let idx = (val.abs() as usize) % alphabet.size();
                alphabet.symbol(idx).unwrap().clone()
            })
            .collect();

        assert_eq!(encoded.len(), scale_series.len());

        // Stage 5: Symbolic dynamics analysis
        let dynamics = SymbolicDynamics::new(encoded.clone())
            .expect("Dynamics creation should succeed");

        let entropy = dynamics.shannon_entropy();
        assert!(entropy >= 0.0 && entropy <= 4.0);

        // Stage 6: Anomaly detection (Eq. 37-39)
        let normal_features = vec![0.1, 0.15, 0.12, 0.08];
        let anomalous_features = vec![0.8, 0.85, 0.82, 0.78];

        let detector = AnomalyDetector::new(AnomalyMeasure::Norm, 0.3)
            .expect("Detector creation should succeed");

        let is_anom = detector
            .is_anomalous(&anomalous_features, &normal_features)
            .expect("Anomaly detection should succeed");

        assert!(is_anom);
    }

    /// Phase 8.1: Test 2 - Wavelet Transform Validation (Eq. 9)
    #[test]
    fn test_wavelet_transform_equation_9() {
        let signal = generate_test_signal(100);
        let scales = vec![1, 2, 4, 8, 16];
        let wavelets = vec![
            WaveletBasis::Haar,
            WaveletBasis::Morlet,
            WaveletBasis::MexicanHat,
        ];

        for wavelet in wavelets {
            let cwt = WaveletTransform::continuous(&signal, &scales, wavelet)
                .expect("CWT should compute");

            // Verify shape
            assert_eq!(cwt.coefficients.nrows(), signal.len());
            assert_eq!(cwt.coefficients.ncols(), scales.len());

            // Verify scale norms (lower scales should have higher energy for typical signals)
            let norms = cwt.compute_scale_norms();
            assert_eq!(norms.len(), scales.len());
            for norm in &norms {
                assert!(norm.is_finite() && norm >= 0.0);
            }
        }
    }

    /// Phase 8.1: Test 3 - Scale Selection Validation
    #[test]
    fn test_scale_selection_strategies() {
        let signal = generate_test_signal(200);
        let sampling_rate = 100.0;

        // Logarithmic scale selection
        let selector = ScaleSelector::logarithmic_scales(1, 128, 5);
        assert_eq!(selector.scales.len(), 5);

        // Linear scale selection
        let selector = ScaleSelector::linear_scales(1, 64, 8);
        assert_eq!(selector.scales.len(), 8);

        // Adaptive selection
        let selector = ScaleSelector::adaptive(&signal, WaveletBasis::Morlet, sampling_rate)
            .expect("Adaptive selection should succeed");
        assert!(selector.scales.len() > 0);
    }

    /// Phase 8.1: Test 4 - D-Markov Machine Construction (Eq. 27-28)
    #[test]
    fn test_d_markov_machine_construction() {
        let symbols: Vec<char> = "abaababbaabaabb".chars().collect();
        let depth = 2;
        let alphabet_size = 2;

        let machine = DMarkovMachine::new(&symbols, depth, alphabet_size)
            .expect("D-Markov creation should succeed");

        assert_eq!(machine.order(), depth);
        assert_eq!(machine.n_states(), 4); // 2^2 = 4 states

        // Verify transition matrix
        let matrix = machine.transition_matrix();
        assert_eq!(matrix.nrows(), 4);
        assert_eq!(matrix.ncols(), 4);

        // Check that rows sum to approximately 1 (stochastic matrix)
        for i in 0..4 {
            let row_sum: f64 = (0..4).map(|j| matrix[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6 || row_sum == 0.0);
        }
    }

    /// Phase 8.1: Test 5 - Entropy Rate Computation (Eq. 29)
    #[test]
    fn test_entropy_rate_computation() {
        // Highly ordered sequence (low entropy)
        let ordered_symbols: Vec<char> = "ababab".chars().collect();
        let machine_ordered = DMarkovMachine::new(&ordered_symbols, 1, 2)
            .expect("Should create machine");
        let entropy_ordered = machine_ordered.entropy_rate();

        // Random sequence (high entropy)
        let random_symbols: Vec<char> = "aabbbaab".chars().collect();
        let machine_random = DMarkovMachine::new(&random_symbols, 1, 2)
            .expect("Should create machine");
        let entropy_random = machine_random.entropy_rate();

        // Both entropies should be non-negative and <= log2(alphabet_size)
        assert!(entropy_ordered >= 0.0 && entropy_ordered <= 1.0);
        assert!(entropy_random >= 0.0 && entropy_random <= 1.0);
    }

    /// Phase 8.1: Test 6 - Anomaly Measure Validation (Eq. 37-39)
    #[test]
    fn test_anomaly_measures_equation_37_39() {
        let nominal = vec![0.25, 0.25, 0.25, 0.25];
        let normal = vec![0.3, 0.25, 0.25, 0.2];
        let anomalous = vec![0.9, 0.05, 0.03, 0.02];

        // L2 norm (Eq. 37)
        let norm_l2_normal = AnomalyMeasure::compute_norm_based(&normal, &nominal, NormType::L2)
            .expect("Should compute");
        let norm_l2_anomalous = AnomalyMeasure::compute_norm_based(&anomalous, &nominal, NormType::L2)
            .expect("Should compute");

        // Normal should have lower norm than anomalous
        assert!(norm_l2_anomalous > norm_l2_normal);

        // Angle measure (Eq. 38)
        let angle_normal = AnomalyMeasure::compute_angle(&normal, &nominal)
            .expect("Should compute");
        let angle_anomalous = AnomalyMeasure::compute_angle(&anomalous, &nominal)
            .expect("Should compute");

        assert!(angle_anomalous > angle_normal);

        // KL divergence (Eq. 39)
        let kl_normal = AnomalyMeasure::compute_kullback_leibler(&normal, &nominal)
            .expect("Should compute");
        let kl_anomalous = AnomalyMeasure::compute_kullback_leibler(&anomalous, &nominal)
            .expect("Should compute");

        assert!(kl_anomalous > kl_normal);
    }

    /// Phase 8.1: Test 7 - Electronic Circuit Example (Duffing oscillator)
    #[test]
    fn test_electronic_circuit_duffing_oscillator() {
        // Example 10.1 from paper: Duffing equation
        // d²y/dt² + β*dy/dt + y + y³ = A*cos(Ωt)

        // Nominal condition: β = 0.1
        let nominal_signal = generate_duffing_signal(0.1, 5.0, 1.0, 1000);
        let cwt_nominal = WaveletTransform::continuous(&nominal_signal, &[1, 2, 4, 8], WaveletBasis::Morlet)
            .expect("CWT should compute");
        let norms_nominal = cwt_nominal.compute_scale_norms();

        // Anomalous condition: β = 0.25
        let anomalous_signal = generate_duffing_signal(0.25, 5.0, 1.0, 1000);
        let cwt_anomalous = WaveletTransform::continuous(&anomalous_signal, &[1, 2, 4, 8], WaveletBasis::Morlet)
            .expect("CWT should compute");
        let norms_anomalous = cwt_anomalous.compute_scale_norms();

        // Compute anomaly measure
        let anomaly_score = AnomalyMeasure::compute_norm_based(&norms_anomalous, &norms_nominal, NormType::L2)
            .expect("Should compute");

        // Should detect significant difference
        assert!(anomaly_score > 0.05);
    }

    /// Phase 8.1: Test 8 - Fatigue Detection Example
    #[test]
    fn test_fatigue_crack_detection() {
        // Example 10.2 from paper: Ultrasonic fatigue monitoring

        // Nominal (no damage)
        let nominal_signal = generate_fatigue_signal(50, 0.0);
        let cwt_nominal = WaveletTransform::continuous(&nominal_signal, &[2, 4, 8], WaveletBasis::Gaussian(2))
            .expect("CWT should compute");
        let features_nominal = cwt_nominal.compute_scale_norms();

        // 45k cycles (visible crack)
        let damaged_signal = generate_fatigue_signal(50, 0.5);
        let cwt_damaged = WaveletTransform::continuous(&damaged_signal, &[2, 4, 8], WaveletBasis::Gaussian(2))
            .expect("CWT should compute");
        let features_damaged = cwt_damaged.compute_scale_norms();

        // Detect using multiple measures
        let angle_measure = AnomalyMeasure::compute_angle(&features_damaged, &features_nominal)
            .expect("Should compute");
        let norm_measure = AnomalyMeasure::compute_norm_based(&features_damaged, &features_nominal, NormType::L2)
            .expect("Should compute");

        // Both should detect the damage
        assert!(angle_measure > 0.1);
        assert!(norm_measure > 0.05);
    }

    /// Phase 8.1: Test 9 - Wavelet Scale Properties (Eq. 15)
    #[test]
    fn test_wavelet_scale_properties() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let scales = vec![1, 2, 4, 8];
        let wavelet = WaveletBasis::Haar;

        let cwt = WaveletTransform::continuous(&signal, &scales, wavelet)
            .expect("CWT should work");

        let norms = cwt.compute_scale_norms();
        assert_eq!(norms.len(), 4);

        // Test pseudo-frequency computation (Eq. 15)
        let freqs = cwt.compute_pseudo_frequencies(0.01);
        assert_eq!(freqs.len(), 4);
        // Smaller scales should give higher frequencies
        assert!(freqs[0] > freqs[3]);
        // All frequencies should be positive
        for freq in &freqs {
            assert!(freq > 0.0);
        }
    }

    /// Phase 8.1: Test 10 - Adaptive Threshold Setting
    #[test]
    fn test_anomaly_detection_adaptive_threshold() {
        let mut detector = AnomalyDetector::new(AnomalyMeasure::Angle, 0.5)
            .expect("Detector creation should succeed");

        assert_eq!(detector.threshold(), 0.5);

        detector.set_threshold(0.7).expect("Threshold setting should succeed");
        assert_eq!(detector.threshold(), 0.7);

        // Verify invalid thresholds are rejected
        assert!(detector.set_threshold(-0.5).is_err());
    }

    /// Phase 8.1: Test 11 - Scale Series Arrangement
    #[test]
    fn test_scale_series_arrangement() {
        let signal = vec![1.0, 2.0, 1.0];
        let scales = vec![1, 2, 4];
        let wavelet = WaveletBasis::Haar;

        let cwt = WaveletTransform::continuous(&signal, &scales, wavelet)
            .expect("CWT should work");

        let series = cwt.arrange_scale_series();
        // Series should be: [all coeff for scale 1, 2, 4, then 4, 2, 1]
        assert_eq!(series.len(), 2 * signal.len() * scales.len());
    }

    /// Phase 8.1: Test 12 - Symbol Alphabet Validation
    #[test]
    fn test_alphabet_sizes() {
        // Test various alphabet sizes
        for size in &[2, 4, 8, 16] {
            let alphabet = Alphabet::new(*size).expect("Should create alphabet");
            assert_eq!(alphabet.size(), *size);

            // Test symbol retrieval
            for i in 0..size.min(&26) {
                let symbol = alphabet.symbol(i);
                assert!(symbol.is_some());
            }
        }
    }
}
