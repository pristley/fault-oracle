//! Integration tests
//!
//! Basic tests for the complete SDF pipeline

#[cfg(test)]
mod tests {
    use sdf::wavelets::{WaveletBasis, WaveletTransform, ScaleSelector};
    use sdf::partitioning::types::Partition;
    use sdf::symbolic::{Alphabet, SymbolicEncoder};
    use sdf::markov::DMarkovMachine;
    use sdf::anomaly::measures::AnomalyMeasure;

    /// Generate synthetic test signal
    fn generate_test_signal(size: usize) -> Vec<f64> {
        (0..size)
            .map(|i| {
                let t = i as f64 / 100.0;
                (t * std::f64::consts::PI * 2.0).sin() + 0.1 * t
            })
            .collect()
    }

    #[test]
    fn test_wavelet_transform_basic() {
        let signal = generate_test_signal(50);
        let scales = vec![1, 2, 4, 8];
        let wavelet = WaveletBasis::Morlet;

        let cwt = WaveletTransform::continuous(&signal, &scales, wavelet)
            .expect("CWT should succeed");

        assert_eq!(cwt.coefficients.nrows(), signal.len());
        assert_eq!(cwt.coefficients.ncols(), scales.len());

        let norms = cwt.compute_scale_norms();
        assert_eq!(norms.len(), scales.len());
        for norm in &norms {
            assert!(norm.is_finite() && *norm >= 0.0);
        }
    }

    #[test]
    fn test_scale_selector_default() {
        let selector = ScaleSelector::default();
        let linear = selector.linear_scales();
        assert!(!linear.is_empty());
        
        let log_result = selector.logarithmic_scales();
        assert!(log_result.is_ok());
        let log = log_result.unwrap();
        assert!(!log.is_empty());
    }

    #[test]
    fn test_partition_creation() {
        let partition = Partition::new(vec![0.0, 0.5, 1.0])
            .expect("Partition creation should succeed");
        assert_eq!(partition.num_regions, 4);
    }

    #[test]
    fn test_alphabet_creation() {
        for size in &[2, 4, 8, 16] {
            let alphabet = Alphabet::new(*size)
                .expect("Should create alphabet");
            assert_eq!(alphabet.size(), *size);

            // Test symbol retrieval
            for i in 0..*size.min(&26) {
                let symbol = alphabet.symbol(i);
                assert!(symbol.is_ok());
            }
        }
    }

    #[test]
    fn test_symbolic_encoder() {
        let alphabet = Alphabet::new(4)
            .expect("Should create alphabet");
        let partition = Partition::new(vec![0.0, 0.5, 1.0])
            .expect("Partition should be created");
        
        let encoder = SymbolicEncoder::new(alphabet, partition)
            .expect("Encoder should be created");
        
        let signal = vec![0.1, 0.3, 0.7, 0.9];
        let encoded = encoder.encode_timeseries(&signal)
            .expect("Encoding should succeed");
        
        assert_eq!(encoded.len(), signal.len());
    }

    #[test]
    fn test_d_markov_machine() {
        let symbols: Vec<char> = "abaababbaabaabb".chars().collect();
        let depth = 2;
        let alphabet_size = 2;

        let machine = DMarkovMachine::new(&symbols, depth, alphabet_size)
            .expect("D-Markov creation should succeed");

        assert_eq!(machine.order(), depth);
        assert_eq!(machine.n_states(), 4); // 2^2 = 4 states

        let matrix = machine.transition_matrix();
        assert_eq!(matrix.nrows(), 4);
        assert_eq!(matrix.ncols(), 4);
    }

    #[test]
    fn test_entropy_rate_computation() {
        let symbols: Vec<char> = "ababab".chars().collect();
        let machine = DMarkovMachine::new(&symbols, 1, 2)
            .expect("Should create machine");
        
        let entropy = machine.entropy_rate();
        assert!(entropy >= 0.0 && entropy <= 1.0);
    }

    #[test]
    fn test_anomaly_measure_norm() {
        let nominal = vec![0.25, 0.25, 0.25, 0.25];
        let normal = vec![0.3, 0.25, 0.25, 0.2];

        let score = AnomalyMeasure::compute_norm_based(&normal, &nominal, sdf::anomaly::measures::NormType::L2)
            .expect("Should compute");
        
        assert!(score.is_finite() && score >= 0.0);
    }

    #[test]
    fn test_anomaly_measure_angle() {
        let nominal = vec![0.25, 0.25, 0.25, 0.25];
        let normal = vec![0.3, 0.25, 0.25, 0.2];

        let score = AnomalyMeasure::compute_angle(&normal, &nominal)
            .expect("Should compute");
        
        assert!(score.is_finite() && score >= 0.0);
    }

    #[test]
    fn test_complete_pipeline() {
        let signal = generate_test_signal(100);

        // Wavelet transform
        let cwt = WaveletTransform::continuous(&signal, &[1, 2, 4], WaveletBasis::Morlet)
            .expect("CWT should succeed");

        // Extract features
        let norms = cwt.compute_scale_norms();
        assert_eq!(norms.len(), 3);

        // Create encoder
        let alphabet = Alphabet::new(4).expect("Alphabet creation");
        let partition = Partition::new(vec![0.0, 0.5, 1.0]).expect("Partition creation");
        let encoder = SymbolicEncoder::new(alphabet, partition).expect("Encoder creation");

        // Encode part of the signal
        let encoded = encoder.encode_timeseries(&signal[0..50])
            .expect("Encoding should succeed");
        assert_eq!(encoded.len(), 50);

        // Create D-Markov machine
        let symbols: Vec<char> = encoded.into_iter().take(20).collect();
        let machine = DMarkovMachine::new(&symbols, 1, 4)
            .expect("Machine creation should succeed");
        assert!(machine.entropy_rate() >= 0.0);
    }
}
