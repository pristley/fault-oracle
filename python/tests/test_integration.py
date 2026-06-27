"""Phase 8.2: Integration tests for Symbolic Dynamic Filtering

Comprehensive Python tests validating:
- Complete SDF pipeline
- Electronic circuit example (Duffing oscillator)
- Fatigue damage detection
- Anomaly measure implementations (Eq. 37-39)
"""

import unittest
import numpy as np

try:
    from symbolic_dynamic_filtering import SDF
    BINDINGS_AVAILABLE = True
except ImportError:
    BINDINGS_AVAILABLE = False


class TestSDFPipeline(unittest.TestCase):
    """Test complete SDF pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        if not BINDINGS_AVAILABLE:
            self.skipTest("Native bindings not available")

    def generate_test_signal(self, size, noise_level=0.0):
        """Generate synthetic test signal"""
        t = np.linspace(0, 4 * np.pi, size)
        signal = np.sin(t) + 0.1 * np.cos(2 * t)
        if noise_level > 0:
            signal += noise_level * np.random.randn(size)
        return signal

    def test_sdf_initialization(self):
        """Test SDF object creation and configuration"""
        sdf = SDF(alphabet_size=8, depth=2, wavelet='gaus1')
        config = sdf.get_config()

        self.assertEqual(config['alphabet_size'], 8)
        self.assertEqual(config['depth'], 2)
        self.assertEqual(config['wavelet'], 'gaus1')
        self.assertFalse(config['is_trained'])

    def test_sdf_training(self):
        """Test forward problem: training on nominal data"""
        sdf = SDF(alphabet_size=8, depth=1)
        nominal_data = self.generate_test_signal(500)

        # Should not raise
        sdf.train(nominal_data)

        # Check training state
        self.assertTrue(sdf.get_config()['is_trained'])

    def test_sdf_detection(self):
        """Test inverse problem: anomaly detection"""
        sdf = SDF(alphabet_size=8, depth=1, anomaly_measure='angle')
        nominal_data = self.generate_test_signal(500)
        sdf.train(nominal_data)

        # Normal data should have low anomaly score
        normal_test = self.generate_test_signal(100)
        result_normal = sdf.detect(normal_test)

        # Anomalous data should have high anomaly score
        anomalous_test = self.generate_test_signal(100) + 2.0
        result_anomalous = sdf.detect(anomalous_test)

        # Anomalous should have higher score
        self.assertGreater(
            result_anomalous['anomaly_measure'],
            result_normal['anomaly_measure']
        )

        # Check result structure
        self.assertIn('anomaly_measure', result_normal)
        self.assertIn('is_anomalous', result_normal)
        self.assertIn('state_probabilities', result_normal)
        self.assertIn('threshold', result_normal)

    def test_error_detection_without_training(self):
        """Test that detection fails if training not done"""
        sdf = SDF(alphabet_size=8, depth=1)
        test_data = self.generate_test_signal(100)

        with self.assertRaises(RuntimeError):
            sdf.detect(test_data)

    def test_threshold_setting(self):
        """Test adaptive threshold setting"""
        sdf = SDF(alphabet_size=8, depth=1)
        sdf.train(self.generate_test_signal(500))

        # Set threshold
        sdf.set_threshold(0.3)
        config = sdf.get_config()
        self.assertEqual(config['threshold'], 0.3)

        # Invalid threshold should raise
        with self.assertRaises(ValueError):
            sdf.set_threshold(-0.1)


class TestElectronicCircuitExample(unittest.TestCase):
    """Phase 8.2: Example 10.1 - Electronic Circuit (Duffing Oscillator)"""

    def setUp(self):
        if not BINDINGS_AVAILABLE:
            self.skipTest("Native bindings not available")

    def generate_duffing_signal(self, beta, amplitude, omega, duration_cycles):
        """Generate Duffing oscillator signal
        d²y/dt² + β*dy/dt + y + y³ = A*cos(Ωt)
        """
        dt = 0.01
        steps = int(duration_cycles * 2 * np.pi / omega / dt)

        signal = np.zeros(steps)
        y, dy = 0.0, 0.0

        for i in range(steps):
            t = i * dt
            forcing = amplitude * np.cos(omega * t)
            ddy = forcing - beta * dy - y - y**3
            dy += ddy * dt
            y += dy * dt
            signal[i] = y

        return signal

    def test_duffing_nominal_vs_anomalous(self):
        """Reproduce Example 10.1: Duffing equation anomaly detection"""
        sdf = SDF(alphabet_size=8, depth=1, wavelet='morlet')

        # Nominal condition: β = 0.1
        nominal_signal = self.generate_duffing_signal(
            beta=0.1, amplitude=5.0, omega=1.0, duration_cycles=50
        )
        sdf.train(nominal_signal)

        # Anomalous condition 1: β = 0.2 (moderate damping increase)
        anomalous_1 = self.generate_duffing_signal(
            beta=0.2, amplitude=5.0, omega=1.0, duration_cycles=50
        )
        result_1 = sdf.detect(anomalous_1)

        # Anomalous condition 2: β = 0.3 (large damping increase)
        anomalous_2 = self.generate_duffing_signal(
            beta=0.3, amplitude=5.0, omega=1.0, duration_cycles=50
        )
        result_2 = sdf.detect(anomalous_2)

        # Verify detection
        self.assertGreater(result_1['anomaly_measure'], 0.05)
        self.assertGreater(result_2['anomaly_measure'], result_1['anomaly_measure'])
        self.assertTrue(result_2['is_anomalous'])


class TestFatigueDamageExample(unittest.TestCase):
    """Phase 8.2: Example 10.2 - Fatigue Damage Detection"""

    def setUp(self):
        if not BINDINGS_AVAILABLE:
            self.skipTest("Native bindings not available")

    def generate_fatigue_signal(self, num_cycles, damage_level=0.0):
        """Generate ultrasonic signal with fatigue damage effects"""
        base_freq = 1000.0
        dt = 1.0 / (10.0 * base_freq)
        samples_per_cycle = int(1.0 / (base_freq * dt))
        total_samples = num_cycles * samples_per_cycle

        signal = np.zeros(total_samples)
        for i in range(total_samples):
            t = i * dt
            # Amplitude increases with damage level
            amplitude = 1.0 + damage_level * min(t / (num_cycles * 0.1), 1.0)
            signal[i] = amplitude * np.sin(2 * np.pi * base_freq * t)

        return signal

    def test_fatigue_crack_detection(self):
        """Reproduce Example 10.2: Ultrasonic fatigue monitoring"""
        sdf = SDF(
            alphabet_size=8, depth=1, wavelet='gaus2', anomaly_measure='angle'
        )

        # Nominal (no damage, 5k cycles)
        nominal_signal = self.generate_fatigue_signal(5, damage_level=0.0)
        sdf.train(nominal_signal)

        # Early stage (30k cycles, minor damage)
        early_damage = self.generate_fatigue_signal(5, damage_level=0.2)
        result_early = sdf.detect(early_damage)

        # Advanced stage (45k cycles, visible crack)
        visible_crack = self.generate_fatigue_signal(5, damage_level=0.5)
        result_crack = sdf.detect(visible_crack)

        # Late stage (60k+ cycles, propagating crack)
        propagating = self.generate_fatigue_signal(5, damage_level=0.8)
        result_propagating = sdf.detect(propagating)

        # Verify detection progression
        self.assertGreater(
            result_crack['anomaly_measure'],
            result_early['anomaly_measure']
        )
        self.assertGreater(
            result_propagating['anomaly_measure'],
            result_crack['anomaly_measure']
        )

        # Visible crack should be detected as anomalous
        self.assertTrue(result_crack['is_anomalous'])


class TestAnomalyMeasures(unittest.TestCase):
    """Test anomaly measure implementations (Eq. 37-39)"""

    def setUp(self):
        if not BINDINGS_AVAILABLE:
            self.skipTest("Native bindings not available")

    def test_angle_measure_equation_38(self):
        """Validate angle-based measure (Eq. 38)"""
        from symbolic_dynamic_filtering import SDF

        sdf = SDF(anomaly_measure='angle')

        # Identical distributions should give 0 angle
        p_identical = np.array([0.25, 0.25, 0.25, 0.25])
        angle = sdf._angle_measure(p_identical, p_identical)
        self.assertAlmostEqual(angle, 0.0, places=5)

        # Orthogonal distributions should give 0.5 (π/2 / π)
        p_orthogonal = np.array([1.0, 0.0, 0.0, 0.0])
        p_ref = np.array([0.0, 1.0, 0.0, 0.0])
        angle = sdf._angle_measure(p_orthogonal, p_ref)
        self.assertGreater(angle, 0.4)  # Should be ~0.5

    def test_norm_measures_equation_37(self):
        """Validate norm-based measures (Eq. 37)"""
        from symbolic_dynamic_filtering import SDF

        sdf = SDF(anomaly_measure='norm_l2')

        p1 = np.array([0.25, 0.25, 0.25, 0.25])
        p2 = np.array([0.5, 0.2, 0.2, 0.1])

        # L2 norm: sqrt(sum((p1-p2)²))
        l2_dist = sdf._compute_anomaly_measure(p2, p1)
        expected_l2 = np.sqrt(np.sum((p2 - p1) ** 2))
        self.assertAlmostEqual(l2_dist, expected_l2, places=5)

    def test_kl_divergence_equation_39(self):
        """Validate KL divergence (Eq. 39)"""
        from symbolic_dynamic_filtering import SDF

        sdf = SDF(anomaly_measure='kl_divergence')

        # Identical distributions should give KL = 0
        p_identical = np.array([0.25, 0.25, 0.25, 0.25])
        kl = sdf._kl_divergence(p_identical, p_identical)
        self.assertAlmostEqual(kl, 0.0, places=5)

        # Different distributions should give KL > 0
        p1 = np.array([0.9, 0.05, 0.03, 0.02])
        p2 = np.array([0.25, 0.25, 0.25, 0.25])
        kl = sdf._kl_divergence(p1, p2)
        self.assertGreater(kl, 0.0)


if __name__ == '__main__':
    unittest.main()
