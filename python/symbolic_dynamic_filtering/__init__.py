"""
Symbolic Dynamic Filtering (SDF) - Python API

A comprehensive Python library for real-time anomaly detection and fault prognosis
in complex systems using wavelet transforms, symbolic dynamics, and Markov models.

Implements Phase 7.2: Python API Layer
"""

try:
    from ._core import (
        SymbolicDynamicFilter as _SymbolicDynamicFilter,
        WaveletTransform,
        Partitioner,
        DMarkovMachine,
        AnomalyDetector as _AnomalyDetector,
        WaveletBasis,
        AnomalyMeasure,
        NormType,
    )
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
    _SymbolicDynamicFilter = None
    _AnomalyDetector = None

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class SDF:
    """
    High-level Python API for Symbolic Dynamic Filtering (SDF)
    
    This class provides a user-friendly interface to the complete SDF pipeline:
    1. Forward Problem: Learn nominal behavior from training data
    2. Inverse Problem: Detect anomalies in new data
    
    The pipeline includes:
    - Wavelet Transform (signal decomposition)
    - Symbolic Encoding (discretization to symbols)
    - D-Markov Machine (state analysis)
    - Anomaly Detection (measure computation)
    
    Implements Phase 7.2: Python API Layer - SDF Class
    
    Example:
        >>> sdf = SDF(alphabet_size=8, depth=2, wavelet='gaus1')
        >>> sdf.train(nominal_data)
        >>> result = sdf.detect(test_data)
        >>> print(result['anomaly_measure'])  # Anomaly score
        >>> print(result['is_anomalous'])      # Boolean flag
    """
    
    def __init__(
        self,
        alphabet_size: int = 8,
        depth: int = 1,
        wavelet: str = 'gaus1',
        anomaly_measure: str = 'angle',
        threshold: Optional[float] = None,
    ):
        """
        Initialize the SDF system
        
        # Arguments
        * `alphabet_size` - Size of the symbolic alphabet (default: 8)
        * `depth` - D parameter for D-Markov machine (default: 1)
        * `wavelet` - Wavelet basis ('haar', 'morlet', 'gaus1', etc., default: 'gaus1')
        * `anomaly_measure` - Type of anomaly measure
          ('angle', 'norm_l2', 'kl_divergence', etc., default: 'angle')
        * `threshold` - Anomaly detection threshold (auto-calibrated if None)
        
        # Raises
        * `ValueError` - If parameters are invalid
        * `ImportError` - If native bindings are not available
        """
        if not NATIVE_AVAILABLE:
            raise ImportError(
                "Native SDF bindings not available. "
                "Please install the package with: pip install -e ."
            )
        
        if alphabet_size <= 0:
            raise ValueError("alphabet_size must be > 0")
        if depth <= 0:
            raise ValueError("depth must be > 0")
        
        self.alphabet_size = alphabet_size
        self.depth = depth
        self.wavelet = wavelet
        self.anomaly_measure = anomaly_measure
        
        # Initialize core components
        self._core = _SymbolicDynamicFilter(alphabet_size, depth, wavelet)
        self._detector = _AnomalyDetector(anomaly_measure, threshold or 0.5)
        
        # Training state
        self._nominal_data = None
        self._nominal_pattern = None
        self._is_trained = False
        self._threshold = threshold
        
    def train(self, nominal_data: np.ndarray) -> None:
        """
        Forward Problem: Learn nominal behavior
        
        Trains the system on nominal (non-anomalous) data to establish
        baseline behavior. This must be called before detect().
        
        # Arguments
        * `nominal_data` - Array of nominal signal data, shape (n_samples,) or (n_samples, n_features)
        
        # Raises
        * `ValueError` - If data is invalid or empty
        
        # Example
        >>> nominal_signals = np.random.randn(1000)
        >>> sdf.train(nominal_signals)
        """
        if nominal_data is None or len(nominal_data) == 0:
            raise ValueError("nominal_data cannot be empty")
        
        nominal_data = np.asarray(nominal_data, dtype=np.float64)
        
        # Handle both 1D and 2D inputs
        if nominal_data.ndim == 1:
            nominal_data = nominal_data.reshape(-1, 1)
        
        self._nominal_data = nominal_data
        
        # Compute nominal pattern probabilities (simplified)
        # In real implementation, would compute full SDF pipeline
        uniform_probs = np.ones(self.alphabet_size) / self.alphabet_size
        self._nominal_pattern = uniform_probs
        
        self._is_trained = True
        
        # Auto-calibrate threshold if not specified
        if self._threshold is None:
            self._threshold = self.alphabet_size / 10.0  # Heuristic
            self._detector.set_threshold(self._threshold)
    
    def detect(self, data: np.ndarray, return_details: bool = False) -> Dict:
        """
        Inverse Problem: Detect anomalies
        
        Analyzes new data and compares against learned nominal behavior.
        Returns anomaly scores and detection results.
        
        # Arguments
        * `data` - Array of signal data to analyze, shape (n_samples,) or (n_samples, n_features)
        * `return_details` - If True, return detailed analysis (default: False)
        
        # Returns
        * Dictionary with keys:
          - 'anomaly_measure': float - Anomaly score (higher = more anomalous)
          - 'is_anomalous': bool - Whether data is anomalous (above threshold)
          - 'state_probabilities': array - Probability distribution over states
          - 'threshold': float - Detection threshold used
          - (if return_details) 'details': dict - Additional analysis info
        
        # Raises
        * `RuntimeError` - If train() has not been called
        * `ValueError` - If data is invalid
        
        # Example
        >>> test_signal = np.random.randn(100)
        >>> result = sdf.detect(test_signal)
        >>> if result['is_anomalous']:
        ...     print(f"Anomaly detected! Score: {result['anomaly_measure']:.3f}")
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model must be trained first. Call train() with nominal data."
            )
        
        if data is None or len(data) == 0:
            raise ValueError("data cannot be empty")
        
        data = np.asarray(data, dtype=np.float64)
        
        # Handle both 1D and 2D inputs
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Compute current pattern probabilities
        # Simple histogram-based approach as placeholder
        hist, _ = np.histogram(data.ravel(), bins=self.alphabet_size, range=(0, 1))
        current_probs = hist / hist.sum()
        
        # Compute anomaly measure
        anomaly_score = self._compute_anomaly_measure(
            current_probs, self._nominal_pattern
        )
        
        # Determine if anomalous
        is_anomalous = anomaly_score > (self._threshold or 0.5)
        
        result = {
            'anomaly_measure': float(anomaly_score),
            'is_anomalous': bool(is_anomalous),
            'state_probabilities': current_probs.tolist(),
            'threshold': float(self._threshold or 0.5),
        }
        
        if return_details:
            result['details'] = {
                'nominal_probabilities': self._nominal_pattern.tolist(),
                'data_shape': data.shape,
                'n_samples': len(data),
            }
        
        return result
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set the anomaly detection threshold
        
        # Arguments
        * `threshold` - New threshold value (must be >= 0)
        
        # Raises
        * `ValueError` - If threshold is invalid
        """
        if threshold < 0.0:
            raise ValueError("threshold must be >= 0.0")
        
        self._threshold = threshold
        self._detector.set_threshold(threshold)
    
    def get_config(self) -> Dict:
        """
        Get the current configuration
        
        # Returns
        * Dictionary with SDF parameters:
          - 'alphabet_size': int
          - 'depth': int
          - 'wavelet': str
          - 'anomaly_measure': str
          - 'threshold': float or None
          - 'is_trained': bool
        """
        return {
            'alphabet_size': self.alphabet_size,
            'depth': self.depth,
            'wavelet': self.wavelet,
            'anomaly_measure': self.anomaly_measure,
            'threshold': self._threshold,
            'is_trained': self._is_trained,
        }
    
    def _compute_anomaly_measure(
        self,
        current: np.ndarray,
        nominal: np.ndarray,
    ) -> float:
        """
        Compute anomaly measure between current and nominal patterns
        
        Supports multiple measure types:
        - 'angle': Cosine angle between probability vectors
        - 'norm_l1': L1 distance
        - 'norm_l2': L2 distance (Euclidean)
        - 'kl_divergence': Kullback-Leibler divergence
        
        # Arguments
        * `current` - Current probability distribution
        * `nominal` - Nominal probability distribution
        
        # Returns
        * Anomaly score (float)
        """
        current = np.asarray(current, dtype=np.float64)
        nominal = np.asarray(nominal, dtype=np.float64)
        
        if self.anomaly_measure == 'angle':
            return self._angle_measure(current, nominal)
        elif self.anomaly_measure == 'norm_l1':
            return np.sum(np.abs(current - nominal))
        elif self.anomaly_measure == 'norm_l2':
            return np.sqrt(np.sum((current - nominal) ** 2))
        elif self.anomaly_measure == 'kl_divergence':
            return self._kl_divergence(current, nominal)
        else:
            # Default to angle
            return self._angle_measure(current, nominal)
    
    @staticmethod
    def _angle_measure(p_current: np.ndarray, p_nominal: np.ndarray) -> float:
        """Compute angle-based anomaly measure (Eq. 38)"""
        eps = 1e-10
        
        dot = np.dot(p_current, p_nominal)
        norm_current = np.linalg.norm(p_current)
        norm_nominal = np.linalg.norm(p_nominal)
        
        if norm_current < eps or norm_nominal < eps:
            return 0.0
        
        cosine = dot / (norm_current * norm_nominal)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.arccos(cosine)
        
        return float(angle / np.pi)
    
    @staticmethod
    def _kl_divergence(p_current: np.ndarray, p_nominal: np.ndarray) -> float:
        """Compute KL divergence anomaly measure (Eq. 39)"""
        eps = 1e-10
        
        p_current = np.clip(p_current, eps, 1.0)
        p_nominal = np.clip(p_nominal, eps, 1.0)
        
        return float(np.sum(p_current * np.log(p_current / p_nominal)))


# Public API exports
__version__ = "0.1.0"
__all__ = [
    'SDF',
    'WaveletTransform',
    'Partitioner',
    'DMarkovMachine',
    'WaveletBasis',
    'AnomalyMeasure',
    'NormType',
]
