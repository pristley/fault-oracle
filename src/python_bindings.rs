//! Python bindings for PyO3
//!
//! Exposes the Rust library to Python via PyO3.
//! Implements Phase 7: Python Bindings

use pyo3::prelude::*;
use crate::wavelets::WaveletBasis;

/// Initialize the Python module
/// Implements Phase 7.1: Main Binding File
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Main API classes
    m.add_class::<PySymbolicDynamicFilter>()?;
    m.add_class::<PyWaveletTransform>()?;
    m.add_class::<PyPartitioner>()?;
    m.add_class::<PyDMarkovMachine>()?;
    m.add_class::<PyAnomalyDetector>()?;

    // Enum wrappers
    m.add_class::<PyWaveletBasis>()?;
    m.add_class::<PyAnomalyMeasureType>()?;
    m.add_class::<PyNormType>()?;

    Ok(())
}

/// Python wrapper for the complete SDF pipeline
#[pyclass]
pub struct PySymbolicDynamicFilter {
    alphabet_size: usize,
    depth: usize,
    wavelet: String,
    nominal_trained: bool,
}

#[pymethods]
impl PySymbolicDynamicFilter {
    /// Create a new Symbolic Dynamic Filter
    /// 
    /// # Arguments
    /// * `alphabet_size` - Size of the symbolic alphabet
    /// * `depth` - D-Markov depth parameter
    /// * `wavelet` - Wavelet basis name ('haar', 'morlet', etc.)
    #[new]
    fn new(alphabet_size: usize, depth: usize, wavelet: String) -> PyResult<Self> {
        if alphabet_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "alphabet_size must be > 0",
            ));
        }

        if depth == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "depth must be > 0",
            ));
        }

        Ok(PySymbolicDynamicFilter {
            alphabet_size,
            depth,
            wavelet,
            nominal_trained: false,
        })
    }

    /// Get the alphabet size
    pub fn get_alphabet_size(&self) -> usize {
        self.alphabet_size
    }

    /// Get the D-Markov depth
    pub fn get_depth(&self) -> usize {
        self.depth
    }

    /// Get the wavelet type
    pub fn get_wavelet(&self) -> String {
        self.wavelet.clone()
    }

    /// Check if nominal data has been trained
    pub fn is_trained(&self) -> bool {
        self.nominal_trained
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SymbolicDynamicFilter(alphabet_size={}, depth={}, wavelet='{}')",
            self.alphabet_size, self.depth, self.wavelet
        )
    }
}

/// Python wrapper for Wavelet Transform
#[pyclass]
pub struct PyWaveletTransform {
    coefficients: Vec<Vec<f64>>,
    scales: Vec<usize>,
    basis_name: String,
}

#[pymethods]
impl PyWaveletTransform {
    /// Create a new Wavelet Transform
    #[new]
    fn new(basis_name: String) -> PyResult<Self> {
        Ok(PyWaveletTransform {
            coefficients: Vec::new(),
            scales: Vec::new(),
            basis_name,
        })
    }

    /// Compute wavelet transform on signal
    pub fn compute(&mut self, signal: Vec<f64>) -> PyResult<()> {
        if signal.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Signal cannot be empty",
            ));
        }

        // Simple placeholder: just store signal as coefficients
        self.coefficients = vec![signal];
        self.scales = vec![1];

        Ok(())
    }

    /// Get the coefficients
    pub fn get_coefficients(&self) -> Vec<Vec<f64>> {
        self.coefficients.clone()
    }

    /// Get the scales
    pub fn get_scales(&self) -> Vec<usize> {
        self.scales.clone()
    }

    pub fn __repr__(&self) -> String {
        format!("WaveletTransform(basis='{}')", self.basis_name)
    }
}

/// Python wrapper for Partitioner
#[pyclass]
pub struct PyPartitioner {
    num_regions: usize,
    boundaries: Vec<f64>,
}

#[pymethods]
impl PyPartitioner {
    /// Create a new Partitioner with uniform partitioning
    #[new]
    fn new(num_regions: usize) -> PyResult<Self> {
        if num_regions == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_regions must be > 0",
            ));
        }

        Ok(PyPartitioner {
            num_regions,
            boundaries: Vec::new(),
        })
    }

    /// Compute partitioning for a signal
    pub fn fit(&mut self, signal: Vec<f64>) -> PyResult<()> {
        if signal.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Signal cannot be empty",
            ));
        }

        // Simple uniform partitioning
        let min_val = signal.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        self.boundaries.clear();
        for i in 1..self.num_regions {
            self.boundaries
                .push(min_val + (range * i as f64) / self.num_regions as f64);
        }

        Ok(())
    }

    /// Encode a value to a region index
    pub fn encode(&self, value: f64) -> usize {
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if value < boundary {
                return i;
            }
        }
        self.boundaries.len()
    }

    /// Get the boundaries
    pub fn get_boundaries(&self) -> Vec<f64> {
        self.boundaries.clone()
    }

    pub fn __repr__(&self) -> String {
        format!("Partitioner(num_regions={})", self.num_regions)
    }
}

/// Python wrapper for D-Markov Machine
#[pyclass]
pub struct PyDMarkovMachine {
    order: usize,
    states_count: usize,
}

#[pymethods]
impl PyDMarkovMachine {
    /// Create a new D-Markov Machine
    #[new]
    fn new(order: usize, alphabet_size: usize) -> PyResult<Self> {
        if order == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "order must be > 0",
            ));
        }

        // Calculate number of states (alphabet_size^order)
        let states_count = alphabet_size.pow(order as u32);

        Ok(PyDMarkovMachine { order, states_count })
    }

    /// Get the order
    pub fn get_order(&self) -> usize {
        self.order
    }

    /// Get the number of states
    pub fn get_states_count(&self) -> usize {
        self.states_count
    }

    pub fn __repr__(&self) -> String {
        format!("DMarkovMachine(order={}, states={})", self.order, self.states_count)
    }
}

/// Python wrapper for Anomaly Detector
#[pyclass]
pub struct PyAnomalyDetector {
    measure_type: String,
    threshold: f64,
    history_size: usize,
}

#[pymethods]
impl PyAnomalyDetector {
    /// Create a new Anomaly Detector
    #[new]
    fn new(measure_type: String, threshold: f64) -> PyResult<Self> {
        if threshold < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "threshold must be >= 0.0",
            ));
        }

        Ok(PyAnomalyDetector {
            measure_type,
            threshold,
            history_size: 0,
        })
    }

    /// Get the measure type
    pub fn get_measure_type(&self) -> String {
        self.measure_type.clone()
    }

    /// Get the threshold
    pub fn get_threshold(&self) -> f64 {
        self.threshold
    }

    /// Set a new threshold
    pub fn set_threshold(&mut self, threshold: f64) -> PyResult<()> {
        if threshold < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "threshold must be >= 0.0",
            ));
        }
        self.threshold = threshold;
        Ok(())
    }

    /// Get history size
    pub fn get_history_size(&self) -> usize {
        self.history_size
    }

    pub fn __repr__(&self) -> String {
        format!(
            "AnomalyDetector(measure='{}', threshold={})",
            self.measure_type, self.threshold
        )
    }
}

/// Python wrapper for WaveletBasis enum
#[pyclass]
pub struct PyWaveletBasis {
    inner: WaveletBasis,
}

#[pymethods]
impl PyWaveletBasis {
    /// Create a Haar wavelet
    #[staticmethod]
    pub fn haar() -> Self {
        PyWaveletBasis {
            inner: WaveletBasis::Haar,
        }
    }

    /// Create a Morlet wavelet
    #[staticmethod]
    pub fn morlet() -> Self {
        PyWaveletBasis {
            inner: WaveletBasis::Morlet,
        }
    }

    /// Create a Mexican Hat wavelet
    #[staticmethod]
    pub fn mexican_hat() -> Self {
        PyWaveletBasis {
            inner: WaveletBasis::MexicanHat,
        }
    }

    /// Create a Gaussian wavelet (default order 1)
    #[staticmethod]
    pub fn gaussian() -> Self {
        PyWaveletBasis {
            inner: WaveletBasis::Gaussian(1),
        }
    }

    /// Create a Daubechies wavelet (default order 1)
    #[staticmethod]
    pub fn daubechies() -> Self {
        PyWaveletBasis {
            inner: WaveletBasis::Daubechies(1),
        }
    }

    /// Get center frequency
    pub fn center_frequency(&self) -> f64 {
        self.inner.center_frequency()
    }

    /// Get vanishing moments
    pub fn vanishing_moments(&self) -> usize {
        self.inner.vanishing_moments()
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    pub fn __repr__(&self) -> String {
        format!("WaveletBasis({})", self.inner)
    }
}

/// Python wrapper for NormType enum
#[pyclass]
pub struct PyNormType {
    inner: String,
}

#[pymethods]
impl PyNormType {
    /// L1 norm (Manhattan distance)
    #[staticmethod]
    pub fn l1() -> Self {
        PyNormType {
            inner: "L1".to_string(),
        }
    }

    /// L2 norm (Euclidean distance)
    #[staticmethod]
    pub fn l2() -> Self {
        PyNormType {
            inner: "L2".to_string(),
        }
    }

    /// L-infinity norm
    #[staticmethod]
    pub fn linf() -> Self {
        PyNormType {
            inner: "Linf".to_string(),
        }
    }

    pub fn __str__(&self) -> String {
        self.inner.clone()
    }

    pub fn __repr__(&self) -> String {
        format!("NormType({})", self.inner)
    }
}

/// Python wrapper for AnomalyMeasure enum
#[pyclass]
pub struct PyAnomalyMeasureType {
    inner: String,
}

#[pymethods]
impl PyAnomalyMeasureType {
    /// Norm-based anomaly measure
    #[staticmethod]
    pub fn norm(norm_type: String) -> Self {
        PyAnomalyMeasureType {
            inner: format!("Norm({})", norm_type),
        }
    }

    /// Angle-based anomaly measure
    #[staticmethod]
    pub fn angle() -> Self {
        PyAnomalyMeasureType {
            inner: "Angle".to_string(),
        }
    }

    /// KL divergence-based anomaly measure
    #[staticmethod]
    pub fn kl_divergence() -> Self {
        PyAnomalyMeasureType {
            inner: "KLDivergence".to_string(),
        }
    }

    /// Matrix norm anomaly measure
    #[staticmethod]
    pub fn matrix_norm() -> Self {
        PyAnomalyMeasureType {
            inner: "MatrixNorm".to_string(),
        }
    }

    /// Entropy rate anomaly measure
    #[staticmethod]
    pub fn entropy_rate() -> Self {
        PyAnomalyMeasureType {
            inner: "EntropyRate".to_string(),
        }
    }

    /// Excess entropy anomaly measure
    #[staticmethod]
    pub fn excess_entropy() -> Self {
        PyAnomalyMeasureType {
            inner: "ExcessEntropy".to_string(),
        }
    }

    /// Statistical complexity anomaly measure
    #[staticmethod]
    pub fn statistical_complexity() -> Self {
        PyAnomalyMeasureType {
            inner: "StatisticalComplexity".to_string(),
        }
    }

    pub fn __str__(&self) -> String {
        self.inner.clone()
    }

    pub fn __repr__(&self) -> String {
        format!("AnomalyMeasure({})", self.inner)
    }
}
