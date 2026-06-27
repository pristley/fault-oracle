# Symbolic Dynamic Filtering (SDF)

A comprehensive Rust library implementing Symbolic Dynamic Filtering for real-time anomaly detection and fault prognosis in complex systems. This library combines wavelet transforms, symbolic dynamics, and Markov models to detect and diagnose faults in complex mechanical and electrical systems.

## Features

- **Continuous Wavelet Transform (CWT)**: Multi-scale signal analysis with various wavelet families (Haar, Daubechies, Gaussian, Morlet, Mexican Hat)
- **Symbolic Dynamics**: Conversion of continuous signals to symbolic sequences with customizable partitioning strategies
- **D-Markov Machines**: State-based modeling of symbolic sequences for pattern recognition
- **Anomaly Detection**: Multiple measures (norm-based, angle-based, KL-divergence, Hellinger, Wasserstein)
- **Scale Selection**: Automatic scale selection based on signal properties and Power Spectral Density
- **Python Integration**: Full Python bindings via PyO3 for seamless integration with Python workflows

## Project Structure

```
symbolic-dynamic-filtering/
├── src/
│   ├── lib.rs                    # Library root with error handling
│   ├── main.rs                   # CLI binary
│   ├── python_bindings.rs        # PyO3 Python bindings
│   ├── utils.rs                  # Utility functions
│   ├── wavelets/
│   │   ├── mod.rs               # Module root
│   │   ├── basis.rs             # Wavelet basis selection (Haar, Daubechies, etc.)
│   │   ├── transform.rs         # Continuous wavelet transform (CWT) implementation
│   │   └── scales.rs            # Scale selection strategies
│   ├── partitioning/
│   │   ├── mod.rs
│   │   ├── types.rs             # Partition structures
│   │   ├── uniform.rs           # Uniform partitioning
│   │   └── maximum_entropy.rs   # Maximum entropy partitioning
│   ├── symbolic/
│   │   ├── mod.rs
│   │   ├── alphabet.rs          # Symbol alphabet management
│   │   ├── encoding.rs          # Signal-to-symbol encoding
│   │   └── dynamics.rs          # Symbolic dynamics analysis
│   ├── markov/
│   │   ├── mod.rs
│   │   ├── state.rs             # State definitions
│   │   ├── transitions.rs       # Transition matrices
│   │   └── d_markov.rs          # D-Markov machine implementation
│   └── anomaly/
│       ├── mod.rs
│       ├── measures.rs          # Anomaly measures (Norm, Angle, KL, etc.)
│       └── detection.rs         # Anomaly detection algorithms
├── examples/
│   ├── electronic_circuit.rs    # Electronic circuit anomaly detection
│   └── fatigue_detection.rs     # Fatigue crack detection
├── tests/
│   └── integration_tests.rs     # Integration tests
├── python/
│   ├── __init__.py              # Python package
│   └── tests/
│       ├── test_wavelets.py
│       ├── test_partitioning.py
│       └── test_integration.py
├── Cargo.toml                   # Rust dependencies
├── Cargo.lock                   # Locked versions
├── pyproject.toml               # Python project config
├── build.rs                     # Build script for Python bindings
└── README.md                    # This file
```

## Installation

### Rust

```bash
# Build the library
cargo build --release

# Run tests
cargo test

# Build documentation
cargo doc --open
```

### Python

```bash
# Install from source with maturin
pip install maturin
maturin develop --release
```

## Usage

### Rust Example

```rust
use symbolic_dynamic_filtering::wavelets::{WaveletBasis, WaveletTransform};
use symbolic_dynamic_filtering::anomaly::{AnomalyDetector, AnomalyMeasure};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create signal
    let signal = vec![1.0, 2.0, 1.5, 1.0, 0.5];
    
    // Apply wavelet transform
    let scales = vec![1, 2, 4, 8];
    let wavelet = WaveletBasis::Morlet;
    
    let cwt = WaveletTransform::continuous(&signal, &scales, wavelet)?;
    
    // Extract features
    let features = cwt.compute_scale_norms();
    
    // Detect anomalies
    let reference = vec![0.05, 0.04, 0.03, 0.02];
    let detector = AnomalyDetector::new(AnomalyMeasure::Norm, 0.1)?;
    
    let is_anomalous = detector.is_anomalous(&features, &reference)?;
    println!("Anomaly detected: {}", is_anomalous);
    
    Ok(())
}
```

### Running Examples

```bash
# Electronic circuit anomaly detection
cargo run --example electronic_circuit --release

# Fatigue detection
cargo run --example fatigue_detection --release
```

## Core Modules

### Wavelets Module (`wavelets/`)

Implements continuous wavelet transforms with multiple wavelet families:

- **basis.rs**: WaveletBasis enum with center frequencies, vanishing moments, and wavelet computation
  - `WaveletBasis::Haar` - Simple Haar wavelet
  - `WaveletBasis::Daubechies(n)` - Daubechies wavelets (db1-db10)
  - `WaveletBasis::Gaussian(n)` - Gaussian wavelets (gaus1-gaus10)
  - `WaveletBasis::Morlet` - Morlet (Gabor) wavelet
  - `WaveletBasis::MexicanHat` - Mexican Hat wavelet

- **transform.rs**: Continuous Wavelet Transform
  - `WaveletTransform::continuous()` - Compute CWT
  - `compute_scale_norms()` - Calculate L2 norms for each scale
  - `compute_pseudo_frequencies()` - Convert scales to frequencies
  - `arrange_scale_series()` - Create feature vector from coefficients

- **scales.rs**: Scale Selection
  - `ScaleSelector::from_psd()` - Select scales from Power Spectral Density
  - `ScaleSelector::linear_scales()` - Generate evenly-spaced scales
  - `ScaleSelector::logarithmic_scales()` - Generate log-spaced scales
  - `ScaleSelector::from_center_frequencies()` - Map frequencies to scales
  - `ScaleSelector::noise_suppression_ratio()` - Evaluate SNR improvement

### Partitioning Module (`partitioning/`)

Converts continuous signals to discrete partitions for symbolic encoding:

- **types.rs**: Partition structures and encoding
- **uniform.rs**: Uniform quantization
- **maximum_entropy.rs**: Information-theoretic partitioning

### Symbolic Module (`symbolic/`)

Converts signals to symbolic sequences:

- **alphabet.rs**: Symbol alphabet (a, b, c, ...)
- **encoding.rs**: Signal-to-symbol encoding with word extraction
- **dynamics.rs**: Symbolic sequence analysis (entropy, transitions)

### Markov Module (`markov/`)

State-based modeling:

- **state.rs**: State definitions
- **transitions.rs**: Transition probability matrices
- **d_markov.rs**: D-Markov machines for order-dependent state analysis

### Anomaly Module (`anomaly/`)

Anomaly detection and scoring:

- **measures.rs**: Multiple measures
  - `Norm` - Euclidean distance
  - `Angle` - Cosine angle between vectors
  - `KLDivergence` - Kullback-Leibler divergence
  - `Hellinger` - Hellinger distance
  - `Wasserstein` - Wasserstein distance

- **detection.rs**: Anomaly detector with adaptive thresholding

## Configuration

### Cargo.toml Key Dependencies

- `ndarray` (0.15) - N-dimensional arrays
- `ndarray-stats` (0.5) - Statistics on arrays
- `nalgebra` (0.32) - Linear algebra
- `rustfft` (6.1) - FFT computations
- `pyo3` (0.20) - Python bindings
- `thiserror` (1.0) - Error handling

### Performance Considerations

- Use `--release` flag for optimized builds
- Scales should typically be in range [1, 256] for most signals
- Wavelet choice affects computational cost: Morlet > Daubechies > Haar
- Partition boundaries can be computed from signal statistics

## Testing

```bash
# Run all tests
cargo test

# Run specific test module
cargo test wavelets::

# Run integration tests with output
cargo test --test integration_tests -- --nocapture

# Run with release optimizations
cargo test --release

# Run Python tests (after installing)
python -m pytest python/tests/
```

## Examples

### Electronic Circuit Analysis
Detects voltage spikes and anomalies in circuit signals:
```bash
cargo run --example electronic_circuit --release
```

### Fatigue Detection
Identifies fatigue cracks through vibration analysis:
```bash
cargo run --example fatigue_detection --release
```

## API Documentation

Generate and view API documentation:

```bash
cargo doc --open
```

## Architecture

The library follows a pipeline architecture:

1. **Signal Acquisition** → Raw signal from sensors
2. **Wavelet Transform** → Multi-scale decomposition
3. **Feature Extraction** → Norm/frequency features
4. **Partitioning** → Discretization strategy
5. **Symbolic Encoding** → Symbolic sequence
6. **Markov Modeling** → State transition analysis
7. **Anomaly Detection** → Scoring and thresholding
8. **Diagnosis** → Fault identification

## Research Foundation

This implementation is based on the theory of Symbolic Dynamic Filtering, combining:
- Continuous Wavelet Transforms (Morlet, Daubechies, etc.)
- Maximum Entropy Partitioning for optimal symbol discrimination
- D-Markov machines for capturing order-dependent state transitions
- Multiple anomaly measures for robust detection

## Contributing

Contributions are welcome. Please ensure:
- Code passes `cargo test`
- Code follows Rust conventions (`rustfmt`)
- New features include tests and documentation

## License

[Add appropriate license information]

## Citation

If you use this library in research, please cite:
[Add appropriate citation information] 
