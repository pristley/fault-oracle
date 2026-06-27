# Installation Guide: Symbolic Dynamic Filtering

Comprehensive installation instructions for the Symbolic Dynamic Filtering (SDF) library.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [From Source](#from-source)
4. [From PyPI (Future)](#from-pypi-future)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows (with WSL2 recommended)
- **Rust**: 1.56+ (for building from source)
- **Python**: 3.8+
- **Git**: For cloning the repository

### Install Rust

If you don't have Rust installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Verify installation:

```bash
rustc --version  # Should be 1.56 or higher
cargo --version
```

### Install Python Dependencies

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install numpy scipy scikit-learn matplotlib  # Optional but recommended
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/pristley/fault-oracle.git
cd fault-oracle
```

### 2. Install with Maturin (Development)

```bash
# Install maturin (PyO3 build tool)
pip install maturin

# Build and install development version
maturin develop --release
```

This creates an editable installation that reflects source code changes.

### 3. Verify Installation

```python
python3 -c "from symbolic_dynamic_filtering import SDF; print('Success!')"
```

---

## From Source

### Full Build Process

#### Option A: Using Build Script (Recommended)

```bash
cd /path/to/fault-oracle
chmod +x build.sh
./build.sh
```

This will:
- Build the Rust library (release)
- Run all tests
- Build Python extension
- Generate documentation
- Run linting checks

#### Option B: Manual Build

```bash
# Step 1: Build Rust library
cargo build --release

# Step 2: Run Rust tests
cargo test --release

# Step 3: Install maturin
pip install maturin

# Step 4: Build Python extension
maturin develop --release

# Step 5: Run Python tests
python -m pytest python/tests/

# Step 6: Generate documentation
cargo doc --release --no-deps
```

### Build Artifacts

After successful build:

```
fault-oracle/
├── target/release/
│   ├── symbolic_dynamic_filtering.so  (Linux/macOS)
│   ├── symbolic_dynamic_filtering.pyd (Windows)
│   └── ...
├── python/symbolic_dynamic_filtering/
│   └── _core.so                       (Python extension)
└── target/doc/
    └── symbolic_dynamic_filtering/    (HTML docs)
```

---

## From PyPI (Future)

Once published to PyPI (Python Package Index):

```bash
# Simple installation
pip install symbolic-dynamic-filtering

# With optional dependencies
pip install symbolic-dynamic-filtering[dev]
pip install symbolic-dynamic-filtering[viz]  # For plotting
```

Upgrade to latest version:

```bash
pip install --upgrade symbolic-dynamic-filtering
```

---

## Verification

### 1. Import Test

```python
python3 << 'EOF'
from symbolic_dynamic_filtering import SDF
import numpy as np

print("✓ Module imported successfully")

# Create SDF instance
sdf = SDF(alphabet_size=8, depth=2)
print(f"✓ SDF created: {sdf.get_config()}")

# Test training
nominal_data = np.random.randn(500)
sdf.train(nominal_data)
print("✓ Training completed")

# Test detection
test_data = np.random.randn(100)
result = sdf.detect(test_data)
print(f"✓ Detection completed: anomaly={result['anomaly_measure']:.4f}")

print("\n✓ All verification tests passed!")
EOF
```

### 2. Run Examples

```bash
# Run Rust examples
cargo run --release --example electronic_circuit
cargo run --release --example fatigue_detection

# Run Python examples
python examples/fatigue_detection.py
```

### 3. Run Test Suite

```bash
# Rust tests
cargo test --release

# Python tests
python -m pytest python/tests/ -v

# Individual test modules
python -m unittest symbolic_dynamic_filtering.tests.test_integration
```

### 4. Generate Documentation

```bash
# Generate and open Rust docs
cargo doc --release --no-deps --open

# View README
cat README.md | less
```

---

## Troubleshooting

### Issue: "cargo not found"

**Solution**: Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Issue: "maturin: command not found"

**Solution**: Install maturin

```bash
pip install maturin
```

If still not found, use full path:

```bash
python -m pip install maturin
python -m maturin develop --release
```

### Issue: ImportError when importing symbolic_dynamic_filtering

**Solution 1**: Ensure the package is installed

```bash
maturin develop --release
```

**Solution 2**: Check Python path

```python
import sys
print(sys.path)  # Verify current directory is included
```

**Solution 3**: Reinstall in editable mode

```bash
pip uninstall symbolic-dynamic-filtering -y
maturin develop --release
```

### Issue: "error[E0514]: cannot find crate for `pyo3`"

**Solution**: Ensure PyO3 version compatibility

```bash
# Update dependencies
cargo update

# Clean and rebuild
cargo clean
cargo build --release
```

### Issue: Test failures on Python side

**Solution 1**: Rebuild extension

```bash
maturin develop --release
```

**Solution 2**: Ensure native bindings are available

The Python tests will skip if native bindings aren't available. This is expected behavior for early development.

### Issue: Memory errors during large tests

**Solution**: Reduce test data size or increase system memory

```python
# In test files, use smaller datasets
test_size = 100  # Instead of 10000
```

### Issue: Slow builds

**Solution 1**: Use release mode (faster at runtime)

```bash
cargo build --release  # Takes longer initially but much faster execution
```

**Solution 2**: Parallel compilation

```bash
cargo build --release -j 4  # Use 4 cores
```

---

## Environment Variables

Optional configuration:

```bash
# Rust backtrace for debugging
RUST_BACKTRACE=1 python3 your_script.py

# Verbose logging
RUST_LOG=debug python3 your_script.py
```

---

## Virtual Environments (Recommended)

### Using venv

```bash
# Create environment
python3 -m venv sdf_env

# Activate
source sdf_env/bin/activate  # On Windows: sdf_env\Scripts\activate

# Install
maturin develop --release

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create environment
conda create -n sdf python=3.10 rust -c conda-forge

# Activate
conda activate sdf

# Install
maturin develop --release
```

---

## Development Installation

For developers contributing to the project:

```bash
# Clone with submodules
git clone --recursive https://github.com/pristley/fault-oracle.git
cd fault-oracle

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Build and test
./build.sh
```

---

## Docker (Optional)

For isolated builds:

```dockerfile
FROM rust:latest
RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /app
COPY . .
RUN pip install maturin
RUN maturin develop --release
```

Build and run:

```bash
docker build -t sdf-build .
docker run --rm -it sdf-build python3 -c "from symbolic_dynamic_filtering import SDF; print('Success!')"
```

---

## Support

- **Repository**: https://github.com/pristley/fault-oracle
- **Issues**: Report bugs and request features on GitHub Issues
- **Documentation**: See [README.md](README.md) and `cargo doc --open`

---

## License

The Symbolic Dynamic Filtering library is provided as-is. See LICENSE file for details.

---

## Citation

If you use SDF in your research, please cite:

```bibtex
@software{sdf2024,
  title={Symbolic Dynamic Filtering: Real-time Anomaly Detection},
  author={Pristley, et al.},
  year={2024},
  url={https://github.com/pristley/fault-oracle}
}
```
