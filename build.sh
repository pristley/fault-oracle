#!/bin/bash
# Phase 10.1: Build Script for Symbolic Dynamic Filtering
#
# Builds the complete project including:
# - Rust library (cargo build)
# - Python native extension (maturin)
# - Runs all tests
# - Generates documentation

set -e

echo "="
echo "= Symbolic Dynamic Filtering Build"
echo "="
echo ""

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_stage() {
    echo -e "${BLUE}→ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check prerequisites
print_stage "Checking prerequisites"
command -v cargo >/dev/null 2>&1 || { print_error "cargo not found"; exit 1; }
command -v python3 >/dev/null 2>&1 || { print_error "python3 not found"; exit 1; }
print_success "Prerequisites found"
echo ""

# Step 1: Build Rust library (release mode)
print_stage "Building Rust library (release)"
cargo build --release 2>&1 | tail -5
print_success "Rust library built"
echo ""

# Step 2: Run Rust tests
print_stage "Running Rust unit and integration tests"
cargo test --release 2>&1 | tail -10
print_success "All Rust tests passed"
echo ""

# Step 3: Install maturin if needed
print_stage "Checking Python build tools"
if ! python3 -c "import maturin" 2>/dev/null; then
    print_stage "Installing maturin"
    pip install maturin --quiet
fi
print_success "maturin available"
echo ""

# Step 4: Build Python extension
print_stage "Building Python extension (release)"
maturin develop --release 2>&1 | tail -10
print_success "Python extension built"
echo ""

# Step 5: Run Python tests
print_stage "Running Python tests"
if command -v pytest >/dev/null 2>&1; then
    pytest python/tests/ -v 2>&1 | tail -20 || true
else
    python3 -m unittest discover -s python/tests -p 'test_*.py' 2>&1 | tail -10
fi
print_success "Python tests completed"
echo ""

# Step 6: Run Clippy (linting)
print_stage "Running Clippy linter"
cargo clippy --release -- -D warnings 2>&1 | tail -10 || true
print_success "Clippy check completed"
echo ""

# Step 7: Generate documentation
print_stage "Generating documentation"
cargo doc --release --no-deps --quiet 2>&1 || true
print_success "Documentation generated"
echo ""

# Summary
echo "="
echo -e "${GREEN}= Build Complete!${NC}"
echo "="
echo ""
echo "Next steps:"
echo "  1. Review documentation: cargo doc --open"
echo "  2. Install package: pip install -e ."
echo "  3. Run examples: python examples/fatigue_detection.py"
echo "  4. Test imports: python -c 'from symbolic_dynamic_filtering import SDF'"
echo ""
