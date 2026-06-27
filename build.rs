fn main() {
    // Build script for Python bindings
    // This is used during compilation to set up PyO3 bindings
    println!("cargo:rerun-if-changed=src/python_bindings.rs");
}
