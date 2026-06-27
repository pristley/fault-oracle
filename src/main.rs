//! Main binary for SDF CLI
//!
//! Command-line interface for the Symbolic Dynamic Filtering library.

use sdf::wavelets::{WaveletBasis, ScaleSelector};
use sdf::partitioning::types::Partition;
use sdf::symbolic::{Alphabet, SymbolicEncoder};

fn main() -> sdf::Result<()> {
    println!("Symbolic Dynamic Filtering (SDF) - CLI");
    println!("======================================\n");

    // Example: Create a simple test signal
    let signal = vec![1.0, 2.0, 1.5, 1.0, 0.5, 0.3, 0.5, 1.0, 1.5, 2.0];
    println!("Test signal: {:?}\n", signal);

    // Select wavelet
    let wavelet = WaveletBasis::Morlet;
    println!("Wavelet: {}", wavelet);
    println!("Center frequency: {}", wavelet.center_frequency());
    println!("Vanishing moments: {}\n", wavelet.vanishing_moments());

    // Generate scales
    let selector = ScaleSelector::default();
    let scales = selector.linear_scales();
    println!("Number of scales: {}\n", scales.len());

    // Perform wavelet transform
    match sdf::wavelets::WaveletTransform::continuous(
        &signal,
        &scales,
        wavelet,
    ) {
        Ok(transform) => {
            println!("Wavelet transform computed successfully");
            println!(
                "Coefficients shape: {} × {}",
                transform.coefficients.nrows(),
                transform.coefficients.ncols()
            );

            let norms = transform.compute_scale_norms();
            println!("Scale norms computed: {} values", norms.len());
            println!("Max norm: {:.4}\n", norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        }
        Err(e) => {
            eprintln!("Error in wavelet transform: {}", e);
            return Err(e);
        }
    }

    // Create partition and encoder
    let partition = Partition::new(vec![-0.5, 0.0, 0.5])?;
    let alphabet = Alphabet::new(4)?;
    let encoder = SymbolicEncoder::new(alphabet, partition)?;

    println!("Partition created with {} regions", encoder.partition().num_regions);
    println!("Alphabet size: {}\n", encoder.alphabet().size());

    // Encode signal
    let encoded = encoder.encode_timeseries(&signal)?;
    let sequence: String = encoded.iter().collect();
    println!("Encoded sequence: {}", sequence);

    Ok(())
}
