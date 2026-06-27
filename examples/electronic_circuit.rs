//! Example: Electronic Circuit Analysis
//!
//! Phase 9.1: Demonstrates anomaly detection on Duffing oscillator circuit
//! Example 10.1 from paper: Detects parameter changes in nonlinear circuit

use sdf::wavelets::{WaveletBasis, WaveletTransform};
use sdf::anomaly::measures::{AnomalyMeasure, NormType};

/// Generate Duffing oscillator signal
/// d²y/dt² + β*dy/dt + y + y³ = A*cos(Ωt)
fn generate_duffing_signal(beta: f64, amplitude: f64, omega: f64, steps: usize) -> Vec<f64> {
    let mut signal = Vec::with_capacity(steps);
    let dt = 0.01;
    let mut y = 0.0;
    let mut dy = 0.0;

    for i in 0..steps {
        let t = i as f64 * dt;
        let forcing = amplitude * (omega * t).cos();
        let ddy = forcing - beta * dy - y - y * y * y;
        dy += ddy * dt;
        y += dy * dt;
        signal.push(y);
    }
    signal
}

fn main() -> sdf::Result<()> {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║  Electronic Circuit Anomaly Detection                  ║");
    println!("║  Example 10.1: Duffing Oscillator                      ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Generate nominal circuit signal (β = 0.1, damping coefficient)
    println!("Generating nominal signal (β=0.1, A=5.0, Ω=1.0)...");
    let nominal_signal = generate_duffing_signal(0.1, 5.0, 1.0, 1000);

    // Apply wavelet transform to nominal signal
    let wavelet = WaveletBasis::Morlet;
    let scales = vec![1, 2, 4, 8, 16];

    println!("Applying Continuous Wavelet Transform...");
    println!("  Wavelet: {}", wavelet);
    println!("  Scales:  {:?}\n", scales);

    let nominal_cwt = WaveletTransform::continuous(&nominal_signal, &scales, wavelet)?;
    let nominal_norms = nominal_cwt.compute_scale_norms();

    println!("CWT coefficient matrix: {} × {} (time × scales)", 
        nominal_cwt.coefficients.nrows(),
        nominal_cwt.coefficients.ncols()
    );
    println!("Nominal signal scale norms: {:.6?}\n", nominal_norms);

    // Table header
    println!("╔════════╦═════════════╦═════════════╦═════════════╦══════════════╗");
    println!("║   β    ║  L2 Norm    ║   Angle     ║  KL Div     ║   Status     ║");
    println!("╠════════╬═════════════╬═════════════╬═════════════╬══════════════╣");

    // Test anomaly detection across β parameter range
    let beta_values: Vec<f64> = (10..=35).map(|i| 0.01 * i as f64).collect();
    let mut anomaly_scores = Vec::new();

    for beta in beta_values {
        let test_signal = generate_duffing_signal(beta, 5.0, 1.0, 1000);
        let test_cwt = WaveletTransform::continuous(&test_signal, &scales, wavelet)?;
        let test_norms = test_cwt.compute_scale_norms();

        // Compute anomaly measures (Eq. 37-39 from paper)
        let l2_norm =
            AnomalyMeasure::compute_norm_based(&test_norms, &nominal_norms, NormType::L2)?;
        let angle = AnomalyMeasure::compute_angle(&test_norms, &nominal_norms)?;
        let kl_div = AnomalyMeasure::compute_kullback_leibler(&test_norms, &nominal_norms)?;

        anomaly_scores.push((beta, l2_norm, angle, kl_div));

        let status = if l2_norm > 0.15 { "ANOMALY" } else { "NORMAL" };

        println!(
            "║ {:6.2} ║ {:11.6} ║ {:11.6} ║ {:11.6} ║ {:^12} ║",
            beta, l2_norm, angle, kl_div, status
        );
    }

    println!("╚════════╩═════════════╩═════════════╩═════════════╩══════════════╝");

    // Analysis summary
    println!("\n📊 Analysis Summary:");
    println!("────────────────────────────────");

    // Find transition point
    let mut transition_idx = 0;
    for (i, (_, norm, _, _)) in anomaly_scores.iter().enumerate() {
        if *norm > 0.15 && i > 0 && anomaly_scores[i-1].1 < 0.15 {
            transition_idx = i;
            break;
        }
    }

    if transition_idx > 0 {
        let (beta_transition, _norm, _, _) = anomaly_scores[transition_idx];
        println!("✓ Anomaly detection threshold: β ≈ {:.2}", beta_transition);
        println!("✓ Normal operation: β ≤ 0.10");
        println!("✓ Anomalous region: β > 0.15");
    }

    println!("✓ All {} signals processed successfully", anomaly_scores.len());
    println!("\n✅ Electronic Circuit Anomaly Detection Complete");

    Ok(())
}
