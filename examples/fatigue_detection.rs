//! Example: Fatigue Detection
//!
//! Phase 9.2: Demonstrates fatigue crack detection using ultrasonic signals
//! Example 10.2 from paper: Detects damage progression in vibration data

use sdf::wavelets::{WaveletBasis, WaveletTransform};
use sdf::anomaly::measures::{AnomalyMeasure, NormType};

/// Generate ultrasonic fatigue signal with progressive damage
fn generate_fatigue_signal(cycles: usize, damage_level: f64) -> Vec<f64> {
    let base_freq = 1000.0;
    let dt = 1.0 / (10.0 * base_freq);
    let samples_per_cycle = (base_freq * 10.0) as usize;
    let total_samples = cycles * samples_per_cycle;

    let mut signal = Vec::with_capacity(total_samples);
    for i in 0..total_samples {
        let t = i as f64 * dt;
        // Amplitude increases with damage level
        let amplitude = 1.0 + damage_level * (t / (cycles as f64 * 0.1)).min(1.0);
        let sample = amplitude * (2.0 * std::f64::consts::PI * base_freq * t).sin();
        signal.push(sample);
    }
    signal
}

fn main() -> sdf::Result<()> {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║  Fatigue Damage Detection                              ║");
    println!("║  Example 10.2: Ultrasonic Monitoring                   ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Generate nominal signal (no damage, 5k cycles)
    println!("Generating nominal ultrasonic signal (5k cycles, no damage)...");
    let nominal_signal = generate_fatigue_signal(50, 0.0);

    // Apply wavelet transform
    let wavelet = WaveletBasis::Gaussian(2);
    let scales = vec![2, 4, 8, 16, 32];

    println!("Applying Continuous Wavelet Transform...");
    println!("  Wavelet: {}", wavelet);
    println!("  Center frequency: {:.2} Hz", wavelet.center_frequency());
    println!("  Scales:  {:?}\n", scales);

    let nominal_cwt = WaveletTransform::continuous(&nominal_signal, &scales, wavelet)?;
    let nominal_norms = nominal_cwt.compute_scale_norms();

    println!("Nominal signal scale norms: {:.6?}\n", nominal_norms);

    // Damage progression table
    let damage_stages = vec![
        ("5k (baseline)", 0.0),
        ("30k cycles", 0.2),
        ("45k cycles (visible crack)", 0.5),
        ("60k cycles (propagating)", 0.8),
        ("78k cycles (critical)", 1.0),
    ];

    println!("╔═══════════════════════╦═════════════╦═════════════╦═════════════╗");
    println!("║      Stage            ║  L2 Norm    ║   Angle     ║  KL Div     ║");
    println!("╠═══════════════════════╬═════════════╬═════════════╬═════════════╣");

    let mut anomaly_data = Vec::new();

    for (stage_name, damage_level) in damage_stages {
        let signal = generate_fatigue_signal(50, damage_level);
        let cwt = WaveletTransform::continuous(&signal, &scales, wavelet)?;
        let norms = cwt.compute_scale_norms();

        // Compute anomaly measures (Eq. 37-39 from paper)
        let l2_norm =
            AnomalyMeasure::compute_norm_based(&norms, &nominal_norms, NormType::L2)?;
        let angle = AnomalyMeasure::compute_angle(&norms, &nominal_norms)?;
        let kl_div = AnomalyMeasure::compute_kullback_leibler(&norms, &nominal_norms)?;

        anomaly_data.push((stage_name, damage_level, l2_norm, angle, kl_div));

        println!(
            "║ {:21} ║ {:11.6} ║ {:11.6} ║ {:11.6} ║",
            stage_name, l2_norm, angle, kl_div
        );
    }

    println!("╚═══════════════════════╩═════════════╩═════════════╩═════════════╝");

    // Analysis summary
    println!("\n📊 Damage Detection Analysis:");
    println!("────────────────────────────────");

    let threshold_angle = 0.3;
    let threshold_norm = 0.1;

    for (stage, _, norm, angle, _kl) in &anomaly_data {
        let detected = if angle > &threshold_angle || norm > &threshold_norm {
            "✓ DETECTED"
        } else {
            "○ Not detected"
        };
        println!("  {}: {}", stage, detected);
    }

    // Find crack initiation point
    println!("\n📍 Crack Initiation Analysis:");
    let mut _crack_idx = 0;
    for (i, (_, _, norm, angle, _)) in anomaly_data.iter().enumerate() {
        if (angle > &threshold_angle || norm > &threshold_norm) && i > 0 {
            let prev_detected = anomaly_data[i - 1].3 > threshold_angle
                || anomaly_data[i - 1].2 > threshold_norm;
            if !prev_detected {
                _crack_idx = i;
                println!("✓ Visible crack detected at: {}", anomaly_data[i].0);
                break;
            }
        }
    }

    println!("\n✅ Fatigue Detection Analysis Complete");

    Ok(())
}
