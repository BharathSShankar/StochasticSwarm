#!/usr/bin/env python3
"""
Plot Mean Squared Displacement (MSD) with log-log analysis

Usage:
    python plot_msd.py [path/to/msd_data.csv]
    
If no path provided, will look for output/T_*/msd_data.csv files.

Expected behavior for normal Brownian motion:
    MSD ∝ t^α where α ≈ 1.0 (linear on log-log plot with slope = 1)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob


def plot_msd(csv_file, output_dir=None):
    """
    Load and plot MSD data with log-log analysis
    
    Args:
        csv_file: Path to CSV file with columns 'time,msd'
        output_dir: Directory to save plot (defaults to same dir as csv_file)
    """
    # Load data
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    time = data[:, 0]
    msd = data[:, 1]
    
    # Extract temperature from filename if available
    temp_str = ""
    if "T_" in str(csv_file):
        parts = str(csv_file).split("T_")
        if len(parts) > 1:
            temp_val = parts[1].split("/")[0]
            temp_str = f" (T={temp_val})"
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear plot
    ax1.plot(time, msd, 'o-', linewidth=2, markersize=4, alpha=0.7, label='MSD data')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('MSD', fontsize=12)
    ax1.set_title(f'Mean Squared Displacement{temp_str}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Log-log plot for power-law analysis
    # Filter out zero values for log plot
    mask = (time > 0) & (msd > 0)
    time_filtered = time[mask]
    msd_filtered = msd[mask]
    
    ax2.loglog(time_filtered, msd_filtered, 'o', markersize=5, alpha=0.7, label='MSD data')
    
    # Fit power law to later times (avoid ballistic-to-diffusive crossover)
    # The relaxation time τ = m/γ ≈ 2.0 for typical parameters (m=2, γ=1)
    # We need to fit at t >> τ to see pure diffusive behavior
    # Using fit_threshold = 5τ = 10.0 to ensure we're in the diffusive regime
    #
    # IMPORTANT: If you see superdiffusion (α > 1), it likely means:
    #   1. The simulation hasn't run long enough to reach the diffusive regime
    #   2. Try running with more steps: ./StochasticSwarm -s 5000
    fit_threshold = 5.0  # Should be > 2-3× relaxation time (τ = m/γ)
    fit_mask = time_filtered > fit_threshold
    
    # Check if we have enough data in the diffusive regime
    n_fit_points = np.sum(fit_mask)
    t_max = time_filtered.max()
    
    # Warn if simulation is too short
    if t_max < fit_threshold * 2:
        print(f"⚠️  WARNING: Simulation time ({t_max:.1f}) is short compared to fit threshold ({fit_threshold:.1f})")
        print(f"   The relaxation time τ = m/γ ≈ 2.0 for default parameters.")
        print(f"   For clean diffusive behavior, run longer: ./build/StochasticSwarm -s 5000")
        print()
    
    if n_fit_points >= 3:  # Need at least 3 points for a good fit
        time_fit = time_filtered[fit_mask]
        msd_fit = msd_filtered[fit_mask]
        
        # Log-space linear fit: log(MSD) = log(D) + α·log(t)
        coeffs = np.polyfit(np.log(time_fit), np.log(msd_fit), deg=1)
        slope = coeffs[0]  # This is α (diffusion exponent)
        intercept = coeffs[1]  # This is log(D)
        
        # Calculate theoretical diffusion coefficient for reference
        # D_theory = kBT/(m*γ) = T/2 for default params (kB=1, m=2, γ=1)
        D_fit = np.exp(intercept) / 4  # MSD = 4Dt in 2D, so coefficient = 4D
        
        # Generate fit line
        time_range = np.logspace(np.log10(time_fit.min()), np.log10(time_fit.max()), 100)
        fit_line = np.exp(intercept) * time_range**slope
        
        ax2.loglog(time_range, fit_line, '--', linewidth=2.5,
                  label=f'Fit: MSD ∝ t^{slope:.3f}', color='red')
        
        # Add text box with analysis
        textstr = f'Diffusion exponent α = {slope:.3f}\n'
        textstr += f'Fit range: t ∈ [{time_fit.min():.1f}, {time_fit.max():.1f}]\n'
        if abs(slope - 1.0) < 0.1:
            textstr += 'Normal diffusion ✓'
        elif slope < 1.0:
            textstr += f'Subdiffusion (α < 1)'
        else:
            textstr += f'Superdiffusion (α > 1)\n'
            textstr += '(May need longer simulation)'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        # Print diagnostic info
        print(f"   Fit statistics:")
        print(f"   - Exponent α = {slope:.4f} (expect 1.0 for normal diffusion)")
        print(f"   - Fit range: t = {time_fit.min():.2f} to {time_fit.max():.2f}")
        print(f"   - Points in fit: {n_fit_points}")
    else:
        # Insufficient data for fitting
        warning_text = f'Insufficient data for power-law fit\n'
        warning_text += f'Need more points with t > {fit_threshold:.1f}\n'
        warning_text += f'(Current max t = {t_max:.1f})\n\n'
        warning_text += f'Run longer: -s 5000'
        ax2.text(0.5, 0.5, warning_text,
                transform=ax2.transAxes, fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        print(f"   ⚠️  Not enough data points after t > {fit_threshold:.1f}")
        print(f"   Current max time: {t_max:.1f}")
        print(f"   Try running: ./build/StochasticSwarm -s 5000")
    
    # Reference line for normal diffusion (slope = 1)
    time_ref = np.array([time_filtered.min(), time_filtered.max()])
    msd_ref = time_ref * (msd_filtered[0] / time_filtered[0])  # Scale to match data
    ax2.loglog(time_ref, msd_ref, ':', linewidth=1.5, color='gray', 
              alpha=0.5, label='Reference: MSD ∝ t¹')
    
    ax2.set_xlabel('Time (log scale)', fontsize=12)
    ax2.set_ylabel('MSD (log scale)', fontsize=12)
    ax2.set_title(f'Log-Log Analysis{temp_str}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(csv_file).parent
    
    output_file = Path(output_dir) / 'msd_loglog.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {output_file}")
    
    plt.show()


def main():
    """Main function to handle command-line usage"""
    
    if len(sys.argv) > 1:
        # User provided a specific file
        csv_file = sys.argv[1]
        if not Path(csv_file).exists():
            print(f"Error: File not found: {csv_file}")
            sys.exit(1)
        plot_msd(csv_file)
    else:
        # Look for MSD files in output directory
        pattern = "output/T_*/msd_data.csv"
        files = glob.glob(pattern)
        
        if not files:
            print(f"Error: No MSD data files found matching pattern: {pattern}")
            print("\nUsage:")
            print("  python plot_msd.py [path/to/msd_data.csv]")
            print("\nOr run the simulation first to generate MSD data:")
            print("  ./build/StochasticSwarm -T 1.0")
            sys.exit(1)
        
        print(f"Found {len(files)} MSD data file(s):")
        for f in files:
            print(f"  - {f}")
        
        # Plot all files
        for csv_file in files:
            print(f"\nPlotting: {csv_file}")
            plot_msd(csv_file)


if __name__ == "__main__":
    main()
