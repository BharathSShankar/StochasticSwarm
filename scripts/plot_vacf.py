#!/usr/bin/env python3
"""
Velocity Autocorrelation Function (VACF) Visualization

Plots VACF(τ) to analyze velocity memory decay in Langevin dynamics.
VACF measures how quickly particles "forget" their velocity direction.

Expected behavior:
- VACF(0) = 1.0 (perfect correlation)
- Exponential decay: VACF ∝ exp(-γ·τ)
- Decay time ≈ 1/γ (friction coefficient)

Usage:
    python plot_vacf.py [VACF_CSV_FILE]
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_vacf(filename):
    """Load and plot VACF data from CSV file."""
    # Load data
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return
    
    if len(data) == 0:
        print(f"Error: No data found in {filename}")
        return
    
    lag = data[:, 0]
    vacf = data[:, 1]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(lag, vacf, 'o-', linewidth=2, markersize=4, label='VACF(τ)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    # Mark decay time (where VACF drops to 1/e)
    threshold = 1.0 / np.e
    plt.axhline(y=threshold, color='gray', linestyle=':', alpha=0.5, 
                label=f'1/e ≈ {threshold:.3f}')
    
    if np.any(vacf < threshold):
        decay_idx = np.where(vacf < threshold)[0][0]
        decay_time = lag[decay_idx]
        plt.axvline(x=decay_time, color='r', linestyle='--', alpha=0.7,
                    label=f'Decay time ≈ {decay_time:.2f}')
        plt.plot(decay_time, vacf[decay_idx], 'ro', markersize=8, zorder=5)
        
        # Add annotation
        plt.annotate(f'τ_decay ≈ {decay_time:.2f}',
                     xy=(decay_time, vacf[decay_idx]),
                     xytext=(decay_time * 1.2, threshold * 1.5),
                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                     fontsize=10, color='red')
    
    # Labels and formatting
    plt.xlabel('Time Lag τ', fontsize=12)
    plt.ylabel('VACF(τ)', fontsize=12)
    plt.title('Velocity Autocorrelation Function\nMeasures velocity memory decay', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='best')
    
    # Add text box with physics interpretation
    textstr = '\n'.join([
        'VACF(0) = 1: Perfect correlation',
        'VACF → 0: Velocity "forgotten"',
        'Decay ∝ exp(-γτ) for friction γ'
    ])
    plt.text(0.98, 0.97, textstr, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.dirname(filename)
    if not output_dir:
        output_dir = '.'
    output_file = os.path.join(output_dir, 'vacf_plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ VACF plot saved to {output_file}")
    
    # Show plot
    plt.show()
    
    # Print analysis
    print("\n" + "="*50)
    print("VACF Analysis Summary")
    print("="*50)
    print(f"VACF(0):              {vacf[0]:.6f} (expected: 1.0)")
    print(f"Number of lags:       {len(vacf)}")
    print(f"Maximum lag time:     {lag[-1]:.2f}")
    
    if np.any(vacf < threshold):
        print(f"Decay time (1/e):     {decay_time:.2f}")
        print(f"This represents the timescale over which particles")
        print(f"lose memory of their initial velocity direction.")
    else:
        print("Decay time:           Not reached (extend simulation)")
    
    # Check if VACF(0) is close to 1.0
    if abs(vacf[0] - 1.0) > 0.01:
        print(f"\n⚠ Warning: VACF(0) = {vacf[0]:.3f} (expected 1.0)")
        print("  This may indicate a normalization issue.")
    else:
        print(f"\n✓ VACF(0) = {vacf[0]:.3f} (correct normalization)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Default: look for VACF file in output directory
        # Check different temperature directories
        possible_files = [
            'output/T_1.00/vacf_data.csv',
            'output/T_0.01/vacf_data.csv',
            'output/T_10.00/vacf_data.csv',
            'build/output/T_1.00/vacf_data.csv',
            '../output/T_1.00/vacf_data.csv',
        ]
        
        filename = None
        for f in possible_files:
            if os.path.exists(f):
                filename = f
                print(f"Found VACF data: {filename}")
                break
        
        if filename is None:
            print("Usage: python plot_vacf.py [VACF_CSV_FILE]")
            print("\nExample:")
            print("  python plot_vacf.py output/T_1.00/vacf_data.csv")
            print("\nNo default VACF file found. Please specify the file path.")
            sys.exit(1)
    
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        sys.exit(1)
    
    plot_vacf(filename)
