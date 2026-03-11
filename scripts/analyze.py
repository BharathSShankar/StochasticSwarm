#!/usr/bin/env python3
"""
StochasticSwarm Analysis Tool

Unified script for analyzing simulation output:
- Mean Squared Displacement (MSD)
- Velocity Autocorrelation Function (VACF)
- Trajectory visualization

Usage:
    python scripts/analyze.py msd [path/to/msd_data.csv]
    python scripts/analyze.py vacf [path/to/vacf_data.csv]
    python scripts/analyze.py all [path/to/output_dir]
"""

import sys
import os
from pathlib import Path

# Import individual plotting modules
import plot_msd
import plot_vacf


def find_output_dirs():
    """Find all temperature output directories"""
    output_base = Path("output")
    if not output_base.exists():
        return []
    
    temp_dirs = sorted(output_base.glob("T_*"))
    return [d for d in temp_dirs if d.is_dir()]


def analyze_msd(path=None):
    """Analyze MSD data"""
    if path:
        if not Path(path).exists():
            print(f"Error: File not found: {path}")
            return False
        plot_msd.plot_msd(path)
    else:
        # Find all MSD files
        dirs = find_output_dirs()
        if not dirs:
            print("No output directories found. Run simulation first.")
            return False
        
        for d in dirs:
            msd_file = d / "msd_data.csv"
            if msd_file.exists():
                print(f"\n{'='*60}")
                print(f"Analyzing: {msd_file}")
                print('='*60)
                plot_msd.plot_msd(msd_file)
    return True


def analyze_vacf(path=None):
    """Analyze VACF data"""
    if path:
        if not Path(path).exists():
            print(f"Error: File not found: {path}")
            return False
        plot_vacf.plot_vacf(path)
    else:
        # Find all VACF files
        dirs = find_output_dirs()
        if not dirs:
            print("No output directories found. Run simulation first.")
            return False
        
        for d in dirs:
            vacf_file = d / "vacf_data.csv"
            if vacf_file.exists():
                print(f"\n{'='*60}")
                print(f"Analyzing: {vacf_file}")
                print('='*60)
                plot_vacf.plot_vacf(vacf_file)
    return True


def analyze_all(output_dir=None):
    """Analyze all data in output directory"""
    if output_dir:
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"Error: Directory not found: {output_dir}")
            return False
        
        msd_file = output_path / "msd_data.csv"
        vacf_file = output_path / "vacf_data.csv"
        
        if msd_file.exists():
            print("\n" + "="*60)
            print("MSD Analysis")
            print("="*60)
            plot_msd.plot_msd(msd_file)
        
        if vacf_file.exists():
            print("\n" + "="*60)
            print("VACF Analysis")
            print("="*60)
            plot_vacf.plot_vacf(vacf_file)
    else:
        # Analyze all temperature directories
        dirs = find_output_dirs()
        if not dirs:
            print("No output directories found. Run simulation first.")
            return False
        
        for d in dirs:
            print(f"\n{'#'*60}")
            print(f"# Processing: {d}")
            print('#'*60)
            
            msd_file = d / "msd_data.csv"
            vacf_file = d / "vacf_data.csv"
            
            if msd_file.exists():
                print("\nMSD Analysis:")
                print("-"*60)
                plot_msd.plot_msd(msd_file)
            
            if vacf_file.exists():
                print("\nVACF Analysis:")
                print("-"*60)
                plot_vacf.plot_vacf(vacf_file)
    
    return True


def print_usage():
    """Print usage information"""
    print(__doc__)
    print("\nExamples:")
    print("  # Analyze all MSD files in output/T_* directories")
    print("  python scripts/analyze.py msd")
    print()
    print("  # Analyze specific MSD file")
    print("  python scripts/analyze.py msd output/T_1.00/msd_data.csv")
    print()
    print("  # Analyze all VACF files")
    print("  python scripts/analyze.py vacf")
    print()
    print("  # Analyze all data in a specific directory")
    print("  python scripts/analyze.py all output/T_1.00")
    print()
    print("  # Analyze all temperature directories")
    print("  python scripts/analyze.py all")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if command == "msd":
        success = analyze_msd(path)
    elif command == "vacf":
        success = analyze_vacf(path)
    elif command == "all":
        success = analyze_all(path)
    elif command in ["-h", "--help", "help"]:
        print_usage()
        sys.exit(0)
    else:
        print(f"Error: Unknown command '{command}'")
        print_usage()
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
