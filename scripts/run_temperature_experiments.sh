#!/bin/bash
# Temperature Experiment Script for Week 1 Step 10
# Runs simulations with different temperatures to observe thermal effects

set -e

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  StochasticSwarm - Temperature Experiments (Step 10)  ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Create output directories
mkdir -p output_experiments/low_temp
mkdir -p output_experiments/medium_temp
mkdir -p output_experiments/high_temp

# Store original output directory
mkdir -p output_backup
if [ "$(ls -A output 2>/dev/null)" ]; then
    echo "Backing up existing output files..."
    mv output/* output_backup/ 2>/dev/null || true
fi

echo "This script will run 3 temperature experiments:"
echo "  1. Low T (0.01)  - Particles should barely move (frozen)"
echo "  2. Medium T (1.0) - Normal diffusion"
echo "  3. High T (10.0)  - Violent random motion (gas-like)"
echo ""
echo "Each simulation runs 1000 timesteps."
echo ""

# Function to update config.hpp temperature
update_temperature() {
    local temp=$1
    echo "Setting temperature T = $temp in config.hpp..."
    
    # Use sed to update the temperature value
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/constexpr float T = [0-9.]*f;/constexpr float T = ${temp}f;/" include/config.hpp
    else
        # Linux
        sed -i "s/constexpr float T = [0-9.]*f;/constexpr float T = ${temp}f;/" include/config.hpp
    fi
}

# Function to run simulation
run_simulation() {
    local temp=$1
    local label=$2
    local output_dir=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $label (T = $temp)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Update temperature in config
    update_temperature $temp
    
    # Rebuild
    echo "Recompiling with new temperature..."
    cd build
    cmake .. > /dev/null 2>&1
    make > /dev/null 2>&1
    cd ..
    
    echo "Running simulation..."
    ./build/StochasticSwarm | tail -20
    
    # Move output files
    echo "Moving results to $output_dir..."
    mv output/* $output_dir/
    
    echo "✓ $label complete!"
    echo ""
}

# Run experiments
run_simulation "0.01" "Experiment 1: Low Temperature (Frozen)" "output_experiments/low_temp"
run_simulation "1.0" "Experiment 2: Medium Temperature (Normal)" "output_experiments/medium_temp"
run_simulation "10.0" "Experiment 3: High Temperature (Gas)" "output_experiments/high_temp"

echo "╔═══════════════════════════════════════════════════════╗"
echo "║              All Experiments Complete!                ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to:"
echo "  - output_experiments/low_temp/      (T=0.01)"
echo "  - output_experiments/medium_temp/   (T=1.0)"
echo "  - output_experiments/high_temp/     (T=10.0)"
echo ""
echo "Next steps:"
echo "  1. Visualize with Python:"
echo "     python visualize_particles.py --mode animate"
echo "     python visualize_particles.py --mode trajectories"
echo "     python visualize_particles.py --mode velocities --frame 1000"
echo ""
echo "  2. Compare the three experiments:"
echo "     - Low T: Particles should barely diffuse"
echo "     - Med T: Normal Brownian motion"
echo "     - High T: Rapid spreading (higher velocities)"
echo ""
echo "  3. Restore original temperature:"
echo "     The config.hpp has been modified. Set T back to your"
echo "     preferred value and recompile."
echo ""

# Restore original output directory
if [ "$(ls -A output_backup 2>/dev/null)" ]; then
    echo "Restoring original output files..."
    mv output_backup/* output/ 2>/dev/null || true
fi
rmdir output_backup 2>/dev/null || true

echo "Temperature experiments complete! 🎉"
