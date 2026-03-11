# StochasticSwarm Usage Guide

This guide covers how to build, run, configure, and analyze simulations with StochasticSwarm.

## Table of Contents
- [Building the Project](#building-the-project)
- [Running Simulations](#running-simulations)
- [Configuration](#configuration)
- [Visualization](#visualization)
- [Temperature Experiments](#temperature-experiments)
- [VSCode Integration](#vscode-integration)
- [Troubleshooting](#troubleshooting)

---

## Building the Project

### Using VSCode (Recommended)

1. **Open Command Palette**: `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Linux/Windows)
2. **Run Build Task**: Type "Tasks: Run Build Task" and select it
3. **Choose Build Type**: 
   - `Build StochasticSwarm (Release)` - Default, optimized build
   - `Build StochasticSwarm (Debug)` - Debug build with symbols
   - `Clean Build and Rebuild (Release)` - Fresh build from scratch

**Keyboard Shortcut**: `Cmd+Shift+B` (Mac) or `Ctrl+Shift+B` (Linux/Windows)

### Using Command Line

```bash
# Quick build (recommended)
./build.sh

# Build options
./build.sh --help              # Show all options
./build.sh -d                  # Debug build
./build.sh -c                  # Clean build
./build.sh -r                  # Build and run
./build.sh -c -d -v           # Clean debug build with verbose output
```

### Manual CMake Build

```bash
# Create build directory
mkdir -p build && cd build

# Configure (Release)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Configure (Debug)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Compile
make

# Or compile with parallel jobs
make -j$(nproc)
```

---

## Running Simulations

### From VSCode

**Method 1: Run Task**
1. Open Command Palette (`Cmd+Shift+P`)
2. Select "Tasks: Run Task"
3. Choose "Run Simulation"

**Method 2: Debug/Run**
1. Go to Run and Debug panel (`Cmd+Shift+D`)
2. Select configuration:
   - `Run StochasticSwarm (Release)` - Run optimized build
   - `Debug StochasticSwarm` - Debug with breakpoints
3. Press F5 or click green play button

### From Command Line

```bash
# After building
./build/StochasticSwarm

# Or build and run in one command
./build.sh -r
```

### What to Expect

The simulation will:
1. Display configuration parameters
2. Initialize 5000 particles with thermal velocities
3. Run 1000 timesteps (configurable)
4. Output statistics every 10 steps
5. Save CSV data to `output/` directory

**Sample Output:**
```
╔════════════════════════════════════════════════════╗
║   StochasticSwarm - Week 1: Langevin Dynamics     ║
╚════════════════════════════════════════════════════╝

Simulation Parameters:
  Particles:       5000
  Domain:          100 x 100
  Timestep dt:     0.010

Physics Parameters:
  Damping γ:       1.000
  Temperature T:   1.000
  Mass m:          1.000
  Boltzmann kB:    1.000

Step   10 (t= 0.10):
  Position X: mean=50.001 ±28.456 [0.123, 99.876]
  Position Y: mean=49.998 ±29.102 [0.098, 99.945]
  Velocity X: mean=0.012 ±1.003
  Velocity Y: mean=-0.008 ±0.997
  Average speed: 1.128
```

---

## Configuration

### Editing Parameters

All physical and simulation parameters are centralized in [`include/config.hpp`](../include/config.hpp):

```cpp
namespace Config {
    // Physical parameters
    constexpr float gamma = 1.0f;       // Friction coefficient
    constexpr float kB = 1.0f;          // Boltzmann constant
    constexpr float T = 1.0f;           // Temperature
    constexpr float mass = 1.0f;        // Particle mass
    
    // Simulation parameters
    constexpr float dt = 0.01f;         // Timestep
    constexpr size_t N_particles = 5000; // Number of particles
    constexpr float domain_size = 100.0f; // Domain size
}
```

### Common Modifications

**Change Temperature:**
```cpp
constexpr float T = 10.0f;  // High temperature (gas-like)
constexpr float T = 0.01f;  // Low temperature (frozen)
```

**Change Particle Count:**
```cpp
constexpr size_t N_particles = 10000;  // More particles
```

**Change Domain Size:**
```cpp
constexpr float domain_size = 200.0f;  // Larger domain
```

**Change Timestep:**
```cpp
constexpr float dt = 0.001f;  // Smaller timestep (more accurate, slower)
constexpr float dt = 0.1f;    // Larger timestep (less accurate, faster)
```

**⚠️ Important:** After modifying `config.hpp`, you must **rebuild** the project:
```bash
./build.sh -c  # Clean rebuild
```

### Force Fields

Enable different force fields in [`include/force_field.hpp`](../include/force_field.hpp):

**Option A: Zero Force (Pure Brownian Motion)**
```cpp
return {0.0f, 0.0f};  // Currently active
```

**Option B: Harmonic Potential (Particles attracted to center)**
```cpp
float center_x = 50.0f;
float center_y = 50.0f;
float spring_constant = 0.1f;

float dx = x - center_x;
float dy = y - center_y;

float Fx = -spring_constant * dx;
float Fy = -spring_constant * dy;
return {Fx, Fy};
```

---

## Visualization

### Python Visualization Tool

The included script [`visualize_particles.py`](../visualize_particles.py) provides multiple visualization modes.

**Prerequisites:**
```bash
pip install numpy matplotlib
```

### Visualization Modes

#### 1. Animated GIF
```bash
python visualize_particles.py --mode animate --fps 15
```
- Creates `output/particle_animation.gif`
- Shows system evolution over time

#### 2. Particle Trajectories
```bash
python visualize_particles.py --mode trajectories --num-particles 10
```
- Plots paths of individual particles
- Green dots = start, Red dots = end
- Saves to `output/trajectories.png`

#### 3. Velocity Distribution
```bash
python visualize_particles.py --mode velocities --frame 1000
```
- Histograms of vx, vy, and speed
- Verifies thermal equilibrium
- Should show Gaussian distributions

#### 4. Single Frame Snapshot
```bash
python visualize_particles.py --mode single --frame 500
```
- Static scatter plot at specified frame
- Useful for examining specific moments

### From VSCode

1. Open Command Palette (`Cmd+Shift+P`)
2. Select "Tasks: Run Task"
3. Choose:
   - `Visualize Results (Animate)`
   - `Visualize Results (Trajectories)`

---

## Temperature Experiments

### Automated Script

Run systematic temperature experiments:

```bash
./run_temperature_experiments.sh
```

This script:
1. Runs simulations at T = 0.01, 1.0, and 10.0
2. Automatically rebuilds for each temperature
3. Saves results to separate directories:
   - `output_experiments/low_temp/`
   - `output_experiments/medium_temp/`
   - `output_experiments/high_temp/`

### Expected Behavior

| Temperature | Behavior | Physics Insight |
|-------------|----------|-----------------|
| **T = 0.01** | Particles barely move | Thermal energy << friction |
| **T = 1.0**  | Normal diffusion | Balanced dynamics |
| **T = 10.0** | Rapid spreading | Thermal energy >> friction |

### Analyzing Results

After experiments complete:
```bash
# Compare final states
python visualize_particles.py --mode single --frame 1000

# Analyze velocity distributions
python visualize_particles.py --mode velocities --frame 1000
```

---

## VSCode Integration

### Build Tasks Available

Press `Cmd+Shift+P` → "Tasks: Run Task" → Select:

1. **Build StochasticSwarm (Release)** - Default optimized build
2. **Build StochasticSwarm (Debug)** - Debug build with symbols
3. **Clean Build** - Remove build artifacts
4. **Clean Build and Rebuild (Release)** - Fresh build
5. **Run Simulation** - Build and run
6. **Visualize Results (Animate)** - Create animation
7. **Visualize Results (Trajectories)** - Plot trajectories
8. **Run Temperature Experiments** - Full temperature sweep
9. **Full Pipeline: Build + Run + Visualize** - Complete workflow

### Keyboard Shortcuts

- **Build**: `Cmd+Shift+B` (default build task)
- **Run/Debug**: `F5` (after selecting configuration)
- **Command Palette**: `Cmd+Shift+P`

### Debugging

1. Set breakpoints by clicking line numbers
2. Press `F5` or go to Run and Debug panel
3. Select "Debug StochasticSwarm"
4. Use debug controls:
   - **Continue**: `F5`
   - **Step Over**: `F10`
   - **Step Into**: `F11`
   - **Step Out**: `Shift+F11`

---

## Troubleshooting

### Build Errors

**Problem**: `command not found: cmake`
```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install cmake

# Arch Linux
sudo pacman -S cmake
```

**Problem**: `No such file or directory: config.hpp`
```bash
# Verify file structure
ls include/
# Should show: config.hpp, rng.hpp, particle_system.hpp, force_field.hpp
```

### Runtime Errors

**Problem**: `Warning: Could not open output file`
```bash
# Create output directory
mkdir -p output
```

**Problem**: Particles behave strangely
- Check timestep `dt` in [`config.hpp`](../include/config.hpp)
- If `dt` is too large (>0.1), reduce it
- Rebuild after changes

**Problem**: Simulation too slow
- Reduce `N_particles` in [`config.hpp`](../include/config.hpp)
- Build in Release mode (not Debug)
- Check CPU usage - should be ~100% on one core

### Visualization Errors

**Problem**: `ImportError: No module named 'numpy'`
```bash
pip install numpy matplotlib
```

**Problem**: `FileNotFoundError: output/frame_0.csv`
```bash
# Run simulation first
./build/StochasticSwarm

# Verify CSV files exist
ls output/
```

**Problem**: Animation takes too long
```bash
# Use fewer frames or lower FPS
python visualize_particles.py --mode animate --fps 10
```

---

## Advanced Usage

### Batch Experiments

Create custom experiment scripts:

```bash
#!/bin/bash
for temp in 0.1 0.5 1.0 5.0 10.0; do
    sed -i "s/constexpr float T = [0-9.]*f;/constexpr float T = ${temp}f;/" include/config.hpp
    ./build.sh -c
    ./build/StochasticSwarm
    mv output "output_T_${temp}"
done
```

### Performance Profiling

```bash
# Build with profiling
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-pg" ..
make

# Run and generate profile
./StochasticSwarm
gprof StochasticSwarm gmon.out > profile.txt
```

### Custom Analysis

Read CSV data in Python:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt('output/frame_1000.csv', delimiter=',', skiprows=1)
x, y, vx, vy = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

# Custom analysis
speeds = np.sqrt(vx**2 + vy**2)
print(f"Average speed: {np.mean(speeds):.3f}")
print(f"Max speed: {np.max(speeds):.3f}")
```

---

## Next Steps

- **Week 2**: AVX2 vectorization, MSD/VACF analysis
- **Week 3**: Python bindings (pybind11)
- **Week 4**: RL integration

See [`WEEK1_ROADMAP.md`](../WEEK1_ROADMAP.md) for detailed roadmap.

---

**Need Help?** Check the [main README](../README.md) or review the [Week 1 Roadmap](../WEEK1_ROADMAP.md).
