# StochasticSwarm

A high-performance particle simulation engine with Python bindings for reinforcement learning control. Combines a C++ Langevin dynamics engine with a comprehensive Python RL framework built on Gymnasium, Stable-Baselines3, and PyTorch Lightning.

**Key Features:**
- **C++ Core**: 10,000+ particle Langevin dynamics with SIMD optimization
- **Python Package**: Unified `swarm` package for RL training with multiple tasks
- **PyTorch Lightning**: Custom architecture support with transformers, CNNs, MLPs
- **Zero-Copy Interface**: Direct NumPy access to C++ memory for efficiency
- **Modular Design**: Composable tasks, wrappers, and training configurations

## 🎯 Project Overview

StochasticSwarm simulates particles governed by the **Langevin equation**:
```
dv = -γv·dt + F/m·dt + √(2γkᵦT)·dW
```

Where:
- **γ (gamma)**: Friction/damping coefficient
- **F**: External force field
- **T**: Temperature (controls noise strength)
- **dW**: Wiener process (Brownian noise)

### Features

**Week 1: Stochastic Engine** ✅
- ✅ **Euler-Maruyama integration** for stochastic differential equations
- ✅ **Structure of Arrays (SoA)** memory layout for cache efficiency
- ✅ **PCG random number generator** with Box-Muller Gaussian transform
- ✅ **Periodic boundary conditions** for infinite domain simulation
- ✅ **Thermal equilibrium** with Maxwell-Boltzmann velocity distribution
- ✅ **CSV data export** for scientific analysis
- ✅ **Temperature experiments** to study thermodynamic behavior

**Week 2: Analysis & Optimization** ✅
- ✅ **Mean Squared Displacement (MSD)** analysis with diffusion exponent fitting
- ✅ **Velocity Autocorrelation Function (VACF)** with memory decay measurement
- ✅ **Spatial hashing** for O(k) neighbor queries (70× speedup over O(N²))
- ✅ **ARM NEON SIMD** optimization for vectorized computations
- ✅ **10,000 particle simulation** running smoothly at >100 FPS
- ✅ **Python analysis suite** with unified interface

**Week 3: Python RL Package** ✅
- ✅ **Unified swarm package** consolidating all RL functionality
- ✅ **Multiple task definitions** (concentration, dispersion, corner, pattern)
- ✅ **PyTorch Lightning** support for custom architectures
- ✅ **TensorBoard integration** with visualization callbacks
- ✅ **Curriculum learning** support
- ✅ **Comprehensive training utilities** and presets

## 📋 Requirements

### C++ Build Environment
- **Compiler**: C++17 or later (GCC 7+, Clang 5+, MSVC 2017+)
- **Build System**: CMake 3.10+
- **PyBind11**: For Python bindings (auto-downloaded by CMake)

### Python Environment
- **Python**: 3.8+
- **Core Dependencies**:
  ```bash
  pip install gymnasium numpy
  ```

- **For RL Training** (Stable-Baselines3):
  ```bash
  pip install stable-baselines3 tensorboard
  ```

- **For Custom Architectures** (PyTorch Lightning):
  ```bash
  pip install pytorch-lightning torch
  ```

- **Complete Installation**:
  ```bash
  pip install -r requirements.txt
  ```

## 🚀 Quick Start

### 1. Build C++ Library & Python Bindings

```bash
# Build everything (C++ library + Python module)
./build.sh

# Or manually:
mkdir -p build && cd build
cmake ..
make
cd ..
```

This creates:
- `build/StochasticSwarm` - C++ standalone executable
- `build/stochastic_swarm.*.so` - Python module

### 2. Python Quick Start (RL Training)

```python
from swarm import SwarmEnv, Trainer, TrainingConfig

# Create environment with concentration task
env = SwarmEnv(task='concentration', num_particles=2000)

# Train with PPO (with automatic visualization)
config = TrainingConfig.quick()
config.visualize = True  # Enable auto-visualization

trainer = Trainer(
    env_fn=lambda: SwarmEnv(task='concentration'),
    config=config
)
trainer.train()  # Automatically generates GIF at end
trainer.evaluate()

# Save model
trainer.save('my_model')
```

### 3. Quick Visualization

```python
from swarm import SwarmEnv

env = SwarmEnv(task='concentration', num_particles=2000)
obs, info = env.reset()

# Take some actions
for _ in range(20):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)

# Save snapshot
env.visualize(mode='combined', save_path='snapshot.png')

# Or record frames for GIF
frames = []
for step in range(50):
    frames.append(env.get_state_dict())
    action = env.action_space.sample()
    env.step(action)

# Create GIF
env.create_gif(frames, 'evolution.gif', fps=10)
```

### 3. Run C++ Simulation (Physics Only)

```bash
# Run standalone C++ simulation
./build/StochasticSwarm

# Outputs:
# - Console statistics every 10 steps
# - MSD/VACF analysis data
# - CSV files in output/ directory
```

### 4. Analyze Results

```bash
# Analyze Mean Squared Displacement (diffusion)
python scripts/analyze.py msd

# Analyze Velocity Autocorrelation Function
python scripts/analyze.py vacf

# Analyze all data in output directory
python scripts/analyze.py all output/T_1.00
```

See [`scripts/README.md`](scripts/README.md) for detailed analysis documentation.

## 📁 Project Structure

```
StochasticSwarm/
├── 🔧 C++ Core Engine
│   ├── include/                      # C++ headers
│   │   ├── config.hpp               # Physics parameters (γ, T, dt)
│   │   ├── rng.hpp                  # PCG RNG + Box-Muller Gaussian
│   │   ├── particle_system.hpp     # Langevin dynamics engine (SoA layout)
│   │   ├── force_field.hpp          # Static force computation
│   │   ├── potential_field.hpp      # RL-controllable RBF potential
│   │   ├── density_grid.hpp         # 2D histogram for observations
│   │   ├── spatial_hash.hpp         # O(k) neighbor queries
│   │   ├── analysis.hpp             # MSD & VACF computation
│   │   └── analysis_simd.hpp        # ARM NEON vectorization
│   ├── src/
│   │   └── main.cpp                 # Standalone C++ executable
│   └── bindings/
│       └── bindings.cpp             # PyBind11 Python interface
│
├── 🐍 Python Package (NEW!)
│   └── swarm/                        # Unified RL package
│       ├── __init__.py              # Main API exports
│       ├── envs/                    # Gymnasium environments
│       │   ├── base.py             # SwarmEnv (main environment)
│       │   ├── tasks.py            # Reward functions (concentration, dispersion, etc.)
│       │   ├── curriculum.py       # Curriculum learning wrapper
│       │   └── wrappers.py         # Safety & normalization wrappers
│       ├── training/                # Training infrastructure
│       │   ├── config.py           # TrainingConfig (unified hyperparameters)
│       │   ├── trainer.py          # Trainer (SB3 integration)
│       │   └── callbacks.py        # Visualization, checkpointing, metrics
│       ├── lightning/               # PyTorch Lightning support
│       │   ├── module.py           # PPOModule, ActorCriticModule
│       │   ├── networks.py         # CNN, MLP, Transformer architectures
│       │   ├── data.py             # RolloutBuffer, experience collection
│       │   └── trainer.py          # LightningTrainer
│       └── utils/
│           ├── density.py           # Image → density conversion
│           └── visualization.py     # GIF generation & rendering
│
├── 📚 Documentation & Examples
│   ├── docs/
│   │   ├── RL_TENSORBOARD_TEMPLATE.md      # TensorBoard training guide
│   │   ├── TENSORBOARD_QUICK_REFERENCE.md  # Quick commands
│   │   ├── ADVANCED_FEATURES.md            # Advanced RL features
│   │   ├── VISUALIZATION_GUIDE.md          # Visualization & GIF guide
│   │   └── USAGE.md                        # C++ usage guide
│   ├── examples/
│   │   ├── visualization_demo.py           # Visualization & GIF examples
│   │   ├── tensorboard_training_example.py # Full training demos
│   │   ├── density_grid_demo.py            # Zero-copy interface demo
│   │   └── lightning_custom_architecture.py # Custom network example
│   └── scripts/                     # Analysis tools
│       ├── analyze.py              # MSD/VACF analysis
│       ├── plot_msd.py
│       └── plot_vacf.py
│
├── 🧪 Tests & Benchmarks
│   └── tests/
│       ├── test_python_bindings.py
│       ├── test_swarm_env.py
│       ├── test_density_grid.py
│       ├── benchmark_particle_system.cpp  # step, density, potential, RL pipeline
│       ├── benchmark_spatial_hash.cpp
│       ├── benchmark_spatial_hash_optimized.cpp
│       ├── benchmark_msd.cpp
│       └── benchmark_env.py               # Python / SwarmEnv benchmarks
│
├── ⚙️ Build Configuration
│   ├── CMakeLists.txt              # CMake build system
│   ├── pyproject.toml              # Python package metadata
│   ├── requirements.txt            # Python dependencies
│   └── build.sh                    # Build script
│
└── 📖 Roadmaps
    ├── WEEK1_ROADMAP.md            # Langevin engine
    ├── WEEK2_ROADMAP.md            # Optimization & analysis
    └── WEEK3_ROADMAP.md            # Python bindings & RL
```

---

## 🐍 Python Package Documentation

### Overview

The [`swarm`](swarm/__init__.py:1) package provides a clean, consolidated API for RL control of particle swarms. It replaces the older `python/` directory with a unified, well-structured package.

### Core Components

#### 1. **Environments** ([`swarm.envs`](swarm/envs/__init__.py:1))

**SwarmEnv** - Main Gymnasium environment for swarm control

```python
from swarm import SwarmEnv

# Create environment with task
env = SwarmEnv(
    task='concentration',           # Task: 'concentration', 'dispersion', 'corner', 'pattern'
    num_particles=2000,             # Number of particles
    temperature=1.0,                # Brownian motion temperature
    num_basis=16,                   # Number of RBF basis functions
    grid_resolution=32,             # Observation grid size (32×32)
    physics_steps_per_action=10,    # Physics steps per RL step
    max_steps=100,                  # Episode length
    learnable_max_force=True,       # Include force scaling in action space
    action_smoothing=0.0,           # Exponential smoothing factor
)

# Standard Gym interface
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

**Key Features:**
- **Normalized action space**: Actions in [-1, 1] with learnable force scaling
- **Task-based rewards**: Pluggable reward functions via composition
- **Zero-copy observations**: Direct NumPy access to C++ density grid
- **Temperature coupling**: Force automatically scales with √T

#### 2. **Tasks** ([`swarm.envs.tasks`](swarm/envs/tasks.py:1))

Built-in reward functions:

```python
from swarm.envs.tasks import (
    ConcentrationTask,  # Concentrate particles at center
    DispersionTask,     # Spread particles uniformly
    CornerTask,         # Drive to four corners
    PatternTask,        # Match target density pattern
    CustomTask,         # User-defined reward
)

# Create environment with specific task
env = SwarmEnv(task='concentration')

# Or pass task object for customization
task = ConcentrationTask(threshold=0.9)  # Require 90% in center
env = SwarmEnv(task=task)

# Custom task
def my_reward(density, env):
    center = density[16, 16]
    return center / env.num_particles

env = SwarmEnv(task=CustomTask(my_reward))
```

#### 3. **Training** ([`swarm.training`](swarm/training/__init__.py:1))

**TrainingConfig** - Unified configuration for all algorithms

```python
from swarm import TrainingConfig

# Presets
config = TrainingConfig.quick()     # 50K steps
config = TrainingConfig.medium()    # 500K steps  
config = TrainingConfig.long()      # 1M steps
config = TrainingConfig.massive()   # 10M steps

# Custom configuration
config = TrainingConfig(
    total_timesteps=500_000,
    algorithm='PPO',
    learning_rate=3e-4,
    lr_schedule='cosine',           # 'constant', 'linear', 'cosine'
    n_envs=4,                       # Parallel environments
    batch_size=64,
    n_epochs=10,
    checkpoint_freq=50_000,
    eval_freq=25_000,
    visualize=True,
    tensorboard=True,
)
```

**Trainer** - High-level training interface

```python
from swarm import Trainer, SwarmEnv

trainer = Trainer(
    env_fn=lambda: SwarmEnv(task='concentration'),
    config=TrainingConfig.medium(),
)

# Train
trainer.train()

# Evaluate
stats = trainer.evaluate(n_episodes=10)
print(f"Mean reward: {stats['mean_reward']:.2f}")

# Save/Load
trainer.save('my_model')
trainer.load('my_model.zip')
```

**Supported Algorithms:** PPO, A2C, SAC, TD3

#### 4. **PyTorch Lightning** ([`swarm.lightning`](swarm/lightning/__init__.py:1))

For custom neural network architectures:

```python
from swarm.lightning import PPOModule, ActorCritic, LightningTrainer

# Create custom network with transformer
network = ActorCritic(
    observation_shape=(32, 32),
    action_dim=17,
    network_type='attention',  # 'mlp', 'cnn', 'attention'
)

# Create PPO module
module = PPOModule(
    observation_shape=(32, 32),
    action_dim=17,
    network=network,
    learning_rate=3e-4,
    clip_range=0.2,
)

# Train with Lightning
trainer = LightningTrainer(
    module=module,
    env_fn=lambda: SwarmEnv(task='concentration'),
    max_iterations=100,
)
trainer.train()
```

**Available Networks:**
- **MLPNetwork**: Multi-layer perceptron
- **CNNNetwork**: Convolutional neural network
- **AttentionNetwork**: Transformer/self-attention
- **ActorCritic**: Unified actor-critic wrapper

### Quick Examples

#### Example 1: Quick Training

```python
from swarm import SwarmEnv, Trainer, TrainingConfig

trainer = Trainer(
    env_fn=lambda: SwarmEnv(task='concentration'),
    config=TrainingConfig.quick()
)
trainer.train()
```

#### Example 2: Curriculum Learning

```python
from swarm import CurriculumEnv, Trainer, TrainingConfig

env = CurriculumEnv(
    stages=[
        {'task': 'concentration', 'duration': 100_000},
        {'task': 'dispersion', 'duration': 100_000},
        {'task': 'corner', 'duration': 100_000},
    ]
)

trainer = Trainer(lambda: env, config=TrainingConfig.medium())
trainer.train()
```

#### Example 3: Pattern Matching

```python
from swarm import SwarmEnv
from swarm.utils import image_to_density

# Load target pattern from image
target = image_to_density('path/to/image.png', grid_size=32)

# Train to match pattern
env = SwarmEnv(task='pattern', target_density=target)
# Train as usual...
```

### Migration from `python/` Directory

| Old File | New Location | Usage |
|----------|--------------|-------|
| `python/swarm_env.py` | [`swarm/envs/base.py`](swarm/envs/base.py:1) | `from swarm import SwarmEnv` |
| `python/swarm_env_v2.py` | [`swarm/envs/base.py`](swarm/envs/base.py:1) | Features merged into SwarmEnv |
| `python/rl_template.py` | [`swarm/training/trainer.py`](swarm/training/trainer.py:1) | `from swarm import Trainer` |
| `python/long_training.py` | [`swarm/training/config.py`](swarm/training/config.py:1) | `from swarm import TrainingConfig` |
| `python/image_to_density.py` | [`swarm/utils/density.py`](swarm/utils/density.py:1) | `from swarm.utils import image_to_density` |

---

## 🔧 C++ Core Documentation

### Architecture Overview

The C++ engine uses a **Structure of Arrays (SoA)** memory layout for optimal cache performance and SIMD vectorization.

### Core Classes

#### 1. **ParticleSystem** ([`include/particle_system.hpp`](include/particle_system.hpp:1))

Main simulation class implementing Langevin dynamics:

```
dv = -γv·dt + F/m·dt + √(2γkᵦT/m)·dW
```

**Key Methods:**

```cpp
#include "particle_system.hpp"

// Constructor
ParticleSystem ps(
    5000,      // num_particles
    1.0f,      // temperature
    16,        // num_basis (RBF functions)
    32         // grid_resolution
);

// Initialize particles
ps.initialize_random(100.0f);  // domain_size

// Simulation step (Euler-Maruyama)
ps.step();

// RL interface
ps.set_potential_params(strengths);  // Set RBF amplitudes
ps.update_density_grid();            // Update observation
auto& grid = ps.get_density_grid();  // Get density

// Data access (SoA layout)
const auto& x = ps.get_x();   // X positions
const auto& y = ps.get_y();   // Y positions
const auto& vx = ps.get_vx(); // X velocities
const auto& vy = ps.get_vy(); // Y velocities
```

**Physics Implementation:**
- **Integration**: Euler-Maruyama for SDEs
- **Noise**: Box-Muller Gaussian with correct fluctuation-dissipation coefficient
- **Boundaries**: Periodic wrapping for infinite domain
- **Memory**: SoA layout (`std::vector<float>` per property)

#### 2. **PotentialField** ([`include/potential_field.hpp`](include/potential_field.hpp:1))

Radial Basis Function (RBF) potential for RL control:

```
U(x,y) = Σᵢ Aᵢ · exp(-|x-μᵢ|² / 2σᵢ²)
F(x,y) = -∇U
```

```cpp
#include "potential_field.hpp"

// Create field with N basis functions
PotentialField field(16, 100.0f);  // num_basis, domain_size

// Set strengths (RL action)
std::vector<float> strengths = {10.0, -5.0, ...};
field.set_strengths(strengths);

// Compute force at position
auto [Fx, Fy] = field.compute_force(x, y);

// Advanced: set all parameters
field.set_parameters(centers_x, centers_y, amplitudes, widths);
```

**Design:**
- **Positive amplitude**: Repulsive (hill)
- **Negative amplitude**: Attractive (well)
- **Grid initialization**: Centers uniformly distributed
- **Efficient**: Pre-computed Gaussian basis

#### 3. **DensityGrid** ([`include/density_grid.hpp`](include/density_grid.hpp:1))

2D spatial histogram for RL observations:

```cpp
#include "density_grid.hpp"

// Create 32×32 grid
DensityGrid grid(32, 32, 100.0f);  // nx, ny, domain_size

// Update from particles
grid.update(x_positions, y_positions);

// Access data (row-major, zero-copy for Python)
const auto& data = grid.get_grid();  // std::vector<float>

// Normalize to density (particles per unit area)
grid.normalize();
```

**Features:**
- **Zero-copy Python**: Direct NumPy buffer access via PyBind11
- **Row-major**: Compatible with NumPy default ordering
- **Efficient binning**: O(N) particle → grid mapping

#### 4. **SpatialHash** ([`include/spatial_hash.hpp`](include/spatial_hash.hpp:1))

Uniform grid for O(k) neighbor queries:

```cpp
#include "spatial_hash.hpp"

// Create spatial hash
SpatialHash hash(100.0f, 10);  // domain_size, grid_resolution

// Build hash (do each frame)
hash.clear();
for (size_t i = 0; i < N; ++i) {
    hash.insert(i, x[i], y[i]);
}

// Query neighbors (returns indices in 3×3 cell neighborhood)
auto neighbors = hash.query_neighbors(x, y, radius);

// Check actual distances
for (size_t j : neighbors) {
    float dx = x[j] - x[i];
    float dy = y[j] - y[i];
    float r = sqrt(dx*dx + dy*dy);
    if (r < radius) {
        // Process neighbor
    }
}
```

**Performance:**
- **Speedup**: 70× over O(N²) for 10k particles
- **Query cost**: O(k) where k = average particles per cell
- **Best for**: Interaction forces, collision detection

#### 5. **RNG** ([`include/rng.hpp`](include/rng.hpp:1))

PCG random number generator with Box-Muller transform:

```cpp
#include "rng.hpp"

// Initialize RNG
PCG rng(42);  // seed

// Uniform random [0, 1)
float u = rng.uniform();

// Gaussian random N(μ, σ²)
float g = gaussian_random(rng, mean, stddev);
```

**Implementation:**
- **Algorithm**: PCG XSH-RR (excellent statistical properties)
- **Box-Muller**: Generates pairs of Gaussians, caches second value
- **Performance**: ~40% faster than `std::mt19937`

#### 6. **Analysis** ([`include/analysis.hpp`](include/analysis.hpp:1))

Mean Squared Displacement (MSD) and Velocity Autocorrelation (VACF):

```cpp
#include "analysis.hpp"

// Compute MSD
float msd = compute_msd(x, y, x0, y0, domain_size);

// Compute VACF (requires velocity history)
auto vacf = compute_vacf(vx_history, vy_history, max_lag);

// Save to CSV
save_msd_to_csv(time_points, msd_values, "output/msd.csv");
save_vacf_to_csv(vacf_values, dt, interval, "output/vacf.csv");
```

**SIMD Optimized Version** ([`include/analysis_simd.hpp`](include/analysis_simd.hpp:1)):
- ARM NEON intrinsics for 4-way parallelization
- 1.29× speedup on Apple Silicon

### Configuration

Edit [`include/config.hpp`](include/config.hpp:1):

```cpp
namespace Config {
    // Physics
    constexpr float gamma = 1.0f;         // Friction coefficient
    constexpr float kB = 1.0f;            // Boltzmann constant
    constexpr float T = 1.0f;             // Temperature
    constexpr float mass = 2.0f;          // Particle mass
    
    // Simulation
    constexpr float dt = 0.01f;           // Timestep
    constexpr size_t N_particles = 10000; // Particle count (Week 2: scaled to 10k)
    constexpr float domain_size = 100.0f; // Domain size
    
    // Analysis
    constexpr int total_steps = 5000;
    constexpr int msd_measurement_interval = 10;
    constexpr int output_interval = 500;
}
```

**After changing, rebuild:**
```bash
cd build && make && cd ..
```

### Build System

CMake configuration ([`CMakeLists.txt`](CMakeLists.txt:1)):

- **C++ Executable**: `StochasticSwarm`
- **Python Module**: `stochastic_swarm` (PyBind11)
- **Optimization**: `-O3 -march=native` (SIMD auto-vectorization)
- **Standard**: C++17

Build targets:
```bash
make                   # Build all
make StochasticSwarm  # C++ executable only
make stochastic_swarm # Python module only
```

---

## 📊 Usage Examples

### C++ Standalone Simulation

```cpp
#include "particle_system.hpp"
#include "config.hpp"
#include <iostream>

int main() {
    // Create particle system
    ParticleSystem ps(Config::N_particles, Config::T);
    ps.initialize_random(Config::domain_size);
    
    // Run simulation
    for (int step = 0; step < Config::total_steps; ++step) {
        ps.step();
        
        // Periodic output
        if (step % Config::output_interval == 0) {
            float msd = compute_msd(ps.get_x(), ps.get_y(), 
                                   ps.get_initial_x(), ps.get_initial_y(),
                                   Config::domain_size);
            std::cout << "Step " << step << " | MSD: " << msd << "\n";
        }
    }
    
    return 0;
}
```

### Python RL Training

```python
from swarm import SwarmEnv, Trainer, TrainingConfig

# Quick concentration task
env = SwarmEnv(task='concentration', num_particles=2000)

# Train for 100K steps
trainer = Trainer(
    env_fn=lambda: SwarmEnv(task='concentration'),
    config=TrainingConfig.quick()
)
trainer.train()
stats = trainer.evaluate()
print(f"Final reward: {stats['mean_reward']:.2f}")
```

### TensorBoard Logging

```python
from swarm import Trainer, TrainingConfig, SwarmEnv

config = TrainingConfig(
    total_timesteps=500_000,
    algorithm='PPO',
    tensorboard=True,
    visualize=True,
    viz_freq=5_000,
)

trainer = Trainer(
    env_fn=lambda: SwarmEnv(task='concentration'),
    config=config
)
trainer.train()

# View with: tensorboard --logdir=./runs
```

---

## 🧪 Testing & Benchmarks

### Python Tests
```bash
python tests/test_swarm_env.py        # environment step / reset / reward
python tests/test_density_grid.py     # density binning correctness
python tests/test_python_bindings.py  # C++ binding smoke tests
```

### Python Performance Benchmarks
```bash
python tests/benchmark_env.py              # full suite (~3 min)
python tests/benchmark_env.py --quick      # fast smoke run (~30 s)
python tests/benchmark_env.py --section 3  # scaling sweeps only
```

### C++ Benchmarks (Release build required)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# Individual benchmarks
./build/benchmark_particle_system   # step, density, potential field, RL pipeline
./build/benchmark_spatial_hash      # O(k) vs O(N²) neighbour query
./build/benchmark_spatial_hash_optimized  # grid-resolution sweep
./build/benchmark_msd               # scalar vs ARM NEON SIMD

# Run all at once
cmake --build build --target run_benchmarks
```

### Manual Verification Checklist
- [ ] RNG generates Gaussian distribution (mean≈0, std≈1)
- [ ] Particle count remains constant
- [ ] Velocities thermalize to `√(kB·T/mass)`
- [ ] Positions stay within [0, domain_size) with periodic boundaries
- [ ] Temperature scaling: higher T → larger velocity fluctuations

---

## 📈 Performance

> Measured on Apple M-series (arm64), macOS 26.2, Release build (`-O3 -march=native`).
> Python: 3.12.11 · NumPy 2.4.1 · Stable-Baselines3 2.4.1
> Re-run yourself: `./build/benchmark_particle_system` and `python tests/benchmark_env.py`

### C++ Core Engine

#### Langevin Integrator Throughput

| Particles | Step latency | Throughput |
|----------:|-------------:|-----------:|
| 500 | 8.7 µs | 57.3 M p/s |
| 1 000 | 21.2 µs | 47.2 M p/s |
| 2 000 | 36.7 µs | 54.5 M p/s |
| 5 000 | 76.4 µs | 65.5 M p/s |
| 10 000 | 166.8 µs | 59.9 M p/s |
| 20 000 | 296.4 µs | 67.5 M p/s |
| 50 000 | 736.5 µs | 67.9 M p/s |

Scaling is **O(N) linear** — throughput factor stays within 1.000 ± 0.03 across the full 100×–50 000 range.

#### Density Grid Update (32×32)

| Particles | Update latency | Throughput |
|----------:|---------------:|-----------:|
| 1 000 | 0.64 µs | 1 567 M p/s |
| 5 000 | 3.04 µs | 1 643 M p/s |
| 10 000 | 6.54 µs | 1 529 M p/s |

Grid resolution (16² → 128²) adds ≤ 10 % overhead at 10 000 particles.

#### Potential Field (RBF) — *N = 2 000*

| Basis functions | Step latency | Force evals/sec |
|----------------:|-------------:|----------------:|
| 9 | 31.3 µs | 288 M/s |
| 16 | 38.2 µs | 419 M/s |
| 25 | 56.4 µs | 444 M/s |
| 36 | 70.3 µs | 512 M/s |
| 64 | 105.4 µs | 607 M/s |

#### Full RL Pipeline  (set\_params → N×step → density update)

| Particles | Phys steps | Basis | Grid | Gym step | Steps/sec | Sim-steps/sec |
|----------:|-----------:|------:|-----:|---------:|----------:|--------------:|
| 500 | 5 | 16 | 32² | 0.095 ms | 10 512 | 52 562 |
| 500 | 10 | 16 | 32² | 0.194 ms | 5 165 | 51 649 |
| 2 000 | 10 | 25 | 32² | 1.271 ms | 787 | 7 867 |
| 2 000 | 10 | 25 | 64² | 1.147 ms | 872 | 8 715 |
| 5 000 | 10 | 25 | 32² | 2.899 ms | 345 | 3 450 |
| 5 000 | 10 | 36 | 64² | 3.543 ms | 282 | 2 823 |

---

### Python / SwarmEnv Gym Interface

*(Includes PyBind11 dispatch + NumPy copy overhead)*

#### env.reset() latency

| Particles | Basis | Grid | Reset time |
|----------:|------:|-----:|-----------:|
| 500 | 16 | 32² | 0.01 ms |
| 1 000 | 16 | 32² | 0.01 ms |
| 2 000 | 25 | 32² | 0.03 ms |
| 5 000 | 25 | 32² | 0.07 ms |

#### env.step() latency

| Particles | Physics steps/action | Step time | Gym steps/sec |
|----------:|---------------------:|----------:|--------------:|
| 500 | 5 | 0.10 ms | 9 589 |
| 1 000 | 5 | 0.20 ms | 4 968 |
| 2 000 | 10 | 1.12 ms | 895 |
| 5 000 | 10 | 2.95 ms | 339 |

#### Scaling sweeps — *N=2 000, basis=16, grid=32²*

**`physics_steps_per_action` sweep** — note that simulation throughput (sim-steps/sec) is nearly constant: the Python overhead is dominated by the physics loop, not dispatch.

| Phys steps | Gym step | Gym steps/sec | Sim-steps/sec |
|-----------:|---------:|--------------:|--------------:|
| 1 | 0.08 ms | 11 921 | 11 921 |
| 5 | 0.39 ms | 2 536 | 12 679 |
| 10 | 0.77 ms | 1 291 | 12 913 |
| 20 | 1.52 ms | 657 | 13 141 |
| 50 | 3.78 ms | 264 | 13 211 |

**Grid resolution** has negligible impact on step time:

| Grid | Gym step | Gym steps/sec |
|-----:|---------:|--------------:|
| 16² | 0.78 ms | 1 290 |
| 32² | 0.78 ms | 1 285 |
| 64² | 0.78 ms | 1 285 |
| 128² | 0.79 ms | 1 268 |

**`num_basis` sweep** — scales roughly linearly:

| Basis fns | Gym step | Gym steps/sec |
|----------:|---------:|--------------:|
| 4 | 0.47 ms | 2 125 |
| 16 | 0.77 ms | 1 295 |
| 25 | 1.15 ms | 872 |
| 64 | 2.21 ms | 452 |

#### Task reward computation

| Task | Per call | Calls/sec |
|------|----------:|----------:|
| `ConcentrationTask` | 0.8 µs | 1 253 K/s |
| `DispersionTask` | 6.1 µs | 163 K/s |
| `KLDivergenceTask` | 8.9 µs | 113 K/s |
| `WassersteinTask` | 8 110 µs | 123/s ⚠️ |

> **Note**: `WassersteinTask` uses Earth-Mover Distance (scipy linear programming) and is 1 000× more expensive than KL divergence. Prefer `KLDivergenceTask` for training; use Wasserstein only for final evaluation.

---

### Legacy Benchmarks

- **SpatialHash vs O(N²)**: 70× speedup for 10 000-particle neighbor queries
- **ARM NEON SIMD (MSD)**: 1.29× speedup over scalar on Apple Silicon
- **Memory footprint**: ~320 KB for 10 000 particles (SoA layout)

---

## 🛠️ Development

### Build Options

```bash
# Debug build (with symbols)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..

# With compiler warnings
cmake -DCMAKE_CXX_FLAGS="-Wall -Wextra -pedantic" ..
```

### Code Style
- **Naming**: `snake_case` for variables/functions, `PascalCase` for classes
- **Headers**: Include guards + namespace encapsulation
- **Documentation**: Comments explain physics, not implementation

---

## 📚 Advanced Features

### TensorBoard Integration

Built-in TensorBoard logging for training visualization:

```python
from swarm import Trainer, TrainingConfig, SwarmEnv

config = TrainingConfig(
    tensorboard=True,
    visualize=True,
    viz_freq=5_000,
)

trainer = Trainer(
    env_fn=lambda: SwarmEnv(task='concentration'),
    config=config
)
trainer.train()
```

**View Results:**
```bash
tensorboard --logdir=./runs
# Open http://localhost:6006
```

**Logged Metrics:**
- Episode rewards and lengths
- Policy/value losses
- Entropy and KL divergence
- Density map visualizations
- Action statistics

### Curriculum Learning

Progressive task difficulty:

```python
from swarm import CurriculumEnv, Trainer, TrainingConfig

env = CurriculumEnv(
    stages=[
        {'task': 'concentration', 'duration': 100_000},
        {'task': 'dispersion', 'duration': 100_000},
        {'task': 'corner', 'duration': 100_000},
    ],
    success_threshold=0.8,
    auto_advance=True,
)

trainer = Trainer(lambda: env, config=TrainingConfig.long())
trainer.train()
```

### Custom Architectures

Use PyTorch Lightning for full control:

```python
from swarm.lightning import PPOModule
import torch.nn as nn

# Define custom network
class MyAttentionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Your custom architecture
        self.attention = nn.MultiheadAttention(256, 8)
        # ...
    
    def forward(self, x):
        # Must return: (action_mean, action_std, value)
        # ...

# Train with custom network
module = PPOModule(network=MyAttentionPolicy())
# ... training code
```

### Pattern Matching from Images

```python
from swarm import SwarmEnv
from swarm.utils import image_to_density
import matplotlib.pyplot as plt

# Load image as target pattern
target = image_to_density('smiley.png', grid_size=32)

# Create pattern matching environment
env = SwarmEnv(task='pattern', target_density=target)

# Visualize target
plt.imshow(target, cmap='hot')
plt.title('Target Pattern')
plt.show()
```

---

## 📖 Documentation

- **Visualization & GIF Guide**: [`docs/VISUALIZATION_GUIDE.md`](docs/VISUALIZATION_GUIDE.md) ⭐ NEW
- **RL TensorBoard Guide**: [`docs/RL_TENSORBOARD_TEMPLATE.md`](docs/RL_TENSORBOARD_TEMPLATE.md)
- **Quick Reference**: [`docs/TENSORBOARD_QUICK_REFERENCE.md`](docs/TENSORBOARD_QUICK_REFERENCE.md)
- **Advanced Features**: [`docs/ADVANCED_FEATURES.md`](docs/ADVANCED_FEATURES.md)
- **C++ Usage**: [`docs/USAGE.md`](docs/USAGE.md)
- **Python Package**: [`swarm/README.md`](swarm/README.md)

---

## 📚 References

### Theory
- **Langevin equation**: [Wikipedia](https://en.wikipedia.org/wiki/Langevin_equation)
- **Brownian motion**: [Scholarpedia](http://www.scholarpedia.org/article/Brownian_motion)
- **Euler-Maruyama method**: [Wikipedia](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)

### Implementation
- **PCG Random**: [pcg-random.org](https://www.pcg-random.org/)
- **Box-Muller transform**: [Wikipedia](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
- **Structure of Arrays**: [Agner Fog's optimization guide](https://www.agner.org/optimize/)
- **Stable-Baselines3**: [Documentation](https://stable-baselines3.readthedocs.io/)
- **PyTorch Lightning**: [Documentation](https://lightning.ai/docs/pytorch/)

---

## 📄 License

Educational project - code provided as-is for learning purposes.

---

## 🙏 Acknowledgments

Built following principles from:
- Computational physics (Landau, Páez, Bordeianu)
- Stochastic calculus (Kloeden, Platen)
- High-performance computing best practices

---

**Project Status**: Week 4 In Progress 🚧 | Distribution matching (KL / Wasserstein tasks)

**Quick Start**:
```bash
./build.sh                                           # Build C++ + Python
python -c "from swarm import *; print('Ready!')"     # Verify install
```

For detailed guides, see:
- **Week 1**: [`WEEK1_ROADMAP.md`](WEEK1_ROADMAP.md) - Langevin engine
- **Week 2**: [`WEEK2_ROADMAP.md`](WEEK2_ROADMAP.md) - Optimization & analysis
- **Week 3**: [`WEEK3_ROADMAP.md`](WEEK3_ROADMAP.md) - Python bindings & RL package
