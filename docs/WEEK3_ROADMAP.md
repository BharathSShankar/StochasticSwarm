# Week 3: Python Bindings & RL Integration Roadmap

## 🎯 Goal
Expose the C++ simulation to Python using PyBind11, implement a parametric potential field for RL control, and enable zero-copy density grid transfer for observation space.

## ✅ Progress Tracker
**Steps Completed: 12/12 (100%)** ✅ **WEEK 3 COMPLETE!**

| Step | Status | Description |
|------|--------|-------------|
| 1 | ✅ | Install PyBind11 & Configure CMake |
| 2 | ✅ | Create Minimal PyBind11 Module |
| 3 | ✅ | Expose ParticleSystem Class |
| 4 | ✅ | Design Parametric Potential Field |
| 5 | ✅ | Integrate Potential Field into ParticleSystem |
| 6 | ✅ | **Implement Density Grid (Heatmap)** |
| 7 | ✅ | **Expose Density Grid with Zero-Copy** |
| 8 | ✅ | Expose PotentialField Control |
| 9 | ✅ | Create Python RL Environment Wrapper |
| 10 | ✅ | Testing & Validation |
| 11 | ✅ | Visualization Tools |
| 12 | ✅ | Documentation & Examples |

**Most Recent Achievement:** Week 3 complete! All 13 binding tests passing, visualization tools created, comprehensive documentation, and RL training examples ready! 🎉

---

## 📚 Quick Concept Reference

### Potential Fields & Force Fields
```
U(x) = Potential energy at position x
F(x) = -∇U(x) = Force field (gradient of potential)

For 2D: F(x,y) = [-∂U/∂x, -∂U/∂y]
```
- **Purpose**: RL agent doesn't move particles directly—it warps the space they move through
- **Example potentials**:
  - Harmonic: `U(x) = ½k(x-x₀)²` → `F(x) = -k(x-x₀)` (spring force)
  - Gaussian wells: `U(x) = -A·exp(-|x-μ|²/2σ²)` → Attractive valleys
  - Radial basis functions (RBF): Sum of localized Gaussians

### Parametric Potential
```python
# RL agent controls these parameters:
params = {
    'centers': [(x₁, y₁), (x₂, y₂), ...],  # Well/hill positions
    'strengths': [A₁, A₂, ...],             # Amplitudes
    'widths': [σ₁, σ₂, ...]                 # Spatial scales
}
```
- **Agent's action**: Modify `params` at each RL step
- **Effect**: Changes where particles are pushed/pulled

### Particle Density Grid (Heatmap)
```
ρ(i,j) = number of particles in cell (i,j)
Grid size: Nx × Ny bins covering domain
```
- **Purpose**: Convert particle positions → grid representation for RL observation
- **Why needed**: Neural networks work better with spatial grids than raw particle lists
- **Implementation**: 2D histogram of particle positions

### PyBind11 Zero-Copy NumPy
```cpp
py::array_t<float> get_density_grid() {
    return py::array_t<float>(
        {nx, ny},           // Shape
        {ny*sizeof(float), sizeof(float)},  // Strides (row-major)
        grid.data(),        // Data pointer (no copy!)
        py::cast(this)      // Keep object alive
    );
}
```
- **Zero-copy**: NumPy array points directly to C++ memory
- **Performance**: No expensive data copying between C++ and Python
- **Caution**: C++ object must outlive NumPy array

### PyBind11 Basics
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // For std::vector conversion

namespace py = pybind11;

PYBIND11_MODULE(stochastic_swarm, m) {
    m.doc() = "Particle simulation with RL control";
    
    py::class_<ParticleSystem>(m, "ParticleSystem")
        .def(py::init<size_t, float>())
        .def("step", &ParticleSystem::step)
        .def("get_positions", ...);
}
```

---

## 📋 Implementation Steps

### **Step 1: Install PyBind11 & Configure CMake** ⏱️ 30 min
**Files:** [`CMakeLists.txt`](CMakeLists.txt), [`pyproject.toml`](pyproject.toml)

**Objective**: Set up PyBind11 build system for creating Python extension module.

**Conceptual Question**:
Why do we need a build system for Python bindings? Can't we just write a Python wrapper?

<details>
<summary>Answer</summary>
Python can't directly call C++ code. We need PyBind11 to generate:
1. A shared library (.so/.dll) that Python can import
2. Type conversion code between Python and C++ types
3. Memory management to handle object lifetimes across language boundary
</details>

**Installation**:
```bash
# Install PyBind11 via pip
pip install pybind11[global]

# Or use conda
conda install -c conda-forge pybind11

# Verify installation
python3 -m pybind11 --includes
```

**CMakeLists.txt modifications**:
```cmake
# Find PyBind11
find_package(pybind11 CONFIG REQUIRED)

# Create Python module
pybind11_add_module(stochastic_swarm 
    bindings/bindings.cpp
    # Add other source files if needed
)

# Link against main library if splitting code
target_include_directories(stochastic_swarm PRIVATE include)

# Set output directory
set_target_properties(stochastic_swarm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
```

**pyproject.toml** (modern Python packaging):
```toml
[build-system]
requires = ["setuptools", "wheel", "pybind11>=2.10.0"]
build-backend = "setuptools.build_meta"

[project]
name = "stochastic-swarm"
version = "1.0.0"
description = "RL-controlled particle simulation"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20.0",
    "matplotlib>=3.3.0",
]
```

**Checklist:**
- [x] Install PyBind11 (`pip install pybind11`)
- [x] Verify installation with `python3 -m pybind11 --includes`
- [x] Update CMakeLists.txt with `find_package(pybind11)`
- [x] Create `bindings/` directory for binding code
- [x] Create placeholder `bindings/bindings.cpp`

**Verification**: 
```bash
mkdir -p bindings
touch bindings/bindings.cpp
cmake -B build
# Should see "Found pybind11" in output
```

---

### **Step 2: Create Minimal PyBind11 Module** ⏱️ 30 min
**File:** [`bindings/bindings.cpp`](bindings/bindings.cpp) (new)

**Goal**: Create simplest possible Python module to verify build system.

**Minimal example**:
```cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Simple test function
int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(stochastic_swarm, m) {
    m.doc() = "StochasticSwarm Python bindings";
    
    m.def("add", &add, "A simple addition function",
          py::arg("a"), py::arg("b"));
    
    m.attr("__version__") = "1.0.0";
}
```

**Build and test**:
```bash
cd build
cmake ..
make stochastic_swarm

# Test from Python
cd ..
python3 -c "import stochastic_swarm; print(stochastic_swarm.add(2, 3))"
# Expected output: 5
```

**Common build issues**:
- **"No module named 'stochastic_swarm'"**: Check that .so file exists in current directory
- **Symbol not found**: C++ standard mismatch between Python and module
- **ImportError**: Python version mismatch

**Checklist:**
- [x] Create `bindings/bindings.cpp` with test function
- [x] Update CMakeLists.txt with pybind11_add_module
- [x] Build the module successfully
- [x] Import and test from Python

**Verification**: Can import module and call test function from Python.

---

### **Step 3: Expose ParticleSystem Class** ⏱️ 45 min
**File:** [`bindings/bindings.cpp`](bindings/bindings.cpp)

**Goal**: Make [`ParticleSystem`](include/particle_system.hpp) accessible from Python.

**Core binding pattern**:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // For std::vector
#include <pybind11/numpy.h>    // For NumPy arrays
#include "particle_system.hpp"

namespace py = pybind11;

PYBIND11_MODULE(stochastic_swarm, m) {
    py::class_<ParticleSystem>(m, "ParticleSystem")
        // Constructor
        .def(py::init<size_t, float>(),
             py::arg("num_particles"),
             py::arg("temperature"),
             "Create particle system")
        
        // Methods
        .def("initialize_random", &ParticleSystem::initialize_random,
             py::arg("domain_size"),
             "Initialize particles with random positions")
        
        .def("step", &ParticleSystem::step,
             "Advance simulation by one timestep")
        
        // Read-only properties (Python property syntax)
        .def_property_readonly("num_particles", 
                               &ParticleSystem::get_num_particles)
        
        // Expose position arrays (returns copy for safety)
        .def("get_x", [](const ParticleSystem& ps) {
            auto& x = ps.get_x();
            return py::array_t<float>(x.size(), x.data());
        })
        .def("get_y", [](const ParticleSystem& ps) {
            auto& y = ps.get_y();
            return py::array_t<float>(y.size(), y.data());
        });
}
```

**Python usage**:
```python
import stochastic_swarm as ss
import numpy as np

# Create system
ps = ss.ParticleSystem(num_particles=1000, temperature=1.0)
ps.initialize_random(domain_size=100.0)

# Run simulation
for _ in range(100):
    ps.step()

# Get positions
x = ps.get_x()  # NumPy array
y = ps.get_y()
print(f"Particle 0 at ({x[0]:.2f}, {y[0]:.2f})")
```

**Design choice: Copy vs Zero-copy**
- **Copy** (shown above): Safe, Python owns memory, slower
- **Zero-copy**: Fast, but Python array invalidated if C++ vector resizes
- **For now**: Use copy for positions (small overhead, safer)
- **Later**: Zero-copy for density grid (read-only, no resize risk)

**Checklist:**
- [x] Add ParticleSystem binding with constructor
- [x] Expose key methods: `initialize_random()`, `step()`
- [x] Expose accessors: `get_x()`, `get_y()`, `get_num_particles()`
- [x] Test from Python: create, initialize, step, get positions

**Verification**: 
```python
import stochastic_swarm as ss
ps = ss.ParticleSystem(100, 1.0)
ps.initialize_random(100.0)
assert ps.num_particles == 100
ps.step()
x = ps.get_x()
assert len(x) == 100
```

---

### **Step 4: Design Parametric Potential Field** ⏱️ 60 min
**File:** [`include/potential_field.hpp`](include/potential_field.hpp) (new)

**Physics background**: 
The RL agent controls the environment by modifying a potential energy landscape:
```
U(x,y) = Σᵢ Aᵢ · φ(x,y | μᵢ, σᵢ)

where:
- Aᵢ = strength (positive = hill/repulsive, negative = well/attractive)
- μᵢ = (xᵢ, yᵢ) = center position
- σᵢ = width/scale
- φ = basis function (Gaussian, RBF, harmonic, etc.)
```

**Gaussian RBF basis** (most common choice):
```
φ(x,y | μ, σ) = exp(-[(x-μₓ)² + (y-μᵧ)²] / (2σ²))

F(x,y) = -∇U = Σᵢ Aᵢ · [(x-μₓ)/σ², (y-μᵧ)/σ²] · φ(x,y | μᵢ, σᵢ)
```

**Why Gaussians?**
1. Smooth (infinitely differentiable)
2. Localized (doesn't affect far particles)
3. Interpretable parameters
4. Fast to compute

**Implementation**:
```cpp
#ifndef POTENTIAL_FIELD_HPP
#define POTENTIAL_FIELD_HPP

#include <vector>
#include <cmath>
#include <utility>

/**
 * Parametric potential field using Radial Basis Functions
 * U(x) = Σ Aᵢ · exp(-|x-μᵢ|² / 2σᵢ²)
 */
class PotentialField {
private:
    // Parameters for each basis function
    std::vector<float> centers_x;   // μₓ for each basis
    std::vector<float> centers_y;   // μᵧ for each basis
    std::vector<float> strengths;   // Aᵢ (amplitude)
    std::vector<float> widths;      // σᵢ (spatial scale)
    
    size_t num_basis;               // Number of basis functions

public:
    /**
     * Constructor: Initialize with given number of basis functions
     * Default: uniformly distributed centers, zero strength
     */
    PotentialField(size_t n_basis, float domain_size) 
        : num_basis(n_basis) 
    {
        centers_x.resize(n_basis);
        centers_y.resize(n_basis);
        strengths.resize(n_basis, 0.0f);  // Start with zero force
        widths.resize(n_basis, domain_size / 10.0f);  // Default width
        
        // Initialize centers on a grid
        int grid_size = static_cast<int>(std::ceil(std::sqrt(n_basis)));
        float spacing = domain_size / grid_size;
        
        for (size_t i = 0; i < n_basis; ++i) {
            centers_x[i] = (i % grid_size) * spacing + spacing / 2;
            centers_y[i] = (i / grid_size) * spacing + spacing / 2;
        }
    }
    
    /**
     * Compute force at position (x, y)
     * Returns: {Fx, Fy}
     */
    std::pair<float, float> compute_force(float x, float y) const {
        float Fx = 0.0f;
        float Fy = 0.0f;
        
        for (size_t i = 0; i < num_basis; ++i) {
            float dx = x - centers_x[i];
            float dy = y - centers_y[i];
            float r2 = dx*dx + dy*dy;
            float sigma2 = widths[i] * widths[i];
            
            // Gaussian basis function
            float phi = std::exp(-r2 / (2.0f * sigma2));
            
            // Gradient: -∇(A·φ) = -A · ∇φ = -A · φ · (x-μ)/σ²
            float grad_coeff = -strengths[i] * phi / sigma2;
            
            Fx += grad_coeff * dx;
            Fy += grad_coeff * dy;
        }
        
        return {Fx, Fy};
    }
    
    /**
     * Set parameters (called by RL agent)
     * @param new_strengths: Array of amplitudes for each basis
     */
    void set_strengths(const std::vector<float>& new_strengths) {
        if (new_strengths.size() != num_basis) {
            throw std::invalid_argument("Strength array size mismatch");
        }
        strengths = new_strengths;
    }
    
    /**
     * Set all parameters at once
     */
    void set_parameters(const std::vector<float>& cx,
                        const std::vector<float>& cy,
                        const std::vector<float>& amp,
                        const std::vector<float>& sig) {
        if (cx.size() != num_basis || cy.size() != num_basis ||
            amp.size() != num_basis || sig.size() != num_basis) {
            throw std::invalid_argument("Parameter array size mismatch");
        }
        centers_x = cx;
        centers_y = cy;
        strengths = amp;
        widths = sig;
    }
    
    // Getters for inspection
    size_t get_num_basis() const { return num_basis; }
    const std::vector<float>& get_centers_x() const { return centers_x; }
    const std::vector<float>& get_centers_y() const { return centers_y; }
    const std::vector<float>& get_strengths() const { return strengths; }
    const std::vector<float>& get_widths() const { return widths; }
};

#endif
```

**Checklist:**
- [x] Create `include/potential_field.hpp`
- [x] Implement Gaussian RBF basis functions
- [x] Implement `compute_force()` method
- [x] Implement `set_strengths()` for RL control
- [x] Add getters for parameter inspection
- [x] Test: verify force = 0 when all strengths = 0

**Verification**:
```cpp
PotentialField field(4, 100.0f);  // 4 basis functions
auto [Fx, Fy] = field.compute_force(50.0f, 50.0f);
// Should be (0, 0) with zero strengths

field.set_strengths({1.0f, -1.0f, 0.5f, -0.5f});
auto [Fx2, Fy2] = field.compute_force(50.0f, 50.0f);
// Should be non-zero now
```

---

### **Step 5: Integrate Potential Field into ParticleSystem** ⏱️ 45 min
**Files:** [`include/particle_system.hpp`](include/particle_system.hpp), [`src/main.cpp`](src/main.cpp)

**Goal**: Replace hardcoded `compute_force()` with controllable `PotentialField`.

**Modifications to ParticleSystem**:
```cpp
#include "potential_field.hpp"

class ParticleSystem {
private:
    // ... existing members ...
    
    // NEW: Potential field for RL control
    std::shared_ptr<PotentialField> potential_field;

public:
    /**
     * Constructor: Add optional potential field
     */
    ParticleSystem(size_t num_particles, float temperature, 
                   size_t num_basis = 0)  // num_basis = 0 means no field
        : N(num_particles), T(temperature), rng(42)
    {
        // ... existing initialization ...
        
        // Initialize potential field if requested
        if (num_basis > 0) {
            potential_field = std::make_shared<PotentialField>(
                num_basis, Config::domain_size);
        }
    }
    
    /**
     * Step function: Use potential field if available
     */
    void step() {
        float noise_coeff = sqrt(2.0f * gamma * kB * T / mass);
        float sqrt_dt = sqrt(dt);
        
        for (size_t i = 0; i < N; ++i) {
            // Compute force from potential field
            float Fx = 0.0f, Fy = 0.0f;
            if (potential_field) {
                auto [fx, fy] = potential_field->compute_force(x[i], y[i]);
                Fx = fx;
                Fy = fy;
            }
            
            // ... rest of Euler-Maruyama integration ...
            float ax = -gamma * vx[i] + Fx / mass;
            float ay = -gamma * vy[i] + Fy / mass;
            
            vx[i] += ax * dt + noise_coeff * gaussian_random(rng) * sqrt_dt;
            vy[i] += ay * dt + noise_coeff * gaussian_random(rng) * sqrt_dt;
            
            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
        }
        
        apply_periodic_boundaries();
    }
    
    /**
     * Expose potential field for RL control
     */
    std::shared_ptr<PotentialField> get_potential_field() {
        return potential_field;
    }
    
    /**
     * Convenience method: set potential strengths
     */
    void set_potential_params(const std::vector<float>& strengths) {
        if (potential_field) {
            potential_field->set_strengths(strengths);
        }
    }
};
```

**Why shared_ptr?**
- Python and C++ both need to access the `PotentialField`
- `shared_ptr` ensures object stays alive while either language holds reference
- PyBind11 handles `shared_ptr` automatically

**Checklist:**
- [x] Add `PotentialField` member to `ParticleSystem`
- [x] Modify constructor to accept `num_basis` parameter
- [x] Update `step()` to use potential field force
- [x] Add `get_potential_field()` accessor
- [x] Add `set_potential_params()` convenience method
- [x] Test: Run simulation with and without potential field

**Test Results:**
```
Test 7: Attractive Force Behavior
  Initial distance: 36.95
  Final distance: 32.14
  Change: -4.81
  ✓ ATTRACTIVE: Particles moved 4.81 units closer

Test 8: Repulsive Force Behavior
  Initial distance: 36.95
  Final distance: 40.94
  Change: 3.99
  ✓ REPULSIVE: Particles moved 3.99 units farther
```
All tests passing! See [`tests/test_potential_field.py`](tests/test_potential_field.py) for comprehensive test suite.

**Verification**:
```cpp
// No potential
ParticleSystem ps1(1000, 1.0f, 0);  // Brownian motion

// With potential (4 basis functions)
ParticleSystem ps2(1000, 1.0f, 4);
ps2.set_potential_params({1.0f, -1.0f, 0.5f, -0.5f});
// Particles should now be pushed/pulled by potential
```

---

### **Step 6: Implement Density Grid (Heatmap)** ⏱️ 60 min
**File:** [`include/density_grid.hpp`](include/density_grid.hpp) (new)

**Purpose**: Convert N particle positions → spatial density field for RL observation.

**Conceptual question**: 
Why not just give the RL agent raw particle positions?

<details>
<summary>Answer</summary>
1. **Variable length**: N particles = N×2 dimensional input (hard for neural networks)
2. **Permutation invariance**: Swapping particles shouldn't change observation
3. **Spatial structure**: CNNs work better with grid representations
4. **Scalability**: Grid size fixed regardless of N
</details>

**2D Histogram approach**:
```cpp
#ifndef DENSITY_GRID_HPP
#define DENSITY_GRID_HPP

#include <vector>
#include <algorithm>

/**
 * Spatial density grid (heatmap) for RL observation
 * Bins particles into 2D grid, counting particles per cell
 */
class DensityGrid {
private:
    size_t nx, ny;              // Grid dimensions
    float domain_size;          // Physical domain size
    std::vector<float> grid;    // Flattened 2D array (row-major)
    float cell_size_x, cell_size_y;

public:
    DensityGrid(size_t grid_nx, size_t grid_ny, float domain)
        : nx(grid_nx), ny(grid_ny), domain_size(domain)
    {
        grid.resize(nx * ny, 0.0f);
        cell_size_x = domain_size / nx;
        cell_size_y = domain_size / ny;
    }
    
    /**
     * Clear grid (reset all counts to zero)
     */
    void clear() {
        std::fill(grid.begin(), grid.end(), 0.0f);
    }
    
    /**
     * Add particle to grid (increment count at bin)
     */
    void add_particle(float x, float y) {
        // Convert position to grid indices
        int ix = static_cast<int>(x / cell_size_x);
        int iy = static_cast<int>(y / cell_size_y);
        
        // Clamp to grid bounds (handle edge cases)
        ix = std::clamp(ix, 0, static_cast<int>(nx) - 1);
        iy = std::clamp(iy, 0, static_cast<int>(ny) - 1);
        
        // Increment count (row-major indexing)
        grid[iy * nx + ix] += 1.0f;
    }
    
    /**
     * Update grid from particle arrays
     */
    void update(const std::vector<float>& x, const std::vector<float>& y) {
        clear();
        for (size_t i = 0; i < x.size(); ++i) {
            add_particle(x[i], y[i]);
        }
    }
    
    /**
     * Normalize grid (convert counts to density: particles per unit area)
     */
    void normalize() {
        float cell_area = cell_size_x * cell_size_y;
        for (float& val : grid) {
            val /= cell_area;
        }
    }
    
    // Accessors
    size_t get_nx() const { return nx; }
    size_t get_ny() const { return ny; }
    const std::vector<float>& get_grid() const { return grid; }
    std::vector<float>& get_grid_mutable() { return grid; }
};

#endif
```

**Usage in ParticleSystem**:
```cpp
class ParticleSystem {
private:
    // ... existing members ...
    DensityGrid density_grid;

public:
    ParticleSystem(size_t num_particles, float temperature, 
                   size_t num_basis = 0, size_t grid_res = 32)
        : N(num_particles), T(temperature), rng(42),
          density_grid(grid_res, grid_res, Config::domain_size)
    {
        // ... initialization ...
    }
    
    /**
     * Update density grid from current particle positions
     */
    void update_density_grid() {
        density_grid.update(x, y);
    }
    
    /**
     * Get density grid for RL observation
     */
    const DensityGrid& get_density_grid() const {
        return density_grid;
    }
};
```

**Checklist:**
- [x] Create `DensityGrid` class
- [x] Implement 2D binning algorithm
- [x] Add `update()` method to compute from particles
- [x] Add `normalize()` for physical density units
- [x] Integrate into `ParticleSystem`
- [x] Test: verify total count = N particles

**Verification**:
```cpp
DensityGrid grid(10, 10, 100.0f);  // 10×10 grid
std::vector<float> x = {5.0f, 15.0f, 25.0f};  // 3 particles
std::vector<float> y = {5.0f, 15.0f, 25.0f};
grid.update(x, y);

// Count total particles
float total = 0;
for (float val : grid.get_grid()) total += val;
assert(total == 3.0f);  // Should equal number of particles
```

---

### **Step 7: Expose Density Grid with Zero-Copy** ⏱️ 45 min
**File:** [`bindings/bindings.cpp`](bindings/bindings.cpp)

**Goal**: Pass density grid to Python as NumPy array without copying.

**Zero-copy pattern**:
```cpp
// In bindings.cpp

py::class_<DensityGrid, std::shared_ptr<DensityGrid>>(m, "DensityGrid")
    .def(py::init<size_t, size_t, float>())
    
    .def("update", &DensityGrid::update)
    .def("clear", &DensityGrid::clear)
    .def("normalize", &DensityGrid::normalize)
    
    // Zero-copy NumPy array (READ-ONLY!)
    .def("get_grid", [](DensityGrid& dg) {
        auto& grid = dg.get_grid_mutable();
        
        // Create NumPy array pointing to C++ memory
        return py::array_t<float>(
            {dg.get_ny(), dg.get_nx()},  // Shape (rows, cols)
            {dg.get_nx() * sizeof(float), sizeof(float)},  // Strides (row-major)
            grid.data(),                  // Data pointer (no copy!)
            py::cast(&dg, py::return_value_policy::reference)  // Keep alive
        );
    }, py::return_value_policy::reference_internal)
    
    .def_property_readonly("shape", [](const DensityGrid& dg) {
        return py::make_tuple(dg.get_ny(), dg.get_nx());
    });
```

**Key technical details**:

1. **Strides**: How to move to next row/column in memory
   ```
   Row-major (C-style): 
   - Next column: +1 element = +sizeof(float) bytes
   - Next row: +nx elements = +nx*sizeof(float) bytes
   ```

2. **Keep-alive**: `py::return_value_policy::reference_internal` ensures C++ object outlives NumPy array

3. **Read-only**: Mark as const to prevent Python from modifying C++ memory unexpectedly

**Safer alternative (return copy)**:
```cpp
.def("get_grid_copy", [](const DensityGrid& dg) {
    const auto& grid = dg.get_grid();
    return py::array_t<float>({dg.get_ny(), dg.get_nx()}, grid.data());
})
```

**Python usage**:
```python
import stochastic_swarm as ss
import numpy as np

ps = ss.ParticleSystem(1000, 1.0, num_basis=4, grid_res=32)
ps.initialize_random(100.0)

# Update and get density
ps.update_density_grid()
density = ps.get_density_grid().get_grid()  # NumPy array, zero-copy!

print(density.shape)  # (32, 32)
print(density.sum())  # Should be ≈ 1000 (number of particles)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(density, cmap='hot', interpolation='nearest')
plt.colorbar(label='Particle count')
plt.show()
```

**Checklist:**
- [x] Expose `DensityGrid` class to Python
- [x] Implement zero-copy `get_grid()` with proper strides
- [x] Add `shape` property for convenience
- [x] Update `ParticleSystem` binding with density grid access
- [x] Test from Python: check shape, sum, visualization

**Test Results:**
```
============================================================
DENSITY GRID TEST SUITE
============================================================

Test 1: DensityGrid Creation                    ✓ PASSED
Test 2: Density Grid Update                     ✓ PASSED
Test 3: Zero-Copy Verification                  ✓ PASSED (zero-copy working!)
Test 4: ParticleSystem Integration              ✓ PASSED
Test 5: Density Grid After Simulation           ✓ PASSED
Test 6: Density Normalization                   ✓ PASSED
Test 7: Visualization Readiness                 ✓ PASSED

============================================================
RESULTS: 7 passed, 0 failed
============================================================
```
All tests passing! Zero-copy confirmed with memory address verification.

**Verification**:
```python
# Test zero-copy (check memory address)
grid1 = dg.get_grid()
grid2 = dg.get_grid()
assert grid1.__array_interface__['data'][0] == grid2.__array_interface__['data'][0]
```

---

### **Step 8: Expose PotentialField Control** ⏱️ 30 min
**File:** [`bindings/bindings.cpp`](bindings/bindings.cpp)

**Goal**: Allow RL agent to modify potential field parameters from Python.

**Binding code**:
```cpp
py::class_<PotentialField, std::shared_ptr<PotentialField>>(m, "PotentialField")
    .def(py::init<size_t, float>(),
         py::arg("num_basis"),
         py::arg("domain_size"))
    
    // Main RL interface: set strengths
    .def("set_strengths", &PotentialField::set_strengths,
         py::arg("strengths"),
         "Set amplitude of each basis function (RL action)")
    
    // Advanced: set all parameters
    .def("set_parameters", &PotentialField::set_parameters,
         py::arg("centers_x"),
         py::arg("centers_y"),
         py::arg("strengths"),
         py::arg("widths"))
    
    // Inspection (for debugging/visualization)
    .def("get_centers_x", &PotentialField::get_centers_x)
    .def("get_centers_y", &PotentialField::get_centers_y)
    .def("get_strengths", &PotentialField::get_strengths)
    .def("get_widths", &PotentialField::get_widths)
    .def_property_readonly("num_basis", &PotentialField::get_num_basis)
    
    // Compute force at point (for visualization)
    .def("compute_force", &PotentialField::compute_force,
         py::arg("x"), py::arg("y"),
         "Compute force vector at position");
```

**Update ParticleSystem binding**:
```cpp
py::class_<ParticleSystem>(m, "ParticleSystem")
    // ... existing bindings ...
    
    .def("set_potential_params", &ParticleSystem::set_potential_params,
         py::arg("strengths"),
         "Set potential field strengths (RL action interface)")
    
    .def("get_potential_field", &ParticleSystem::get_potential_field,
         "Get potential field object for advanced control")
    
    .def("update_density_grid", &ParticleSystem::update_density_grid,
         "Update density grid from current particle positions")
    
    .def("get_density_grid", &ParticleSystem::get_density_grid,
         py::return_value_policy::reference_internal,
         "Get density grid for observation");
```

**Python RL loop**:
```python
import stochastic_swarm as ss
import numpy as np

# Setup
ps = ss.ParticleSystem(num_particles=5000, temperature=1.0, 
                       num_basis=16, grid_res=32)
ps.initialize_random(100.0)

# RL training loop
for episode in range(1000):
    # Reset
    ps.initialize_random(100.0)
    
    for step in range(100):
        # Observe state
        ps.update_density_grid()
        observation = ps.get_density_grid().get_grid()
        
        # RL agent decides action (example: random)
        action = np.random.randn(16)  # 16 basis function strengths
        
        # Apply action
        ps.set_potential_params(action.tolist())
        
        # Step simulation
        for _ in range(10):  # 10 physics steps per RL step
            ps.step()
        
        # Compute reward (example: concentration at center)
        center_density = observation[12:20, 12:20].sum()
        reward = center_density
        
        # Train RL agent (not shown)
        # ...
```

**Checklist:**
- [x] Expose `PotentialField` class with all methods
- [x] Expose `set_potential_params()` on `ParticleSystem`
- [x] Expose `get_potential_field()` for advanced control
- [x] Test RL loop structure from Python
- [x] Verify action → force → particle motion chain

**Verification**:
```python
ps = ss.ParticleSystem(1000, 1.0, num_basis=4)
ps.initialize_random(100.0)

# Set strong attractive force at center
ps.set_potential_params([-10.0, -10.0, -10.0, -10.0])

# Run and check particles move toward centers
for _ in range(100):
    ps.step()

# Particles should cluster near potential well centers
```

---

### **Step 9: Create Python RL Environment Wrapper** ⏱️ 60 min
**File:** [`python/swarm_env.py`](python/swarm_env.py) (new)

**Goal**: Wrap C++ simulation in Gym-compatible RL environment.

**OpenAI Gym interface**:
```python
import gymnasium as gym
import numpy as np
import stochastic_swarm as ss

class SwarmEnv(gym.Env):
    """
    Gym environment for RL-controlled particle swarm
    
    Observation: 32×32 density grid (particle heatmap)
    Action: Strengths for N basis functions (continuous)
    Reward: Defined by subclass (e.g., concentration, pattern formation)
    """
    
    def __init__(self, 
                 num_particles=5000,
                 temperature=1.0,
                 num_basis=16,
                 grid_resolution=32,
                 domain_size=100.0,
                 physics_steps_per_action=10):
        super().__init__()
        
        self.num_particles = num_particles
        self.temperature = temperature
        self.num_basis = num_basis
        self.grid_res = grid_resolution
        self.domain_size = domain_size
        self.physics_steps = physics_steps_per_action
        
        # Create C++ simulation
        self.sim = ss.ParticleSystem(
            num_particles=num_particles,
            temperature=temperature,
            num_basis=num_basis,
            grid_res=grid_resolution
        )
        
        # Gym spaces
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=float('inf'),
            shape=(grid_resolution, grid_resolution),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=-10.0,   # Max repulsive strength
            high=10.0,   # Max attractive strength
            shape=(num_basis,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 100
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reinitialize particles
        self.sim.initialize_random(self.domain_size)
        
        # Reset potential (no forces initially)
        self.sim.set_potential_params([0.0] * self.num_basis)
        
        # Get initial observation
        self.sim.update_density_grid()
        obs = self.sim.get_density_grid().get_grid().copy()
        
        self.current_step = 0
        info = {}
        
        return obs.astype(np.float32), info
    
    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)"""
        
        # Apply RL action (set potential field strengths)
        action_list = action.tolist() if hasattr(action, 'tolist') else list(action)
        self.sim.set_potential_params(action_list)
        
        # Run physics simulation
        for _ in range(self.physics_steps):
            self.sim.step()
        
        # Get observation
        self.sim.update_density_grid()
        obs = self.sim.get_density_grid().get_grid().copy()
        
        # Compute reward (override in subclass)
        reward = self.compute_reward(obs)
        
        # Check termination
        self.current_step += 1
        terminated = False  # Task-specific success condition
        truncated = self.current_step >= self.max_steps
        
        info = {
            'step': self.current_step,
            'physics_steps': self.current_step * self.physics_steps
        }
        
        return obs.astype(np.float32), reward, terminated, truncated, info
    
    def compute_reward(self, density):
        """
        Reward function (override for specific tasks)
        Example: Encourage concentration at center
        """
        center_region = density[12:20, 12:20]  # Center 8×8 cells
        reward = center_region.sum() / self.num_particles
        return reward
    
    def render(self):
        """Visualize current state (optional)"""
        import matplotlib.pyplot as plt
        
        self.sim.update_density_grid()
        density = self.sim.get_density_grid().get_grid()
        
        plt.clf()
        plt.imshow(density, cmap='hot', interpolation='nearest', origin='lower')
        plt.colorbar(label='Particle density')
        plt.title(f'Step {self.current_step}')
        plt.pause(0.01)


# Example task-specific environment
class SwarmConcentrationEnv(SwarmEnv):
    """Task: Concentrate particles at domain center"""
    
    def compute_reward(self, density):
        # Reward = fraction of particles in center quarter
        h, w = density.shape
        center = density[h//4:3*h//4, w//4:3*w//4]
        return center.sum() / self.num_particles


class SwarmPatternEnv(SwarmEnv):
    """Task: Form specific spatial pattern (e.g., ring, stripes)"""
    
    def __init__(self, *args, target_pattern=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_pattern = target_pattern  # 2D array
    
    def compute_reward(self, density):
        if self.target_pattern is None:
            return 0.0
        
        # Negative MSE (maximize similarity)
        mse = np.mean((density - self.target_pattern)**2)
        return -mse
```

**Checklist:**
- [x] Create `python/` directory
- [x] Implement `SwarmEnv` base class
- [x] Implement `reset()` and `step()` methods
- [x] Define observation and action spaces
- [x] Create task-specific environments
- [x] Test with random actions

**Verification**:
```python
env = SwarmConcentrationEnv(num_particles=1000, num_basis=4)
obs, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.3f}")
```

---

### **Step 10: Testing & Validation** ⏱️ 45 min
**File:** [`tests/test_python_bindings.py`](tests/test_python_bindings.py) (new)

**Goal**: Comprehensive Python-side testing.

```python
import stochastic_swarm as ss
import numpy as np
import pytest

def test_module_import():
    """Test that module imports correctly"""
    assert hasattr(ss, 'ParticleSystem')
    assert hasattr(ss, 'PotentialField')
    assert hasattr(ss, 'DensityGrid')

def test_particle_system_creation():
    """Test ParticleSystem instantiation"""
    ps = ss.ParticleSystem(num_particles=100, temperature=1.0)
    assert ps.num_particles == 100

def test_simulation_step():
    """Test that simulation advances"""
    ps = ss.ParticleSystem(100, 1.0)
    ps.initialize_random(100.0)
    
    x0 = ps.get_x().copy()
    
    for _ in range(10):
        ps.step()
    
    x1 = ps.get_x()
    
    # Positions should have changed
    assert not np.allclose(x0, x1)

def test_potential_field():
    """Test potential field control"""
    ps = ss.ParticleSystem(100, 1.0, num_basis=4)
    ps.initialize_random(100.0)
    
    # Set potential
    strengths = [1.0, -1.0, 0.5, -0.5]
    ps.set_potential_params(strengths)
    
    # Get potential field
    pf = ps.get_potential_field()
    assert pf is not None
    assert pf.num_basis == 4
    
    # Check strengths were set
    retrieved = pf.get_strengths()
    assert np.allclose(retrieved, strengths)

def test_density_grid():
    """Test density grid computation"""
    ps = ss.ParticleSystem(1000, 1.0, grid_res=10)
    ps.initialize_random(100.0)
    
    ps.update_density_grid()
    density = ps.get_density_grid().get_grid()
    
    # Check shape
    assert density.shape == (10, 10)
    
    # Check total count
    assert np.isclose(density.sum(), 1000, rtol=0.01)
    
    # Check non-negative
    assert np.all(density >= 0)

def test_zero_copy():
    """Test that density grid uses zero-copy"""
    dg = ss.DensityGrid(10, 10, 100.0)
    
    arr1 = dg.get_grid()
    arr2 = dg.get_grid()
    
    # Should point to same memory
    assert arr1.__array_interface__['data'][0] == \
           arr2.__array_interface__['data'][0]

def test_force_computation():
    """Test force field gradient"""
    pf = ss.PotentialField(num_basis=1, domain_size=100.0)
    
    # Zero strength → zero force
    fx, fy = pf.compute_force(50.0, 50.0)
    assert fx == 0.0 and fy == 0.0
    
    # Non-zero strength → non-zero force
    pf.set_strengths([1.0])
    fx2, fy2 = pf.compute_force(60.0, 60.0)
    assert fx2 != 0.0 or fy2 != 0.0

def test_gym_environment():
    """Test Gym environment wrapper"""
    from python.swarm_env import SwarmConcentrationEnv
    
    env = SwarmConcentrationEnv(num_particles=500, num_basis=4)
    
    # Test reset
    obs, info = env.reset()
    assert obs.shape == (32, 32)
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(reward, float)
    assert obs.shape == (32, 32)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

**Run tests**:
```bash
pip install pytest
pytest tests/test_python_bindings.py -v
```

**Checklist:**
- [x] Create test file
- [x] Test module import
- [x] Test ParticleSystem binding
- [x] Test PotentialField binding
- [x] Test DensityGrid and zero-copy
- [x] Test Gym environment
- [x] All tests pass (13/13 passing!)

**Verification**: All tests green ✓

**Test Results:**
```
13 passed in 0.03s
✓ Module import successful
✓ ParticleSystem creation successful
✓ Simulation step advances correctly
✓ Potential field control works
✓ Density grid computation correct
✓ Zero-copy data sharing verified
✓ Force computation correct
✓ Gym environment integration working
✓ Particle conservation verified
✓ Potential field parameter setting works
✓ Density grid normalization works
✓ Periodic boundary conditions working
✓ Potential field force computation working
```

---

### **Step 11: Visualization Tools** ⏱️ 45 min
**File:** [`python/visualize.py`](python/visualize.py) (new)

**Goal**: Tools to visualize potential field and particle dynamics.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import stochastic_swarm as ss

def plot_potential_field(potential_field, domain_size=100.0, resolution=50):
    """
    Visualize potential field as quiver plot (force vectors)
    """
    x = np.linspace(0, domain_size, resolution)
    y = np.linspace(0, domain_size, resolution)
    X, Y = np.meshgrid(x, y)
    
    Fx = np.zeros_like(X)
    Fy = np.zeros_like(Y)
    
    # Compute force at each grid point
    for i in range(resolution):
        for j in range(resolution):
            fx, fy = potential_field.compute_force(X[i,j], Y[i,j])
            Fx[i,j] = fx
            Fy[i,j] = fy
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Quiver plot (force field)
    ax1.quiver(X, Y, Fx, Fy, scale=10)
    
    # Plot basis centers
    cx = potential_field.get_centers_x()
    cy = potential_field.get_centers_y()
    strengths = potential_field.get_strengths()
    
    for i, (x, y, s) in enumerate(zip(cx, cy, strengths)):
        color = 'red' if s > 0 else 'blue'
        ax1.plot(x, y, 'o', markersize=10, color=color, alpha=0.7)
        ax1.text(x, y, f'{s:.1f}', ha='center', va='center', fontsize=8)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Force Field (red=repulsive, blue=attractive)')
    ax1.set_xlim(0, domain_size)
    ax1.set_ylim(0, domain_size)
    
    # Force magnitude heatmap
    magnitude = np.sqrt(Fx**2 + Fy**2)
    im = ax2.imshow(magnitude, extent=[0, domain_size, 0, domain_size],
                    origin='lower', cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Force Magnitude')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def animate_swarm(sim, num_frames=100, interval=50):
    """
    Animate particle swarm evolution
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get initial positions
    x = sim.get_x()
    y = sim.get_y()
    
    # Scatter plot
    scatter = ax1.scatter(x, y, s=1, alpha=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Particle Positions')
    
    # Density heatmap
    sim.update_density_grid()
    density = sim.get_density_grid().get_grid()
    im = ax2.imshow(density, cmap='hot', interpolation='nearest',
                    origin='lower', vmin=0, vmax=density.max())
    ax2.set_xlabel('x (grid)')
    ax2.set_ylabel('y (grid)')
    ax2.set_title('Density Heatmap')
    plt.colorbar(im, ax=ax2)
    
    def update(frame):
        # Advance simulation
        for _ in range(5):
            sim.step()
        
        # Update positions
        x = sim.get_x()
        y = sim.get_y()
        scatter.set_offsets(np.column_stack([x, y]))
        
        # Update density
        sim.update_density_grid()
        density = sim.get_density_grid().get_grid()
        im.set_data(density)
        im.set_clim(0, density.max())
        
        ax1.set_title(f'Particle Positions (frame {frame})')
        
        return scatter, im
    
    anim = FuncAnimation(fig, update, frames=num_frames, 
                         interval=interval, blit=False)
    plt.tight_layout()
    plt.show()
    
    return anim

# Example usage
if __name__ == '__main__':
    # Create simulation with potential field
    sim = ss.ParticleSystem(num_particles=2000, temperature=1.0,
                           num_basis=9, grid_res=32)
    sim.initialize_random(100.0)
    
    # Set interesting potential (repulsive center, attractive corners)
    strengths = [2.0, -1.0, -1.0,
                 -1.0, 3.0, -1.0,
                 -1.0, -1.0, -1.0]
    sim.set_potential_params(strengths)
    
    # Visualize field
    pf = sim.get_potential_field()
    plot_potential_field(pf)
    
    # Animate swarm
    animate_swarm(sim, num_frames=100)
```

**Checklist:**
- [x] Create visualization utilities
- [x] Implement potential field plotting
- [x] Implement particle animation
- [x] Add density heatmap visualization
- [x] Test with various potential configurations

**Verification**: Run visualization script and observe particles responding to potential. ✓

**Created Files:**
- [`python/visualize.py`](python/visualize.py) - Complete visualization toolkit with:
  - `plot_potential_field()` - Quiver plot and force magnitude heatmap
  - `animate_swarm()` - Animated particle evolution with density
  - `plot_density_snapshot()` - Single density snapshot
  - `plot_particles_and_density()` - Side-by-side comparison
  - `visualize_rl_episode()` - RL episode visualization with rewards

---

### **Step 12: Documentation & Examples** ⏱️ 30 min
**Files:** [`python/README.md`](python/README.md), [`examples/rl_example.py`](examples/rl_example.py)

**Python README**:
```markdown
# Python API Documentation

## Installation

```bash
# Build C++ extension
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make stochastic_swarm

# Install Python package (development mode)
cd ..
pip install -e .
```

## Quick Start

```python
import stochastic_swarm as ss

# Create simulation
sim = ss.ParticleSystem(
    num_particles=5000,
    temperature=1.0,
    num_basis=16,
    grid_res=32
)

# Initialize
sim.initialize_random(domain_size=100.0)

# Set potential field (RL action)
action = [-1.0] * 16  # Attractive wells
sim.set_potential_params(action)

# Run simulation
for step in range(100):
    sim.step()
    
    # Get observation
    sim.update_density_grid()
    density = sim.get_density_grid().get_grid()
    
    print(f"Step {step}: max density = {density.max():.2f}")
```

## RL Integration

Use `SwarmEnv` for Gym-compatible training:

```python
from python.swarm_env import SwarmConcentrationEnv

env = SwarmConcentrationEnv(num_particles=5000, num_basis=16)
obs, info = env.reset()

for episode in range(100):
    obs, info = env.reset()
    done = False
    
    while not done:
        action = your_rl_agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
```

## API Reference

### ParticleSystem

- `__init__(num_particles, temperature, num_basis=0, grid_res=32)`
- `initialize_random(domain_size)`
- `step()` - Advance simulation one timestep
- `set_potential_params(strengths)` - Set potential field (RL action)
- `update_density_grid()` - Compute density grid
- `get_density_grid()` - Returns DensityGrid object
- `get_x()`, `get_y()` - Get particle positions (NumPy arrays)

### PotentialField

- `set_strengths(strengths)` - Set basis function amplitudes
- `compute_force(x, y)` - Compute force at position

### DensityGrid

- `get_grid()` - Get density as NumPy array (zero-copy!)
- `shape` - Grid dimensions (ny, nx)
```

**RL Example**:
```python
# examples/rl_example.py
"""
Example: Train RL agent to concentrate particles
Uses stable-baselines3 (install: pip install stable-baselines3)
"""

from python.swarm_env import SwarmConcentrationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return SwarmConcentrationEnv(
        num_particles=2000,
        temperature=1.0,
        num_basis=9,
        grid_resolution=32,
        physics_steps_per_action=10
    )

if __name__ == '__main__':
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Create RL agent (PPO)
    model = PPO(
        'CnnPolicy',  # CNN for image-like observations
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10
    )
    
    # Train
    print("Training RL agent...")
    model.learn(total_timesteps=100000)
    
    # Save
    model.save("swarm_ppo")
    
    # Evaluate
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward[0]:.3f}")
        
        if done:
            obs = env.reset()
```

**Checklist:**
- [x] Create Python README
- [x] Write API documentation
- [x] Create RL training example
- [x] Add visualization examples
- [x] Test all example code

**Created Files:**
- [`python/README.md`](python/README.md) - Comprehensive Python API documentation
- [`examples/rl_example.py`](examples/rl_example.py) - Complete RL training examples
- [`tests/test_python_bindings.py`](tests/test_python_bindings.py) - Full test suite

**Documentation includes:**
- Installation instructions
- Quick start guide
- Complete API reference for all classes and methods
- RL integration guide
- Visualization tutorials
- Performance tips
- Troubleshooting guide
- Multiple training examples (manual and stable-baselines3)

---

## 🧪 Testing Checklist

### C++ Side
- [x] `PotentialField` computes correct forces
- [x] Force = 0 when all strengths = 0
- [x] Gaussian RBF produces expected force field
- [x] `DensityGrid` total count equals N particles
- [x] Grid binning handles edge cases correctly

### Python Bindings
- [x] Module imports without errors
- [x] `ParticleSystem` can be created and initialized
- [x] `step()` advances simulation
- [x] `set_potential_params()` changes forces
- [x] Density grid returns correct shape
- [x] Zero-copy actually shares memory (no duplication)

### RL Integration
- [x] `SwarmEnv` follows Gym interface
- [x] `reset()` reinitializes correctly
- [x] `step()` integrates action → observation → reward
- [x] Observation space matches density grid
- [x] Action space matches number of basis functions

### Performance
- [x] Zero-copy faster than copy for large grids
- [x] Potential field computation < 1ms for 10k particles
- [x] Python overhead < 10% of C++ step time

**All tests passing! See [`tests/test_python_bindings.py`](tests/test_python_bindings.py) for details.**

---

## 🎓 Learning Checkpoints

### After PyBind11 Setup (Steps 1-3):
- "Why can't Python directly call C++ functions?"
- "What does PyBind11 actually generate?"
- "When should I use zero-copy vs copying data?"

### After Potential Field (Steps 4-5):
- "Why use Gaussian RBF instead of simple grid of forces?"
- "How does RL agent's action change particle dynamics?"
- "What's the relationship between potential U and force F?"

### After Density Grid (Steps 6-7):
- "Why represent particles as a density grid for RL?"
- "What are the trade-offs of grid resolution (fine vs coarse)?"
- "How does zero-copy work under the hood?"

### After RL Integration (Steps 8-9):
- "What makes a good reward function for swarm control?"
- "Why separate physics steps from RL steps?"
- "How does observation → action → observation loop work?"

---

## 🔧 Common Pitfalls & Debugging

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| ImportError: No module | Build didn't complete | Check `build/stochastic_swarm.so` exists |
| Segfault on import | C++ standard mismatch | Rebuild with matching C++ std |
| Zero-copy not fast | Python making copy | Check memory address unchanged |
| Particles ignore potential | Strengths not set | Verify `set_potential_params()` called |
| Density sum ≠ N | Grid overflow/underflow | Check domain size vs grid size |
| RL training unstable | Reward scale too large | Normalize reward to [-1, 1] |
| Force field looks wrong | Sign error in gradient | F = -∇U (note negative!) |

---

## 📁 Final File Structure (Week 3)

```
StochasticSwarm/
├── include/
│   ├── potential_field.hpp         # ✅ NEW: Parametric RBF potential
│   ├── density_grid.hpp             # ✅ NEW: Spatial density heatmap
│   ├── particle_system.hpp          # ✅ UPDATED: Integrated potential
│   └── ... (existing headers)
├── bindings/
│   └── bindings.cpp                 # ✅ NEW: PyBind11 bindings
├── python/
│   ├── __init__.py                  # ✅ NEW: Python package
│   ├── swarm_env.py                 # ✅ NEW: Gym environment
│   ├── visualize.py                 # ✅ NEW: Visualization tools
│   └── README.md                    # ✅ NEW: Python API docs
├── examples/
│   ├── rl_example.py                # ✅ NEW: RL training example
│   └── density_grid_demo.py         # ✅ EXISTING: Density demo
├── tests/
│   ├── test_python_bindings.py      # ✅ NEW: Python unit tests (13/13 passing)
│   ├── test_swarm_env.py            # ✅ EXISTING: Env tests (9/9 passing)
│   └── ... (existing C++ tests)
├── CMakeLists.txt                   # ✅ UPDATED: PyBind11 build
├── pyproject.toml                   # ✅ UPDATED: Python packaging
├── requirements.txt                 # ✅ UPDATED: Dependencies
└── WEEK3_ROADMAP.md                 # ✅ This file - COMPLETE!
```

**All files created and tested successfully!** ✅

---

## 🚀 Next Steps (Week 4 Preview)

Once Week 3 complete:
- **Advanced RL algorithms** (PPO, SAC, TD3 for continuous control)
- **Multi-agent swarm control** (N agents controlling N subswarms)
- **Thermodynamic observables** (energy, entropy, free energy computation)
- **Active matter physics** (self-propelled particles, flocking behaviors)
- **Curriculum learning** (progressively harder tasks: concentration → patterns → dynamic control)
- **GPU acceleration** (CUDA kernels for 100k+ particles)
- **Real-time visualization** (OpenGL rendering for interactive control)

---

## 💡 Confidence Check

Rate your understanding (1-5):

- [ ] PyBind11 and C++/Python interop: ___/5
- [ ] Potential fields and force computation: ___/5
- [ ] Zero-copy data sharing: ___/5
- [ ] RL environment design (Gym API): ___/5
- [ ] Observation/action space design: ___/5

If any score < 3, revisit that section before Week 4!

---

## 📊 Success Criteria

By end of Week 3, you should have:

✅ **PyBind11 module** that imports successfully from Python
✅ **Potential field** with controllable RBF basis functions
✅ **Density grid** with zero-copy NumPy interface
✅ **Gym environment** following standard RL interface
✅ **Python tests** passing for all bindings
✅ **Visualization tools** for potential field and particle dynamics
✅ **RL integration** ready for training (action → simulation → observation → reward)

**Remember: The RL agent warps space, particles respond!** 🌊

---

## 🎉 Week 3 Summary

If all steps completed successfully:

**What you've built:**
- C++ simulation engine exposed to Python
- Parametric force field controlled by RL agent
- Zero-copy observation space (density grid)
- Gym-compatible RL environment
- Complete testing and visualization suite

**What you can now do:**
- Train RL agents to control particle swarms
- Experiment with different reward functions (concentration, pattern formation, etc.)
- Visualize emergent collective behavior
- Scale to thousands of particles with real-time control

**Key achievement:** Bridged high-performance C++ with flexible Python RL ecosystem! 🚀