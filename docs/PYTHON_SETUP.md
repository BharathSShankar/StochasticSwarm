****# Python Visualization Setup Guide

## Using `uv` for Fast Python Package Management

`uv` is an extremely fast Python package installer and resolver, written in Rust. It's a drop-in replacement for `pip` but **10-100x faster**.

---

## 🚀 Quick Start

### Step 1: Install `uv` (if not already installed)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Or via Homebrew:**
```bash
brew install uv
```

**Or via pip:**
```bash
pip install uv
```

**Verify installation:**
```bash
uv --version
```

---

### Step 2: Create Virtual Environment and Install Dependencies

From the project root directory:

```bash
# Create virtual environment using uv (already done!)
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate

# Install dependencies from requirements.txt (using uv, much faster than pip!)
uv pip install -r requirements.txt
```

**What this does:**
- Creates `.venv/` directory with isolated Python environment
- Installs `numpy` and `matplotlib` from `requirements.txt`
- Uses `uv` for blazing-fast installation (~3 seconds vs 30 seconds with pip!)

**To verify installation:**
```bash
python -c "import numpy, matplotlib; print('✓ All dependencies installed!')"
```

---

### Step 3: Make Visualization Script Executable (Optional)

```bash
chmod +x visualize.py
```

---

## 📊 Running the Visualization

### Option 1: After Running C++ Simulation

```bash
# First, compile and run the C++ simulation
cd build
cmake ..
make
./StochasticSwarm

# This creates output/frame_*.csv files

# Now visualize (make sure venv is activated!)
cd ..
python visualize.py
```

### Option 2: Direct Commands

**Plot initial frame:**
```bash
python visualize.py frame 0
```

**Plot final frame:**
```bash
python visualize.py frame 100
```

**Create animation:**
```bash
python visualize.py animate
```

**Interactive mode:**
```bash
python visualize.py
```

---

## 🎨 What the Visualization Shows

### Static Frame Plot
- **Left panel**: Scatter plot of particle positions (x, y)
- **Right panel**: Histogram of particle speeds with mean marked

### Animation
- Shows temporal evolution of particle positions
- Great for observing diffusion patterns
- Can save as MP4 (requires `ffmpeg`)

---

## 🔬 Workflow for Temperature Experiments

To observe temperature effects (Week 1, Step 10):

### 1. **Low Temperature (T=0.01) - Frozen State**

Edit `include/config.hpp`:
```cpp
constexpr float T = 0.01f;
```

```bash
# Recompile and run
cd build && make && ./StochasticSwarm && cd ..

# Visualize
python visualize.py frame 100

# Save results
mv output output_T_0.01
```

### 2. **Medium Temperature (T=1.0) - Normal Diffusion**

Edit `include/config.hpp`:
```cpp
constexpr float T = 1.0f;
```

```bash
cd build && make && ./StochasticSwarm && cd ..
python visualize.py frame 100
mv output output_T_1.0
```

### 3. **High Temperature (T=10.0) - Gas Phase**

Edit `include/config.hpp`:
```cpp
constexpr float T = 10.0f;
```

```bash
cd build && make && ./StochasticSwarm && cd ..
python visualize.py frame 100
mv output output_T_10.0
```

### 4. **Compare Results**

Create a comparison script or manually observe:
- How far particles spread from initial positions
- Velocity distribution width (should scale with √T)
- Visual "energy" of the motion

---

## 📦 Alternative: Using Standard pip

If you prefer not to use `uv`:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install numpy matplotlib

# Run visualization
python visualize.py
```

**Note:** `pip` works fine but is ~10-50x slower than `uv` for large projects.

---

## 🐛 Troubleshooting

### Issue: "numpy not found"
```bash
# Make sure virtual environment is activated!
source .venv/bin/activate  # You should see (.venv) in prompt

# Reinstall
uv pip install numpy matplotlib
```

### Issue: "No module named matplotlib"
```bash
uv pip install matplotlib
```

### Issue: "ffmpeg not found" (when saving animation)
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Or skip saving animation, just view it
```

### Issue: CSV files not found
```bash
# Make sure you ran the C++ simulation first!
ls output/  # Should show frame_*.csv files

# If empty, run:
cd build && ./StochasticSwarm && cd ..
```

---

## 📚 Python Dependencies Explained

From `pyproject.toml`:

| Package | Purpose | Why This Version |
|---------|---------|------------------|
| `numpy>=1.24.0` | Fast array operations, CSV loading | Recent stable version with performance improvements |
| `matplotlib>=3.7.0` | Plotting and animation | Modern API, better performance |

**Optional (for interactive exploration):**
- `ipython>=8.0.0` - Enhanced Python shell for experimentation

---

## 🎓 Learning Tips

### Understanding the Data Format

Each CSV file (`output/frame_N.csv`) has format:
```
x,y,vx,vy
50.2,48.3,0.12,-0.08
51.1,49.7,-0.15,0.21
...
```

- Column 0 (x): X position
- Column 1 (y): Y position
- Column 2 (vx): X velocity
- Column 3 (vy): Y velocity

### Loading Data Manually (for custom analysis)

```python
import numpy as np

# Load frame 100
data = np.loadtxt('output/frame_100.csv', delimiter=',', skiprows=1)

x, y = data[:, 0], data[:, 1]      # Positions
vx, vy = data[:, 2], data[:, 3]    # Velocities

# Calculate speeds
speeds = np.sqrt(vx**2 + vy**2)

# Statistics
print(f"Mean position: ({x.mean():.2f}, {y.mean():.2f})")
print(f"Mean speed: {speeds.mean():.3f}")
print(f"Std speed: {speeds.std():.3f}")

# Should match thermal prediction: <v> ≈ √(kB*T/m)
```

---

## 🚀 Pro Tips

### 1. **Batch Processing Multiple Temperatures**

Create a script `run_experiment.sh`:
```bash
#!/bin/bash
for T in 0.01 0.1 1.0 10.0; do
    # Update config (requires sed or manual edit)
    echo "Running T=$T"
    
    # Compile and run
    cd build && make && ./StochasticSwarm && cd ..
    
    # Save results
    mkdir -p experiments/T_$T
    cp output/*.csv experiments/T_$T/
    
    # Visualize
    python visualize.py frame 100
    mv output/visualization_frame_100.png experiments/T_$T/
done
```

### 2. **Using IPython for Interactive Analysis**

```bash
uv pip install ipython
ipython
```

```python
import numpy as np
import matplotlib.pyplot as plt

# Load and explore
data = np.loadtxt('output/frame_100.csv', delimiter=',', skiprows=1)
x, y, vx, vy = data.T

# Quick plot
plt.scatter(x, y, s=1, alpha=0.5)
plt.show()

# Check thermal distribution
speeds = np.sqrt(vx**2 + vy**2)
plt.hist(speeds, bins=50)
plt.show()
```

### 3. **Performance: uv vs pip**

For this small project, difference is minor. But for larger projects:
- `uv`: ~2-3 seconds to install numpy + matplotlib
- `pip`: ~20-30 seconds

`uv` really shines when you have 50+ dependencies!

---

## ✅ Verification Checklist

Before running experiments:

- [ ] `uv` installed and in PATH (`uv --version` works)
- [ ] Virtual environment created (`.venv/` directory exists)
- [ ] Virtual environment activated (prompt shows `(.venv)`)
- [ ] Dependencies installed (`python -c "import numpy, matplotlib"` succeeds)
- [ ] C++ simulation ran (`ls output/*.csv` shows files)
- [ ] Visualization script runs (`python visualize.py`)

---

## 📖 Additional Resources

- **uv documentation**: https://github.com/astral-sh/uv
- **matplotlib tutorials**: https://matplotlib.org/stable/tutorials/index.html
- **numpy documentation**: https://numpy.org/doc/stable/

---

**Ready to visualize your particle swarm!** 🎉
