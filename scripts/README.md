# Analysis Scripts

Python scripts for analyzing StochasticSwarm simulation output.

## Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda install numpy matplotlib
```

## Scripts

### `analyze.py` - Main Analysis Tool

Unified interface for all analysis tasks.

**Usage:**
```bash
# Analyze all MSD files in output directories
python scripts/analyze.py msd

# Analyze specific MSD file
python scripts/analyze.py msd output/T_1.00/msd_data.csv

# Analyze all VACF files
python scripts/analyze.py vacf

# Analyze specific VACF file
python scripts/analyze.py vacf output/T_1.00/vacf_data.csv

# Analyze all data in a directory
python scripts/analyze.py all output/T_1.00

# Analyze all temperature directories
python scripts/analyze.py all
```

### `plot_msd.py` - Mean Squared Displacement

Analyzes particle diffusion behavior.

**Features:**
- Linear and log-log plots
- Power-law fitting: MSD ∝ t^α
- Diffusion coefficient estimation
- Normal/sub/super-diffusion classification

**Usage:**
```bash
python scripts/plot_msd.py [path/to/msd_data.csv]
```

**Expected Results:**
- α ≈ 1.0: Normal Brownian diffusion
- α < 1.0: Subdiffusion (obstacles, crowding)
- α > 1.0: Superdiffusion (may need longer simulation)

### `plot_vacf.py` - Velocity Autocorrelation Function

Analyzes velocity memory decay.

**Features:**
- VACF(τ) decay visualization
- Decay time measurement (1/e point)
- Normalization verification

**Usage:**
```bash
python scripts/plot_vacf.py [path/to/vacf_data.csv]
```

**Expected Results:**
- VACF(0) = 1.0 (perfect normalization)
- Exponential decay: VACF ∝ exp(-γτ)
- Decay time ≈ 1/γ (friction coefficient)

## Output

All plots are saved as PNG files in the same directory as the input CSV:
- `msd_loglog.png` - MSD analysis with log-log fit
- `vacf_plot.png` - VACF decay visualization

## Example Workflow

```bash
# 1. Run simulation
cd build
./StochasticSwarm -T 1.0 -s 5000

# 2. Analyze results
cd ..
python scripts/analyze.py all output/T_1.00

# 3. Compare different temperatures
./build/StochasticSwarm -T 0.01
./build/StochasticSwarm -T 10.0
python scripts/analyze.py all
```

## Physics Background

### Mean Squared Displacement (MSD)
```
MSD(t) = ⟨|x(t) - x(0)|²⟩
```
- Measures how far particles diffuse from initial positions
- Slope on log-log plot gives diffusion exponent α
- Related to diffusion coefficient: D = lim[t→∞] MSD/(4t) in 2D

### Velocity Autocorrelation Function (VACF)
```
VACF(τ) = ⟨v(t)·v(t+τ)⟩ / ⟨v²⟩
```
- Measures how quickly particles forget their velocity direction
- Decay rate controlled by friction coefficient γ
- Related to MSD through Green-Kubo relation

## Troubleshooting

**"No data files found"**
- Run the simulation first: `./build/StochasticSwarm`
- Check that output directory exists

**"Insufficient data for power-law fit"**
- Simulation too short, run with more steps: `-s 10000`
- Relaxation time τ = m/γ, need simulation time > 10τ

**"VACF(0) ≠ 1.0"**
- Normalization issue in VACF computation
- Check velocity history recording in simulation

**Import errors**
- Install dependencies: `pip install numpy matplotlib`
