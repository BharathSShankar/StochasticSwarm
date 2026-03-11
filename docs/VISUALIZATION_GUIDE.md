# Visualization and GIF Generation Guide

This guide explains how to easily get visualizations and GIFs during training or inference in the StochasticSwarm environment.

## Features

- **Automatic visualization during training** via callbacks
- **Manual frame recording** for custom GIF generation
- **Built-in environment methods** for quick snapshots
- **Flexible rendering** (density maps, particle positions, or combined)
- **TensorBoard integration** for training monitoring

## Quick Start

### 1. Simple Snapshot During Inference

```python
from swarm.envs import SwarmEnv

env = SwarmEnv(task='concentration', num_particles=2000)
obs, info = env.reset()

# Run some steps
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)

# Save a snapshot
env.visualize(mode='combined', save_path='snapshot.png')
```

### 2. Manual GIF Generation

```python
from swarm.envs import SwarmEnv

env = SwarmEnv(task='dispersion', num_particles=1500)
obs, info = env.reset()

# Record frames manually
frames = []
for step in range(50):
    frames.append(env.get_state_dict())
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)

# Create GIF from recorded frames
env.create_gif(
    frames,
    output_path='evolution.gif',
    fps=10,
    mode='combined'
)
```

### 3. Automatic Visualization During Training

```python
from stable_baselines3 import PPO
from swarm.envs import SwarmEnv
from swarm.training.callbacks import VisualizationCallback

env = SwarmEnv(task='concentration', num_particles=1000)

# Setup callback for automatic visualization
callback = VisualizationCallback(
    log_dir='./runs/my_experiment',
    viz_freq=5000,        # Visualize every 5000 steps
    max_frames=100,       # Store up to 100 frames
    save_gif=True,        # Auto-save GIF at end of training
    gif_mode='combined',  # Show density + particles
    gif_fps=10,
    domain_size=env.domain_size,
    verbose=1,
)

# Train with automatic visualization
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000, callback=callback)

# GIF is automatically saved to ./runs/my_experiment/training_evolution.gif
```

## API Reference

### Environment Methods

#### `env.get_state_dict()`

Get current environment state as a dictionary.

```python
state = env.get_state_dict()
# Returns: {
#     'density': np.ndarray,      # 2D density grid
#     'positions': np.ndarray,    # Nx2 particle positions
#     'timestep': int,            # Current step
#     'max_force': float,         # Current max force
#     'temperature': float,       # Temperature
#     'episode_return': float,    # Cumulative reward
# }
```

#### `env.visualize(mode, save_path, title)`

Visualize current state.

**Parameters:**
- `mode`: `'density'`, `'particles'`, or `'combined'` (default)
- `save_path`: Optional path to save image (if None, returns RGB array)
- `title`: Optional custom title

```python
# Save snapshot
env.visualize(mode='combined', save_path='snapshot.png')

# Get RGB array for custom processing
img = env.visualize(mode='density')
```

#### `env.create_gif(frames, output_path, fps, mode)`

Create GIF from recorded frames.

**Parameters:**
- `frames`: List of state dicts from `get_state_dict()`
- `output_path`: Path to save GIF
- `fps`: Frames per second (default: 10)
- `mode`: Visualization mode (default: 'combined')

```python
frames = []
# ... collect frames ...
env.create_gif(frames, 'animation.gif', fps=15, mode='particles')
```

### Utility Functions

```python
from swarm.utils import create_gif, save_snapshot, render_density

# Create GIF from frames
create_gif(
    frames,
    output_path='out.gif',
    fps=10,
    mode='combined',  # or 'density', 'particles'
    domain_size=100.0,
    verbose=True,
)

# Save single snapshot
save_snapshot(
    density=density_grid,
    positions=particle_positions,
    output_path='snapshot.png',
    domain_size=100.0,
    title='Custom Title',
    mode='combined',
    dpi=150,
)

# Render density to RGB array
img = render_density(density_grid, title='Density Map')
```

### VisualizationCallback

Automatically log visualizations during training.

**Parameters:**
- `log_dir`: Directory for TensorBoard logs
- `viz_freq`: Visualization frequency in steps (default: 5000)
- `max_frames`: Maximum frames to store (default: 200)
- `save_gif`: Auto-save GIF at end (default: True)
- `gif_mode`: GIF visualization mode (default: 'combined')
- `gif_fps`: GIF frame rate (default: 10)
- `domain_size`: Physical domain size (default: 100.0)
- `verbose`: Verbosity level (default: 0)

```python
callback = VisualizationCallback(
    log_dir='./runs/experiment',
    viz_freq=1000,
    max_frames=50,
    save_gif=True,
    gif_mode='combined',
    verbose=1,
)

# Access recorded frames
frames = callback.get_frames()

# Manually trigger GIF save
callback.save_gif_now('custom_path.gif')
```

## Visualization Modes

### `'density'` Mode
Shows only the density heatmap.
- Best for: Analyzing density distributions
- Smaller file size

### `'particles'` Mode
Shows particle positions as scatter plot.
- Best for: Tracking individual particles
- Good for lower particle counts

### `'combined'` Mode
Shows both density map and particle positions side-by-side.
- Best for: Complete view of system state
- Larger file size but most informative

## Examples

See [`examples/visualization_demo.py`](../examples/visualization_demo.py) for comprehensive examples including:

1. Manual frame recording and GIF creation
2. Environment built-in visualization methods
3. Training with automatic visualization
4. Inference visualization
5. Snapshot saving at different stages

Run the demo:
```bash
python examples/visualization_demo.py
```

## TensorBoard Integration

When using `VisualizationCallback`, visualizations are automatically logged to TensorBoard:

```bash
tensorboard --logdir=./runs
```

Navigate to the **IMAGES** tab to see:
- `visualization/density` - Density heatmaps over time
- `visualization/particles` - Particle positions over time

## Tips

1. **Reduce GIF size**: Record every Nth frame instead of every step:
   ```python
   if step % 5 == 0:  # Record every 5th step
       frames.append(env.get_state_dict())
   ```

2. **High-quality snapshots**: Use higher DPI for publications:
   ```python
   save_snapshot(density, 'figure.png', dpi=300)
   ```

3. **Memory management**: Limit `max_frames` in callback to avoid memory issues during long training runs.

4. **Performance**: Visualization adds overhead. Use higher `viz_freq` (e.g., 10000) for faster training.

## Requirements

- `matplotlib` - For rendering
- `Pillow` - For GIF creation
- `numpy` - For array operations
- `stable_baselines3` - For training callbacks (optional)

Install with:
```bash
pip install matplotlib Pillow numpy
```

## Troubleshooting

**Issue**: "No module named 'PIL'"
```bash
pip install Pillow
```

**Issue**: GIF creation fails
- Check disk space
- Ensure output directory exists
- Verify frames list is not empty

**Issue**: Visualization is slow
- Reduce `viz_freq`
- Use `mode='density'` instead of `'combined'`
- Reduce particle count or grid resolution
