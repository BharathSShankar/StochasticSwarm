# Advanced Features (v2.0)

This document describes the new features added in v2.0 of StochasticSwarm:

1. **Image-to-Density Grid Conversion** - Upload images and convert them to target density patterns
2. **Curriculum Learning** - Progressive difficulty training to prevent failure
3. **Learnable Max Force** - Adaptive force scaling with normalized action space
4. **Long Training Runs** - Configurations for extended training sessions

## Table of Contents

- [Image-to-Density Conversion](#image-to-density-conversion)
- [Curriculum Learning](#curriculum-learning)
- [Learnable Max Force](#learnable-max-force)
- [Long Training Runs](#long-training-runs)
- [Anti-Failure Wrapper](#anti-failure-wrapper)
- [Quick Start Examples](#quick-start-examples)

---

## Image-to-Density Conversion

Convert images into target density grids that the RL agent can learn to match.

### Basic Usage

```python
from python.image_to_density import image_to_density_grid, create_pattern

# From an image file
density = image_to_density_grid(
    'path/to/image.png',
    grid_resolution=32,
    total_particles=5000,
    invert=False,  # Dark = low density, Light = high density
)

# From predefined patterns
density = create_pattern(
    'ring',  # Options: 'ring', 'gaussian', 'corners', 'cross', 
             #          'stripes', 'checkerboard', 'spiral', 'center'
    grid_resolution=32,
    total_particles=5000,
)
```

### Using with Environment

```python
from python.swarm_env_v2 import SwarmImageTargetEnv

# Create environment with image target
env = SwarmImageTargetEnv(
    target_image='path/to/target.png',  # Image file
    # OR
    target_image='pattern:ring',  # Predefined pattern
    # OR
    target_image=numpy_array,  # Direct numpy array
    
    error_metric='mse',  # Options: 'mse', 'mae', 'correlation'
    success_threshold=0.85,  # Correlation threshold for success
    invert_image=False,
)

# Change target during training
env.set_target('pattern:spiral')

# Get target density for visualization
target = env.get_target_density()
```

### Comparing Densities

```python
from python.image_to_density import compute_density_error, visualize_density_comparison

# Compute error metrics
mse = compute_density_error(current, target, 'mse')
correlation = compute_density_error(current, target, 'correlation')

# Visualize comparison (saves figure)
visualize_density_comparison(current, target, save_path='comparison.png')
```

---

## Curriculum Learning

Progressive difficulty training that prevents the AI from getting stuck.

### Basic Usage

```python
from python.swarm_env_v2 import CurriculumSwarmEnv

# Use default curriculum (8 stages from easy to hard)
env = CurriculumSwarmEnv(
    auto_progress=True,       # Automatically advance stages
    progress_threshold=0.7,   # 70% success rate to advance
    progress_window=20,       # Evaluate over last 20 episodes
)
```

### Custom Curriculum

```python
curriculum_stages = [
    {
        'name': 'easy_center',
        'pattern': 'center',       # Target pattern
        'difficulty': 0.2,         # Difficulty rating (0-1)
        'temperature_scale': 0.5,  # Lower = easier
        'max_steps_scale': 1.5,    # More time = easier
        'success_threshold': 0.6,  # Easier success criterion
    },
    {
        'name': 'medium_ring',
        'pattern': 'ring',
        'difficulty': 0.5,
        'temperature_scale': 0.8,
        'max_steps_scale': 1.2,
        'success_threshold': 0.7,
    },
    {
        'name': 'hard_spiral',
        'pattern': 'spiral',
        'difficulty': 1.0,
        'temperature_scale': 1.2,
        'max_steps_scale': 0.8,
        'success_threshold': 0.8,
    },
]

env = CurriculumSwarmEnv(
    curriculum_stages=curriculum_stages,
    auto_progress=True,
)
```

### Curriculum Control

```python
# Check progress
progress = env.get_curriculum_progress()
print(f"Stage: {progress['current_stage']}/{progress['total_stages']}")
print(f"Success rate: {progress['success_rate']:.2%}")

# Manual stage control
env.set_stage(2)  # Jump to stage 2
```

### How Curriculum Learning Prevents Failure

1. **Gradual Difficulty** - Agent masters simple patterns before complex ones
2. **Temperature Scaling** - Start with less thermal noise
3. **Extended Time** - Give more steps for early stages
4. **Lower Thresholds** - Easier success criteria at first
5. **Automatic Progression** - Only advance when ready

---

## Learnable Max Force

Allow the agent to modulate its maximum force based on conditions.

### How It Works

The action space is normalized to [-1, 1]:
- First N actions: Force directions/strengths (scaled by max_force)
- Last action (if learnable): Force scaling factor

```python
from python.swarm_env_v2 import SwarmEnvV2

env = SwarmEnvV2(
    num_basis=9,
    initial_max_force=1000.0,     # Starting max force
    learnable_max_force=True,      # Add force scale action
    temperature_force_coupling=True,  # Scale force with temperature
)

# Action space is now (10,) instead of (9,)
# action[0:9] = normalized force strengths (-1 to 1)
# action[9] = force scale modifier (-1 to 1) -> scales 0.1x to 2.0x
```

### Temperature-Force Coupling

When enabled, forces are automatically scaled based on temperature:
```python
effective_max_force = max_force * sqrt(temperature)
```

This allows the agent to overcome higher thermal motion with stronger forces.

### Action Smoothing

Prevent jerky control with exponential smoothing:
```python
env = SwarmEnvV2(
    action_smoothing=0.3,  # 0 = no smoothing, 1 = full smoothing
)
# Smooth: new_action = alpha * prev_action + (1 - alpha) * action
```

---

## Long Training Runs

Configurations optimized for extended training sessions.

### Preset Configurations

```python
from python.long_training import LongTrainingConfig, train_long

# Available presets
config = LongTrainingConfig.quick()   # 100K steps
config = LongTrainingConfig.medium()  # 1M steps
config = LongTrainingConfig.large()   # 5M steps
config = LongTrainingConfig.massive() # 10M steps

# Curriculum-specific long training
config = LongTrainingConfig.curriculum_long(stages=5)  # 5M steps across 5 stages
```

### Custom Configuration

```python
config = LongTrainingConfig(
    total_timesteps=5_000_000,
    algorithm='PPO',
    policy='MlpPolicy',
    
    # Learning rate schedule
    learning_rate_start=3e-4,
    learning_rate_end=1e-5,
    lr_schedule='cosine',  # 'linear', 'constant', 'cosine'
    
    # Parallelization
    n_envs=8,
    use_subproc=True,  # Use SubprocVecEnv
    
    # Checkpointing
    checkpoint_freq=250_000,
    keep_checkpoints=5,
    save_best=True,
    
    # Evaluation
    eval_freq=100_000,
    n_eval_episodes=10,
    
    # Logging
    log_dir='./runs/my_experiment',
    log_interval=10_000,
)

# Train
model = train_long(env_fn, config)
```

### Multi-Phase Training

```python
from python.long_training import train_phases

# Define training phases
config = LongTrainingConfig(
    training_phases=[
        {
            'name': 'warmup',
            'timesteps': 100_000,
            'learning_rate': 3e-4,
        },
        {
            'name': 'main_training',
            'timesteps': 1_000_000,
            'learning_rate': 1e-4,
        },
        {
            'name': 'fine_tuning',
            'timesteps': 500_000,
            'learning_rate': 3e-5,
        },
    ]
)

# Optional callback between phases
def on_phase_change(env_fn, phase, phase_idx):
    print(f"Starting phase {phase_idx}: {phase['name']}")
    return env_fn  # Can return modified env

model = train_phases(env_fn, config, phase_callback=on_phase_change)
```

---

## Anti-Failure Wrapper

Safety measures to prevent training instabilities.

```python
from python.swarm_env_v2 import AntiFailureWrapper, make_safe_env

# Basic usage
env = SwarmEnvV2(...)
safe_env = AntiFailureWrapper(
    env,
    clip_actions=True,       # Clip actions to [-1, 1]
    clip_rewards=10.0,       # Clip rewards to [-10, 10]
    normalize_obs=True,      # Running mean/std normalization
    detect_divergence=True,  # Terminate on NaN/Inf
    divergence_threshold=100.0,  # Max density threshold
)

# Convenience function
safe_env = make_safe_env(base_env)
```

### Safety Features

1. **Action Clipping** - Prevents extreme actions
2. **Reward Clipping** - Stabilizes gradients
3. **Observation Normalization** - Zero-mean, unit-variance observations
4. **Divergence Detection** - Terminates episodes on numerical instability
5. **NaN/Inf Handling** - Replaces with safe values

---

## Quick Start Examples

### Example 1: Train on Custom Image

```python
from python import SwarmImageTargetEnv, train_long, LongTrainingConfig

# Create environment
def make_env():
    return SwarmImageTargetEnv(
        target_image='my_shape.png',
        learnable_max_force=True,
        num_particles=3000,
    )

# Train
config = LongTrainingConfig.medium()
model = train_long(make_env, config)
```

### Example 2: Curriculum Learning

```python
from python import CurriculumSwarmEnv, make_safe_env, train_long

def make_env():
    env = CurriculumSwarmEnv(
        learnable_max_force=True,
        auto_progress=True,
    )
    return make_safe_env(env)

config = LongTrainingConfig.large()
model = train_long(make_env, config)
```

### Example 3: Running the Demo

```bash
# Run feature demonstrations (no training)
python examples/advanced_training_demo.py --mode demo

# Quick training test (10K steps)
python examples/advanced_training_demo.py --mode quick

# Medium training run (100K steps)
python examples/advanced_training_demo.py --mode medium --output ./my_output

# Curriculum training
python examples/advanced_training_demo.py --mode curriculum --timesteps 500000

# Train on specific image
python examples/advanced_training_demo.py --mode image --image target.png
```

---

## Best Practices

### For Stable Training

1. **Start with curriculum learning** - Even simple tasks benefit
2. **Use anti-failure wrapper** - Always wrap production environments
3. **Enable action smoothing** - Prevents jerky control
4. **Use learnable max_force** - Lets agent adapt to conditions

### For Long Training

1. **Use parallel environments** - 4-16 for PPO
2. **Enable checkpointing** - Save progress regularly
3. **Use learning rate scheduling** - Decay LR over time
4. **Monitor with TensorBoard** - Watch for instabilities

### For Pattern Matching

1. **Start with simple patterns** - Center, Gaussian
2. **Use MSE for initial training** - More informative gradients
3. **Switch to correlation for fine-tuning** - Better quality metric
4. **Lower success thresholds initially** - Build up gradually

---

## Troubleshooting

### Training Doesn't Converge

- Lower initial learning rate
- Use curriculum learning
- Enable observation normalization
- Check reward scale (should be ~1.0)

### Agent Not Progressing Through Curriculum

- Lower progress_threshold
- Increase progress_window
- Check that early stages are solvable

### Force Scale Not Learning

- Check temperature_force_coupling
- Ensure reward is sensitive to force magnitude
- May need longer training

### Memory Issues with Long Training

- Reduce buffer_size for off-policy algorithms
- Use fewer parallel environments
- Enable gradient accumulation
