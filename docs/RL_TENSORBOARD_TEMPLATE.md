# RL Training Template with TensorBoard Logging

This guide explains how to use the reusable RL training template with comprehensive TensorBoard logging for tracking and visualizing your reinforcement learning experiments.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Using the Template](#using-the-template)
- [TensorBoard Logging](#tensorboard-logging)
- [Visualization Features](#visualization-features)
- [Advanced Usage](#advanced-usage)
- [Examples](#examples)

## Overview

The RL template (`python/rl_template.py`) provides a standardized way to train reinforcement learning agents with automatic TensorBoard logging. It supports multiple RL algorithms from Stable-Baselines3 and includes custom callbacks for metrics tracking and visualization.

### Key Components

- **`RLTrainer`**: Main class for training RL agents
- **`TensorBoardConfig`**: Configuration for logging and checkpoints
- **`CustomMetricsCallback`**: Logs custom environment metrics
- **`VisualizationCallback`**: Logs density maps and particle visualizations
- **`quick_train()`**: Helper function for rapid experimentation

## Features

### ✅ Comprehensive Logging
- Episode rewards and lengths
- Training loss and learning rate
- Custom environment-specific metrics
- Action statistics
- Evaluation performance

### ✅ Visual Tracking
- Density map heatmaps at training milestones
- Particle position scatter plots
- Real-time training visualization via TensorBoard

### ✅ Multiple Algorithms
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- DQN (Deep Q-Network)

### ✅ Automatic Checkpointing
- Periodic model saves
- Best model tracking
- Resume training capability

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from python.rl_template import RLTrainer, TensorBoardConfig
from python.swarm_env import SwarmConcentrationEnv

# Create environment function
def make_env():
    return SwarmConcentrationEnv(
        num_particles=1000,
        num_basis=9,
        grid_resolution=32
    )

# Configure logging
config = TensorBoardConfig(
    experiment_name='my_first_experiment',
    log_visualizations=True
)

# Create trainer
trainer = RLTrainer(
    env_fn=make_env,
    algorithm='PPO',
    tb_config=config
)

# Train
trainer.train(total_timesteps=50000)

# Save and evaluate
trainer.save()
trainer.evaluate(n_episodes=10)
trainer.close()
```

### View Results

```bash
tensorboard --logdir=./runs
```

Then open your browser to `http://localhost:6006`

## Using the Template

### 1. Configure TensorBoard Logging

```python
from python.rl_template import TensorBoardConfig

config = TensorBoardConfig(
    log_dir='./runs',                    # Base directory for logs
    experiment_name='my_experiment',      # Name of this experiment
    log_interval=100,                     # Log metrics every N steps
    checkpoint_freq=10000,                # Save checkpoint every N steps
    eval_freq=5000,                       # Evaluate every N steps
    n_eval_episodes=5,                    # Number of evaluation episodes
    save_best_model=True,                 # Save best performing model
    custom_metrics=['concentration'],     # Custom metrics from info dict
    log_visualizations=True,              # Enable visualization logging
    visualization_freq=5000               # Visualize every N steps
)
```

### 2. Create the Trainer

```python
from python.rl_template import RLTrainer

trainer = RLTrainer(
    env_fn=make_env,              # Function that creates your environment
    algorithm='PPO',               # RL algorithm to use
    policy='MlpPolicy',            # Policy architecture
    tb_config=config,              # TensorBoard configuration
    hyperparams=None,              # Custom hyperparameters (optional)
    n_envs=1,                      # Number of parallel environments
    seed=42,                       # Random seed for reproducibility
    verbose=1                      # Verbosity level (0, 1, or 2)
)
```

### 3. Train the Agent

```python
trainer.train(
    total_timesteps=100000,
    progress_bar=True
)
```

### 4. Evaluate Performance

```python
stats = trainer.evaluate(
    n_episodes=10,
    deterministic=True
)

print(f"Mean reward: {stats['mean_reward']:.2f}")
```

## TensorBoard Logging

### Metrics Logged

#### Rollout Metrics
- `rollout/ep_reward` - Episode reward
- `rollout/ep_length` - Episode length
- `rollout/ep_reward_mean` - Mean reward (100 episode window)
- `rollout/ep_reward_std` - Reward standard deviation
- `rollout/episode_count` - Total episodes completed

#### Training Metrics (algorithm-specific)
- `train/learning_rate` - Current learning rate
- `train/loss` - Policy loss
- `train/policy_gradient_loss` - Policy gradient loss
- `train/value_loss` - Value function loss
- `train/entropy_loss` - Entropy loss
- `train/clip_fraction` - Clipped fraction (PPO)
- `train/approx_kl` - Approximate KL divergence

#### Action Statistics
- `actions/mean` - Mean action value
- `actions/std` - Action standard deviation
- `actions/min` - Minimum action
- `actions/max` - Maximum action

#### Custom Metrics
Any metrics in the `info` dict returned by your environment will be logged if you specify them in `custom_metrics`:

```python
config = TensorBoardConfig(
    custom_metrics=['concentration', 'max_density', 'entropy']
)
```

## Visualization Features

The template automatically logs visual representations of your environment state:

### Density Maps
Heatmap visualizations of particle density at regular intervals during training.

- **Location**: TensorBoard IMAGES tab → `visualization/density_map`
- **Frequency**: Controlled by `visualization_freq` parameter
- **Format**: 2D heatmap with colorbar

### Particle Positions
Scatter plots showing particle positions in the environment.

- **Location**: TensorBoard IMAGES tab → `visualization/particle_positions`
- **Frequency**: Controlled by `visualization_freq` parameter
- **Format**: 2D scatter plot

### How to View

1. Start TensorBoard: `tensorboard --logdir=./runs`
2. Navigate to the **IMAGES** tab
3. Select your experiment
4. Use the slider to see evolution over time

## Advanced Usage

### Custom Hyperparameters

```python
# Override default hyperparameters
custom_hyperparams = {
    'learning_rate': 1e-4,
    'n_steps': 1024,
    'batch_size': 128,
    'gamma': 0.95,
    'gae_lambda': 0.9,
}

trainer = RLTrainer(
    env_fn=make_env,
    algorithm='PPO',
    hyperparams=custom_hyperparams
)
```

### Compare Multiple Algorithms

```python
algorithms = ['PPO', 'A2C', 'SAC']
results = {}

for algo in algorithms:
    config = TensorBoardConfig(
        experiment_name=f'{algo}_comparison'
    )
    
    trainer = RLTrainer(
        env_fn=make_env,
        algorithm=algo,
        tb_config=config
    )
    
    trainer.train(total_timesteps=50000)
    results[algo] = trainer.evaluate(n_episodes=10)
    trainer.close()

# View all in TensorBoard
# tensorboard --logdir=./runs
```

### Resume Training

```python
# Initial training
trainer = RLTrainer(env_fn=make_env, algorithm='PPO')
trainer.train(total_timesteps=50000)
trainer.save('./models/initial_model')

# Resume later
trainer.load('./models/initial_model')
trainer.train(total_timesteps=50000, reset_num_timesteps=False)
```

### Quick Training Helper

For rapid experimentation:

```python
from python.rl_template import quick_train

trainer = quick_train(
    env_fn=make_env,
    algorithm='PPO',
    total_timesteps=50000,
    experiment_name='quick_test'
)
```

This automatically trains, saves, evaluates, and returns the trainer.

## Examples

### Example 1: Basic Training

```python
from python.rl_template import RLTrainer, TensorBoardConfig
from python.swarm_env import SwarmConcentrationEnv

def make_env():
    return SwarmConcentrationEnv(num_particles=1000, num_basis=9)

config = TensorBoardConfig(experiment_name='basic_training')
trainer = RLTrainer(env_fn=make_env, algorithm='PPO', tb_config=config)
trainer.train(total_timesteps=50000)
trainer.save()
trainer.evaluate(n_episodes=10)
trainer.close()
```

### Example 2: With Visualizations

```python
config = TensorBoardConfig(
    experiment_name='viz_demo',
    log_visualizations=True,
    visualization_freq=2000  # Visualize every 2000 steps
)

trainer = RLTrainer(env_fn=make_env, tb_config=config)
trainer.train(total_timesteps=50000)
```

### Example 3: Custom Metrics

```python
# Make sure your environment returns these in the info dict
config = TensorBoardConfig(
    experiment_name='custom_metrics',
    custom_metrics=['concentration', 'max_density', 'entropy']
)

trainer = RLTrainer(env_fn=make_env, tb_config=config)
trainer.train(total_timesteps=50000)
```

### Example 4: Multiple Experiments

Run the comprehensive example:

```bash
python examples/tensorboard_training_example.py
```

This demonstrates:
- Basic training
- Custom hyperparameters
- Algorithm comparison
- Visualization logging
- Quick training helper

## TensorBoard Tips

### Comparing Experiments

1. Run multiple experiments with different names
2. Start TensorBoard: `tensorboard --logdir=./runs`
3. All experiments appear in the same dashboard
4. Use the selection panel to overlay metrics
5. Use smoothing slider to reduce noise

### Best Practices

1. **Naming**: Use descriptive experiment names
   ```python
   experiment_name='ppo_lr1e4_gamma095_particles2000'
   ```

2. **Organization**: Group related experiments in subdirectories
   ```python
   log_dir='./runs/swarm_experiments'
   ```

3. **Cleanup**: Remove old experiments periodically
   ```bash
   rm -rf ./runs/old_experiment_*
   ```

4. **Remote Access**: Run TensorBoard on a server
   ```bash
   tensorboard --logdir=./runs --host=0.0.0.0 --port=6006
   ```

### Useful TensorBoard Commands

```bash
# Basic usage
tensorboard --logdir=./runs

# Specify port
tensorboard --logdir=./runs --port=6007

# Remote access
tensorboard --logdir=./runs --host=0.0.0.0

# Multiple log directories
tensorboard --logdir_spec=exp1:./runs/exp1,exp2:./runs/exp2

# Load faster (update every N seconds)
tensorboard --logdir=./runs --reload_interval=30
```

## Troubleshooting

### TensorBoard not showing logs
- Ensure training has started and logged at least one step
- Check that `log_dir` path exists
- Refresh TensorBoard in browser

### Visualizations not appearing
- Verify `log_visualizations=True` in config
- Check that environment observation is 2D array
- Ensure enough training steps have passed (`n_calls >= visualization_freq`)

### Out of memory
- Reduce `n_envs` (number of parallel environments)
- Reduce `batch_size` in hyperparameters
- Reduce `visualization_freq` or disable visualizations

### Slow training
- Increase `physics_steps_per_action` in environment
- Reduce `checkpoint_freq` and `eval_freq`
- Reduce `n_eval_episodes`

## Reference: Default Hyperparameters

### PPO
```python
{
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
}
```

### A2C
```python
{
    'learning_rate': 7e-4,
    'n_steps': 5,
    'gamma': 0.99,
    'gae_lambda': 1.0,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
}
```

### SAC
```python
{
    'learning_rate': 3e-4,
    'buffer_size': 1000000,
    'batch_size': 256,
    'gamma': 0.99,
    'tau': 0.005,
    'ent_coef': 'auto',
}
```

## Next Steps

- Run the examples in `examples/tensorboard_training_example.py`
- Experiment with different algorithms and hyperparameters
- Add custom metrics specific to your environment
- Use TensorBoard to compare and optimize your training

For more information on Stable-Baselines3 algorithms, see the [official documentation](https://stable-baselines3.readthedocs.io/).
