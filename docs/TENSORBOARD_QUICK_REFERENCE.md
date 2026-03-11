# TensorBoard RL Template - Quick Reference

## One-Liner Training

```python
from python.rl_template import quick_train
from python.swarm_env import SwarmConcentrationEnv

trainer = quick_train(
    env_fn=lambda: SwarmConcentrationEnv(),
    algorithm='PPO',
    total_timesteps=50000
)
```

## Basic Setup

```python
from python.rl_template import RLTrainer, TensorBoardConfig
from python.swarm_env import SwarmConcentrationEnv

# 1. Define environment
def make_env():
    return SwarmConcentrationEnv(num_particles=1000, num_basis=9)

# 2. Configure logging
config = TensorBoardConfig(experiment_name='my_experiment')

# 3. Create trainer
trainer = RLTrainer(env_fn=make_env, algorithm='PPO', tb_config=config)

# 4. Train
trainer.train(total_timesteps=50000)

# 5. Evaluate
trainer.evaluate(n_episodes=10)

# 6. Clean up
trainer.close()
```

## Common Configurations

### Minimal (Fastest)
```python
config = TensorBoardConfig(
    checkpoint_freq=0,      # No checkpoints
    eval_freq=0,            # No evaluation
    log_visualizations=False  # No visualizations
)
```

### Standard (Recommended)
```python
config = TensorBoardConfig(
    checkpoint_freq=10000,
    eval_freq=5000,
    log_visualizations=True,
    visualization_freq=5000
)
```

### Detailed (Maximum tracking)
```python
config = TensorBoardConfig(
    log_interval=10,         # Log very frequently
    checkpoint_freq=5000,
    eval_freq=2000,
    log_visualizations=True,
    visualization_freq=1000,  # Frequent visualizations
    custom_metrics=['concentration', 'max_density']
)
```

## Algorithms

| Algorithm | Best For | Continuous | Discrete |
|-----------|----------|------------|----------|
| PPO | General purpose, stable | ✓ | ✓ |
| A2C | Fast training, on-policy | ✓ | ✓ |
| SAC | Sample efficiency, continuous | ✓ | ✗ |
| TD3 | Continuous control | ✓ | ✗ |
| DQN | Discrete actions, off-policy | ✗ | ✓ |

## Hyperparameter Examples

### Conservative (Stable)
```python
hyperparams = {
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'n_steps': 2048,
    'batch_size': 64
}
```

### Aggressive (Fast learning)
```python
hyperparams = {
    'learning_rate': 3e-3,
    'gamma': 0.95,
    'n_steps': 512,
    'batch_size': 128
}
```

## TensorBoard Commands

```bash
# Start TensorBoard
tensorboard --logdir=./runs

# Specify port
tensorboard --logdir=./runs --port=6007

# For remote server
tensorboard --logdir=./runs --host=0.0.0.0
```

## File Structure

```
your_project/
├── runs/                          # TensorBoard logs
│   ├── experiment_1/
│   │   ├── events.out.tfevents... # TensorBoard events
│   │   └── checkpoints/           # Model checkpoints
│   └── experiment_2/
├── examples/
│   └── tensorboard_training_example.py  # Usage examples
└── python/
    └── rl_template.py             # Main template
```

## Cheat Sheet

### Training Flow
```python
# 1. Create → 2. Configure → 3. Train → 4. Evaluate → 5. Save → 6. Close
trainer = RLTrainer(env_fn=make_env)
trainer.train(50000)
stats = trainer.evaluate()
trainer.save()
trainer.close()
```

### Compare Algorithms
```python
for algo in ['PPO', 'A2C']:
    config = TensorBoardConfig(experiment_name=f'{algo}_test')
    trainer = RLTrainer(env_fn=make_env, algorithm=algo, tb_config=config)
    trainer.train(50000)
    trainer.close()
```

### Resume Training
```python
trainer.load('./checkpoints/model.zip')
trainer.train(total_timesteps=50000, reset_num_timesteps=False)
```

## Common Patterns

### Hyperparameter Tuning
```python
learning_rates = [1e-5, 3e-4, 1e-3]

for lr in learning_rates:
    config = TensorBoardConfig(experiment_name=f'lr_{lr}')
    hyperparams = {'learning_rate': lr}
    
    trainer = RLTrainer(
        env_fn=make_env,
        hyperparams=hyperparams,
        tb_config=config
    )
    trainer.train(50000)
    trainer.close()
```

### Different Environment Sizes
```python
particle_counts = [500, 1000, 2000]

for n in particle_counts:
    env_fn = lambda n=n: SwarmConcentrationEnv(num_particles=n)
    config = TensorBoardConfig(experiment_name=f'particles_{n}')
    
    trainer = RLTrainer(env_fn=env_fn, tb_config=config)
    trainer.train(50000)
    trainer.close()
```

## TensorBoard Navigation

### Scalars Tab
- **rollout/** - Episode rewards, lengths
- **train/** - Loss, learning rate
- **eval/** - Evaluation metrics
- **custom/** - Your custom metrics
- **actions/** - Action statistics

### Images Tab
- **visualization/density_map** - Particle density heatmaps
- **visualization/particle_positions** - Particle scatter plots

## Tips

1. **Start Small**: Use `quick_train()` for initial experiments
2. **Name Clearly**: Use descriptive experiment names
3. **Compare Side-by-Side**: Run multiple experiments, view in TensorBoard
4. **Watch Visualizations**: Check IMAGES tab for environment evolution
5. **Save Best Model**: Use `save_best_model=True` in config
6. **Clean Up**: Delete old runs to save space

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No logs appearing | Wait for first logging interval, refresh browser |
| Out of memory | Reduce `batch_size`, `n_envs`, or disable visualizations |
| Slow training | Increase `physics_steps_per_action`, reduce logging frequency |
| Unstable training | Lower `learning_rate`, increase `n_steps` |

## Full Example

```python
from python.rl_template import RLTrainer, TensorBoardConfig
from python.swarm_env import SwarmConcentrationEnv

def make_env():
    return SwarmConcentrationEnv(
        num_particles=1000,
        temperature=1.0,
        num_basis=9,
        grid_resolution=32,
        physics_steps_per_action=10
    )

config = TensorBoardConfig(
    experiment_name='complete_example',
    checkpoint_freq=10000,
    eval_freq=5000,
    n_eval_episodes=5,
    log_visualizations=True,
    visualization_freq=5000,
    custom_metrics=['concentration', 'max_density']
)

hyperparams = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'gamma': 0.99
}

trainer = RLTrainer(
    env_fn=make_env,
    algorithm='PPO',
    policy='MlpPolicy',
    tb_config=config,
    hyperparams=hyperparams,
    seed=42,
    verbose=1
)

print("Training started...")
trainer.train(total_timesteps=100000, progress_bar=True)

print("Evaluating...")
stats = trainer.evaluate(n_episodes=10)

print(f"Mean reward: {stats['mean_reward']:.2f}")

trainer.save('./models/final_model')
trainer.close()

print("Done! View logs with: tensorboard --logdir=./runs")
```

## Next Steps

- Run `python examples/tensorboard_training_example.py`
- Read full documentation in `docs/RL_TENSORBOARD_TEMPLATE.md`
- Experiment with different algorithms and hyperparameters
- Monitor training in real-time with TensorBoard
