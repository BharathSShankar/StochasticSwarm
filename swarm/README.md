# Swarm - Consolidated RL Package (v3.0)

This is the **clean, consolidated** version of the Stochastic Swarm RL package. It removes duplicates from the original `python/` directory and provides a unified API.

## What Changed

### Consolidation Summary

The original `python/` directory had significant code duplication:

| Original Files | Issue | Consolidated Into |
|----------------|-------|-------------------|
| `swarm_env.py` + `swarm_env_v2.py` | Duplicate base environment classes | `swarm/envs/base.py` |
| Multiple callback implementations | Same patterns in 3 files | `swarm/training/callbacks.py` |
| Hyperparameter configs | Defined in 3 different places | `swarm/training/config.py` |
| Training utilities | `rl_template.py`, `long_training.py` | `swarm/training/trainer.py` |
| Image density conversion | Only in `image_to_density.py` | `swarm/utils/density.py` |

### New Structure

```
swarm/
├── __init__.py              # Main entry point
├── envs/
│   ├── base.py              # Unified SwarmEnv (combines v1 + v2)
│   ├── tasks.py             # Task definitions (reward functions)
│   ├── curriculum.py        # Curriculum learning environment
│   └── wrappers.py          # Safety and normalization wrappers
├── training/
│   ├── config.py            # Unified TrainingConfig
│   ├── callbacks.py         # Consolidated callbacks
│   └── trainer.py           # Unified Trainer class
├── utils/
│   └── density.py           # Image-to-density utilities
└── lightning/               # NEW: PyTorch Lightning support
    ├── module.py            # PPOModule, ActorCriticModule
    ├── networks.py          # MLP, CNN, Attention architectures
    ├── data.py              # RolloutBuffer, EnvCollector
    └── trainer.py           # LightningTrainer
```

## Quick Start

### Using the Consolidated Package

```python
from swarm import SwarmEnv, Trainer, TrainingConfig

# Create environment
env = SwarmEnv(task='concentration', num_particles=2000)

# Train with stable-baselines3
trainer = Trainer(
    env_fn=lambda: SwarmEnv(task='concentration'),
    config=TrainingConfig.medium()
)
trainer.train()
trainer.evaluate()
```

### Using PyTorch Lightning (Custom Architecture)

```python
from swarm import SwarmEnv
from swarm.lightning import PPOModule, LightningTrainer, ActorCritic

# Create custom network
network = ActorCritic(
    observation_shape=(32, 32),
    action_dim=10,
    network_type='attention',  # Use transformer!
)

# Create Lightning module
module = PPOModule(
    observation_shape=(32, 32),
    action_dim=10,
    network=network,
)

# Train
trainer = LightningTrainer(
    module=module,
    env_fn=lambda: SwarmEnv(task='concentration'),
    max_iterations=100,
)
trainer.train()
```

## Key Features

### 1. Unified Environment (`SwarmEnv`)

Combines features from both `swarm_env.py` and `swarm_env_v2.py`:
- ✅ Normalized action space [-1, 1]
- ✅ Learnable max_force
- ✅ Temperature-coupled force scaling
- ✅ Action smoothing
- ✅ Pluggable reward functions (tasks)

### 2. Composable Tasks

Tasks define reward functions and can be customized:

```python
from swarm.envs.tasks import ConcentrationTask, PatternTask, CustomTask

# Built-in tasks
env = SwarmEnv(task='concentration')
env = SwarmEnv(task='dispersion')
env = SwarmEnv(task='corner')
env = SwarmEnv(task='pattern')

# Custom task
def my_reward(density, env):
    return density[16, 16] / env.num_particles

env = SwarmEnv(task=CustomTask(my_reward))
```

### 3. Unified TrainingConfig

Single configuration class for all training scenarios:

```python
from swarm import TrainingConfig

# Presets
config = TrainingConfig.quick()    # 50K steps
config = TrainingConfig.medium()   # 500K steps
config = TrainingConfig.long()     # 1M steps
config = TrainingConfig.massive()  # 10M steps

# Custom
config = TrainingConfig(
    total_timesteps=500_000,
    algorithm='PPO',
    learning_rate=3e-4,
    n_envs=4,
    lr_schedule='cosine',
)
```

### 4. PyTorch Lightning Support

Full Lightning integration for custom architectures:

```python
from swarm.lightning import (
    PPOModule,           # Lightning PPO implementation
    ActorCriticModule,   # Generic actor-critic
    MLPNetwork,          # MLP backbone
    CNNNetwork,          # CNN backbone
    AttentionNetwork,    # Transformer backbone
    LightningTrainer,    # High-level trainer
)
```

## Migration from `python/`

### Before (old python/ directory)

```python
from python.swarm_env import SwarmConcentrationEnv
from python.swarm_env_v2 import SwarmEnvV2, CurriculumSwarmEnv
from python.rl_template import RLTrainer, TensorBoardConfig
from python.long_training import LongTrainingConfig, train_long
```

### After (new swarm/ package)

```python
from swarm import SwarmEnv, Trainer, TrainingConfig
from swarm.envs import CurriculumEnv
from swarm.lightning import PPOModule, LightningTrainer
```

## Network Architectures

### Built-in Options

```python
from swarm.lightning import ActorCritic

# MLP (for flat observations)
net = ActorCritic(obs_shape, action_dim, network_type='mlp')

# CNN (for 2D density grids)
net = ActorCritic(obs_shape, action_dim, network_type='cnn')

# Attention/Transformer
net = ActorCritic(obs_shape, action_dim, network_type='attention')
```

### Custom Network

```python
import torch.nn as nn

class MyNetwork(nn.Module):
    def forward(self, x):
        # Must return: (action_mean, action_std, value)
        ...
    
    def get_action(self, x, deterministic=False):
        # Must return: (action, log_prob, value)
        ...

module = PPOModule(network=MyNetwork())
```

## Files to Remove

The old `python/` directory can be removed once you migrate:

```
python/
├── swarm_env.py       → Use swarm/envs/base.py
├── swarm_env_v2.py    → Use swarm/envs/base.py, swarm/envs/curriculum.py
├── rl_template.py     → Use swarm/training/trainer.py
├── long_training.py   → Use swarm/training/config.py, trainer.py
├── algo_sweep.py      → Use swarm/training/ (sweep functionality retained)
├── image_to_density.py → Use swarm/utils/density.py
└── __init__.py        → Use swarm/__init__.py
```

## Dependencies

Core:
- gymnasium
- numpy
- stochastic_swarm (C++ bindings)

For stable-baselines3 training:
- stable-baselines3
- tensorboard

For PyTorch Lightning:
- pytorch-lightning
- torch

Install:
```bash
pip install gymnasium numpy stable-baselines3 tensorboard pytorch-lightning torch
```
