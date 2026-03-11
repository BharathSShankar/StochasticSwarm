"""
Stochastic Swarm - Clean Consolidated Package

This package provides a clean, unified API for particle swarm RL control.
It consolidates duplicate code and provides a consistent interface.

Main Components:
- envs: Gymnasium environments for swarm control
- training: Training utilities, configs, and callbacks
- utils: Density conversion and visualization utilities

Quick Start:
    from swarm import SwarmEnv, TrainingConfig, Trainer
    
    # Create environment
    env = SwarmEnv(task='concentration', num_particles=2000)
    
    # Train with stable-baselines3
    trainer = Trainer(env, algorithm='PPO')
    trainer.train(total_timesteps=100_000)
    
    # Or use PyTorch Lightning for custom architectures
    from swarm.lightning import SwarmLightningModule
    model = SwarmLightningModule(env, custom_network=MyNetwork())
"""

__version__ = "3.0.0"

# Environment classes
from swarm.envs.base import SwarmEnv
from swarm.envs.tasks import (
    ConcentrationTask,
    DispersionTask,
    CornerTask,
    PatternTask,
    # Week 4: distribution matching
    KLDivergenceTask,
    WassersteinTask,
)
from swarm.envs.curriculum import CurriculumEnv
from swarm.envs.wrappers import SafetyWrapper, NormalizeWrapper

# Training utilities
from swarm.training.config import TrainingConfig, PRESETS
from swarm.training.trainer import Trainer
from swarm.training.callbacks import (
    VisualizationCallback,
    ProgressCallback,
    CheckpointCallback,
)

# Utilities
from swarm.utils.density import (
    image_to_density,
    create_pattern,
    compute_error,
    # Week 4
    create_target,
    kl_divergence,
    symmetric_kl,
    wasserstein_distance_2d,
)

__all__ = [
    # Environments
    'SwarmEnv',
    'ConcentrationTask',
    'DispersionTask',
    'CornerTask',
    'PatternTask',
    # Week 4: distribution matching tasks
    'KLDivergenceTask',
    'WassersteinTask',
    'CurriculumEnv',
    'SafetyWrapper',
    'NormalizeWrapper',
    # Training
    'TrainingConfig',
    'PRESETS',
    'Trainer',
    'VisualizationCallback',
    'ProgressCallback',
    'CheckpointCallback',
    # Utilities
    'image_to_density',
    'create_pattern',
    'create_target',
    'compute_error',
    'kl_divergence',
    'symmetric_kl',
    'wasserstein_distance_2d',
]
