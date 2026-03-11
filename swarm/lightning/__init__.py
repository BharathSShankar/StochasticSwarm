"""
PyTorch Lightning Module for Custom Swarm RL

This module provides a PyTorch Lightning-based template for training
custom neural network architectures on the swarm environment.

Key Features:
- Use any PyTorch model architecture (CNN, Transformer, etc.)
- Full control over training loop
- Built-in logging with TensorBoard, WandB, etc.
- Easy distributed training
- Checkpoint management
- Gradient clipping, mixed precision, etc.
"""

from swarm.lightning.module import SwarmRLModule, PPOModule, ActorCriticModule
from swarm.lightning.networks import (
    MLPNetwork,
    CNNNetwork,
    AttentionNetwork,
    create_network,
)
from swarm.lightning.data import RolloutBuffer, ExperienceDataset
from swarm.lightning.trainer import LightningTrainer

__all__ = [
    'SwarmRLModule',
    'PPOModule',
    'ActorCriticModule',
    'MLPNetwork',
    'CNNNetwork',
    'AttentionNetwork',
    'create_network',
    'RolloutBuffer',
    'ExperienceDataset',
    'LightningTrainer',
]
