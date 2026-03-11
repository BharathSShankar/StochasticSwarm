#!/usr/bin/env python3
"""
PyTorch Lightning Custom Architecture Example

This example shows how to use the new consolidated swarm package
with PyTorch Lightning for training custom neural network architectures.

Key features demonstrated:
1. Creating custom network architectures (CNN, MLP, Attention)
2. Using the PPOModule for training
3. The LightningTrainer for complete RL training loop
4. Evaluating and visualizing learned policies

Usage:
    python examples/lightning_custom_architecture.py

Requirements:
    pip install pytorch-lightning torch
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Import from the new consolidated swarm package
from swarm import SwarmEnv, TrainingConfig
from swarm.lightning import (
    PPOModule,
    ActorCriticModule,
    LightningTrainer,
    MLPNetwork,
    CNNNetwork,
    AttentionNetwork,
    ActorCritic,
)


def example_1_default_cnn():
    """
    Example 1: Train with default CNN architecture.
    
    Uses the built-in CNN network which is optimized for
    the 32x32 density grid observations.
    """
    print("\n" + "="*60)
    print("Example 1: Default CNN Architecture")
    print("="*60)
    
    # Create environment
    def make_env():
        return SwarmEnv(
            task='concentration',
            num_particles=1000,
            num_basis=9,
            learnable_max_force=True,
        )
    
    # Get env info
    env = make_env()
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}")
    env.close()
    
    # Create module with default CNN
    module = PPOModule(
        observation_shape=obs_shape,
        action_dim=action_dim,
        learning_rate=3e-4,
        clip_range=0.2,
        n_epochs=5,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in module.parameters()):,}")
    
    # Create trainer
    trainer = LightningTrainer(
        module=module,
        env_fn=make_env,
        rollout_steps=512,
        max_iterations=20,
        batch_size=64,
        eval_freq=10,
    )
    
    # Train
    result = trainer.train()
    
    # Evaluate
    eval_stats = trainer.evaluate(n_episodes=5)
    print(f"\nFinal evaluation: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    
    trainer.close()
    return result


def example_2_custom_mlp():
    """
    Example 2: Train with custom MLP architecture.
    
    Shows how to create your own network and use it with PPOModule.
    """
    print("\n" + "="*60)
    print("Example 2: Custom MLP Architecture")
    print("="*60)
    
    def make_env():
        return SwarmEnv(
            task='dispersion',
            num_particles=800,
            num_basis=9,
        )
    
    env = make_env()
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    env.close()
    
    # Create custom MLP-based actor-critic
    class CustomMLPActorCritic(nn.Module):
        """Custom MLP network with skip connections."""
        
        def __init__(self, obs_dim: int, action_dim: int):
            super().__init__()
            
            self.flatten = nn.Flatten()
            
            # Shared feature extractor with skip connection
            self.fc1 = nn.Linear(obs_dim, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256 + 512, 128)  # Skip connection
            
            # Actor (policy) head
            self.actor_mean = nn.Linear(128, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
            
            # Critic (value) head
            self.critic = nn.Linear(128, 1)
            
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.zeros_(m.bias)
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = self.flatten(x)
            
            # Feature extraction with skip connection
            h1 = torch.relu(self.fc1(x))
            h2 = torch.relu(self.fc2(h1))
            h3 = torch.relu(self.fc3(torch.cat([h2, h1], dim=-1)))  # Skip
            
            # Actor
            action_mean = self.actor_mean(h3)
            action_std = self.actor_log_std.exp().expand_as(action_mean)
            
            # Critic
            value = self.critic(h3).squeeze(-1)
            
            return action_mean, action_std, value
        
        def get_action(self, x, deterministic=False):
            action_mean, action_std, value = self(x)
            
            if deterministic:
                action = torch.tanh(action_mean)
                log_prob = torch.zeros(x.size(0), device=x.device)
            else:
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                action = torch.tanh(action)
            
            return action, log_prob, value
    
    # Create custom network
    obs_dim = np.prod(obs_shape)
    custom_network = CustomMLPActorCritic(obs_dim, action_dim)
    
    print(f"Custom MLP parameters: {sum(p.numel() for p in custom_network.parameters()):,}")
    
    # Create module with custom network
    module = PPOModule(
        observation_shape=obs_shape,
        action_dim=action_dim,
        network=custom_network,
        learning_rate=1e-4,
    )
    
    # Train
    trainer = LightningTrainer(
        module=module,
        env_fn=make_env,
        rollout_steps=256,
        max_iterations=15,
        batch_size=32,
    )
    
    result = trainer.train()
    trainer.close()
    
    return result


def example_3_attention_network():
    """
    Example 3: Train with Transformer/Attention architecture.
    
    Uses the built-in AttentionNetwork which treats the density
    grid as patches and uses self-attention.
    """
    print("\n" + "="*60)
    print("Example 3: Transformer/Attention Architecture")
    print("="*60)
    
    def make_env():
        return SwarmEnv(
            task='corner',
            num_particles=800,
            num_basis=9,
            grid_resolution=32,
        )
    
    env = make_env()
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    env.close()
    
    # Create attention-based actor-critic
    network = ActorCritic(
        observation_shape=obs_shape,
        action_dim=action_dim,
        network_type='attention',  # Use transformer backbone
        hidden_dims=[256],
        feature_dim=128,
    )
    
    print(f"Attention network parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    # Create module
    module = PPOModule(
        observation_shape=obs_shape,
        action_dim=action_dim,
        network=network,
        learning_rate=1e-4,
        ent_coef=0.01,  # Encourage exploration
    )
    
    # Train
    trainer = LightningTrainer(
        module=module,
        env_fn=make_env,
        rollout_steps=256,
        max_iterations=15,
    )
    
    result = trainer.train()
    trainer.close()
    
    return result


def example_4_separate_actor_critic():
    """
    Example 4: Separate actor and critic networks.
    
    Uses ActorCriticModule which allows completely separate
    networks for policy and value function.
    """
    print("\n" + "="*60)
    print("Example 4: Separate Actor-Critic Networks")
    print("="*60)
    
    def make_env():
        return SwarmEnv(
            task='concentration',
            num_particles=800,
            num_basis=9,
        )
    
    env = make_env()
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    obs_dim = np.prod(obs_shape)
    env.close()
    
    # Create separate networks
    actor = nn.Sequential(
        nn.Flatten(),
        nn.Linear(obs_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_dim),
    )
    
    critic = nn.Sequential(
        nn.Flatten(),
        nn.Linear(obs_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    
    # Create module with separate networks
    module = ActorCriticModule(
        observation_shape=obs_shape,
        action_dim=action_dim,
        actor=actor,
        critic=critic,
        actor_lr=3e-4,   # Separate learning rates
        critic_lr=1e-3,
    )
    
    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    
    # Train
    trainer = LightningTrainer(
        module=module,
        env_fn=make_env,
        rollout_steps=256,
        max_iterations=10,
    )
    
    result = trainer.train()
    trainer.close()
    
    return result


def example_5_full_custom_training():
    """
    Example 5: Fully custom training loop.
    
    Shows how to use the components directly for maximum
    flexibility (without LightningTrainer wrapper).
    """
    print("\n" + "="*60)
    print("Example 5: Direct Lightning Training (No Wrapper)")
    print("="*60)
    
    import pytorch_lightning as pl
    from swarm.lightning.data import EnvCollector
    
    def make_env():
        return SwarmEnv(
            task='concentration',
            num_particles=800,
            num_basis=9,
        )
    
    env = make_env()
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    env.close()
    
    # Create module
    module = PPOModule(
        observation_shape=obs_shape,
        action_dim=action_dim,
    )
    
    # Create collector
    collector = EnvCollector(
        env_fn=make_env,
        module=module,
        buffer_size=512,
    )
    
    # Manual training loop
    print("Running manual training loop...")
    
    for iteration in range(5):
        # Collect rollout
        dataloader = collector.collect(batch_size=64)
        
        # Train with PyTorch Lightning
        trainer = pl.Trainer(
            max_epochs=3,
            accelerator='auto',
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(module, dataloader)
        
        # Log progress
        mean_reward = collector.get_mean_reward()
        print(f"  Iteration {iteration+1}: Mean reward = {mean_reward:.2f}")
    
    collector.close()
    print("Manual training complete!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PYTORCH LIGHTNING CUSTOM ARCHITECTURE EXAMPLES")
    print("="*70)
    print("\nThis demonstrates the new swarm.lightning module for custom RL")
    print("architectures using PyTorch Lightning.\n")
    
    # Check dependencies
    try:
        import pytorch_lightning
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch Lightning: {pytorch_lightning.__version__}")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pytorch-lightning torch")
        return
    
    # Run examples
    print("\nSelect an example to run:")
    print("  1. Default CNN architecture")
    print("  2. Custom MLP with skip connections")
    print("  3. Transformer/Attention architecture")
    print("  4. Separate actor-critic networks")
    print("  5. Direct Lightning training (manual loop)")
    print("  6. Run all examples")
    
    choice = input("\nChoice (1-6) [1]: ").strip() or '1'
    
    try:
        if choice == '1':
            example_1_default_cnn()
        elif choice == '2':
            example_2_custom_mlp()
        elif choice == '3':
            example_3_attention_network()
        elif choice == '4':
            example_4_separate_actor_critic()
        elif choice == '5':
            example_5_full_custom_training()
        elif choice == '6':
            example_1_default_cnn()
            example_2_custom_mlp()
            example_3_attention_network()
            example_4_separate_actor_critic()
        else:
            print("Invalid choice!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  - View TensorBoard logs: tensorboard --logdir=./lightning_runs")
    print("  - Customize networks in swarm/lightning/networks.py")
    print("  - Implement new algorithms in swarm/lightning/module.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
