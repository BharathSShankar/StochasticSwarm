"""
PyTorch Lightning Module for Swarm RL

This is the core module that allows training custom neural network
architectures using PyTorch Lightning's training infrastructure.

Features:
- Full control over model architecture
- PPO algorithm implementation
- Automatic logging to TensorBoard/WandB
- Easy distributed training
- Mixed precision support
- Gradient clipping built-in
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Any, Optional, Tuple, List, Callable
import numpy as np

try:
    import pytorch_lightning as pl
    PL_AVAILABLE = True
except ImportError:
    try:
        import lightning.pytorch as pl
        PL_AVAILABLE = True
    except ImportError:
        PL_AVAILABLE = False
        # Stub for type hints
        class pl:
            class LightningModule:
                pass

from swarm.lightning.networks import ActorCritic, create_network


class SwarmRLModule(pl.LightningModule):
    """
    Base Lightning Module for Swarm RL.
    
    Override `forward`, `training_step`, and `configure_optimizers`
    to implement custom algorithms.
    
    Args:
        observation_shape: Shape of observations from environment
        action_dim: Dimension of action space
        network: Custom network (uses default if None)
        learning_rate: Learning rate
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...] = (32, 32),
        action_dim: int = 17,
        network: Optional[nn.Module] = None,
        learning_rate: float = 3e-4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Use provided network or create default
        if network is not None:
            self.network = network
        else:
            self.network = self._create_default_network()
    
    def _create_default_network(self) -> nn.Module:
        """Create default network architecture."""
        return ActorCritic(
            self.observation_shape,
            self.action_dim,
            network_type='cnn',
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass - implement in subclass."""
        return self.network(x)
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from policy."""
        if hasattr(self.network, 'get_action'):
            return self.network.get_action(obs, deterministic)
        else:
            # Default: treat network output as action mean
            out = self.network(obs)
            if isinstance(out, tuple):
                action_mean = out[0]
            else:
                action_mean = out
            
            if deterministic:
                return action_mean, torch.zeros(obs.size(0)), torch.zeros(obs.size(0))
            else:
                action = action_mean + torch.randn_like(action_mean) * 0.1
                return action, torch.zeros(obs.size(0)), torch.zeros(obs.size(0))
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)


class PPOModule(SwarmRLModule):
    """
    PPO (Proximal Policy Optimization) Lightning Module.
    
    A complete PPO implementation that works with any custom network
    that implements the ActorCritic interface.
    
    Args:
        observation_shape: Shape of observations
        action_dim: Dimension of action space
        network: Custom ActorCritic network
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        vf_coef: Value function coefficient
        ent_coef: Entropy coefficient
        max_grad_norm: Maximum gradient norm
        n_epochs: Number of PPO epochs per update
        batch_size: Mini-batch size
    
    Example:
        >>> from swarm.lightning import PPOModule, CNNNetwork
        >>> 
        >>> # Use custom network
        >>> custom_net = ActorCritic((32, 32), 17, network_type='attention')
        >>> module = PPOModule(network=custom_net)
        >>> 
        >>> # Or use defaults
        >>> module = PPOModule(observation_shape=(32, 32), action_dim=17)
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...] = (32, 32),
        action_dim: int = 17,
        network: Optional[nn.Module] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        normalize_advantage: bool = True,
        **kwargs
    ):
        super().__init__(observation_shape, action_dim, network, learning_rate)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.normalize_advantage = normalize_advantage
        
        # Metrics tracking
        self.training_step_outputs: List[Dict] = []
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returns action mean, std, and value."""
        return self.network(x)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards tensor (T,)
            values: Value estimates (T,)
            dones: Done flags (T,)
            next_value: Value of next state
        
        Returns:
            advantages: GAE advantages (T,)
            returns: Discounted returns (T,)
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        PPO training step.
        
        Expected batch format:
        {
            'observations': (B, *obs_shape),
            'actions': (B, action_dim),
            'old_log_probs': (B,),
            'advantages': (B,),
            'returns': (B,),
            'old_values': (B,),
        }
        """
        obs = batch['observations']
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        old_values = batch['old_values']
        
        # Normalize advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current policy outputs
        action_mean, action_std, values = self(obs)
        
        # Compute log probs
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        # Policy loss (clipped surrogate)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (clipped)
        values_clipped = old_values + torch.clamp(
            values - old_values, -self.clip_range, self.clip_range
        )
        value_loss1 = F.mse_loss(values, returns)
        value_loss2 = F.mse_loss(values_clipped, returns)
        value_loss = torch.max(value_loss1, value_loss2)
        
        # Total loss
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        
        # Logging
        self.log('train/policy_loss', policy_loss, prog_bar=True)
        self.log('train/value_loss', value_loss, prog_bar=True)
        self.log('train/entropy', entropy)
        self.log('train/loss', loss)
        
        # Track for epoch end
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'policy_loss': policy_loss.detach(),
            'value_loss': value_loss.detach(),
        })
        
        return loss
    
    def on_train_epoch_end(self):
        """Log epoch statistics."""
        if self.training_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
            self.log('train/epoch_loss', avg_loss)
            self.training_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)
        return optimizer
    
    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ):
        """Apply gradient clipping."""
        if gradient_clip_val is None:
            gradient_clip_val = self.max_grad_norm
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)


class ActorCriticModule(SwarmRLModule):
    """
    Generic Actor-Critic module for any custom network.
    
    This module provides a flexible interface for implementing
    various actor-critic algorithms (A2C, PPO, TRPO, etc.)
    
    Args:
        observation_shape: Shape of observations
        action_dim: Dimension of action space
        actor: Custom actor network
        critic: Custom critic network
        learning_rate: Learning rate
        actor_lr: Separate actor learning rate (optional)
        critic_lr: Separate critic learning rate (optional)
    
    Example:
        >>> # Create custom actor and critic
        >>> actor = MLPNetwork(1024, [256, 256], 17)
        >>> critic = MLPNetwork(1024, [256, 256], 1)
        >>> 
        >>> module = ActorCriticModule(
        ...     actor=actor,
        ...     critic=critic,
        ...     observation_shape=(32, 32),
        ...     action_dim=17,
        ... )
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...] = (32, 32),
        action_dim: int = 17,
        actor: Optional[nn.Module] = None,
        critic: Optional[nn.Module] = None,
        learning_rate: float = 3e-4,
        actor_lr: Optional[float] = None,
        critic_lr: Optional[float] = None,
        **kwargs
    ):
        # Don't pass network to parent
        super().__init__(observation_shape, action_dim, None, learning_rate)
        
        self.actor_lr = actor_lr or learning_rate
        self.critic_lr = critic_lr or learning_rate
        
        # Setup actor
        if actor is not None:
            self.actor = actor
        else:
            input_dim = int(np.prod(observation_shape))
            self.actor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
            )
        
        # Setup critic
        if critic is not None:
            self.critic = critic
        else:
            input_dim = int(np.prod(observation_shape))
            self.critic = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        
        # Log std for action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Override network reference
        self.network = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        action_mean = self.actor(x)
        action_std = self.log_std.exp().expand_as(action_mean)
        value = self.critic(x).squeeze(-1)
        return action_mean, action_std, value
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from policy."""
        action_mean, action_std, value = self(obs)
        
        if deterministic:
            action = torch.tanh(action_mean)
            log_prob = torch.zeros(obs.size(0), device=obs.device)
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action = torch.tanh(action)
        
        return action, log_prob, value
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Generic actor-critic training step."""
        obs = batch['observations']
        actions = batch['actions']
        returns = batch['returns']
        advantages = batch.get('advantages', returns)
        
        action_mean, action_std, values = self(obs)
        
        # Policy loss
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        entropy = dist.entropy().sum(dim=-1).mean()
        
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        self.log('train/loss', loss)
        self.log('train/policy_loss', policy_loss)
        self.log('train/value_loss', value_loss)
        
        return loss
    
    def configure_optimizers(self):
        """Configure separate optimizers for actor and critic."""
        actor_params = list(self.actor.parameters()) + [self.log_std]
        critic_params = list(self.critic.parameters())
        
        actor_opt = Adam(actor_params, lr=self.actor_lr)
        critic_opt = Adam(critic_params, lr=self.critic_lr)
        
        return [actor_opt, critic_opt]
