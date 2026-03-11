"""
Data Collection and Rollout Buffer

Provides data collection utilities for RL training with PyTorch Lightning.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
import gymnasium as gym


class RolloutBuffer:
    """
    Rollout buffer for collecting environment interactions.
    
    Stores transitions for on-policy algorithms (PPO, A2C).
    
    Args:
        buffer_size: Maximum number of transitions to store
        observation_shape: Shape of observations
        action_dim: Dimension of actions
        device: Torch device
        gamma: Discount factor
        gae_lambda: GAE lambda
    
    Example:
        >>> buffer = RolloutBuffer(2048, (32, 32), 17)
        >>> 
        >>> # Collect rollout
        >>> for step in range(2048):
        ...     action, log_prob, value = module.get_action(obs)
        ...     next_obs, reward, done, info = env.step(action)
        ...     buffer.add(obs, action, reward, done, log_prob, value)
        ...     obs = next_obs
        >>> 
        >>> # Compute returns and get data
        >>> buffer.compute_returns(module.get_value(obs))
        >>> dataloader = buffer.get_dataloader(batch_size=64)
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        device: str = 'cpu',
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.reset()
    
    def reset(self):
        """Reset buffer."""
        self.observations = np.zeros(
            (self.buffer_size,) + self.observation_shape, 
            dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), 
            dtype=np.float32
        )
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        
        # Computed after rollout
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """Add transition to buffer."""
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0
    
    def compute_returns(
        self, 
        last_value: float = 0.0,
        normalize_advantages: bool = True,
    ):
        """
        Compute GAE advantages and returns.
        
        Args:
            last_value: Value estimate for state after rollout
            normalize_advantages: Normalize advantages
        """
        size = self.buffer_size if self.full else self.pos
        
        # GAE computation
        last_gae = 0
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        self.returns[:size] = self.advantages[:size] + self.values[:size]
        
        # Normalize advantages
        if normalize_advantages:
            self.advantages[:size] = (
                (self.advantages[:size] - self.advantages[:size].mean()) / 
                (self.advantages[:size].std() + 1e-8)
            )
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors."""
        size = self.buffer_size if self.full else self.pos
        
        return {
            'observations': torch.tensor(self.observations[:size], device=self.device),
            'actions': torch.tensor(self.actions[:size], device=self.device),
            'old_log_probs': torch.tensor(self.log_probs[:size], device=self.device),
            'advantages': torch.tensor(self.advantages[:size], device=self.device),
            'returns': torch.tensor(self.returns[:size], device=self.device),
            'old_values': torch.tensor(self.values[:size], device=self.device),
        }
    
    def get_dataloader(
        self, 
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Get PyTorch DataLoader for training."""
        data = self.get_all()
        dataset = ExperienceDataset(data)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )


class ExperienceDataset(Dataset):
    """
    PyTorch Dataset for experience data.
    
    Wraps rollout buffer data for use with DataLoader.
    """
    
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
        self.size = len(data['observations'])
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.data.items()}


class EnvCollector:
    """
    Collect environment rollouts for training.
    
    High-level interface that handles environment interaction
    and rollout collection.
    
    Args:
        env_fn: Function to create environment
        module: Lightning module with get_action method
        buffer_size: Rollout buffer size
        n_envs: Number of parallel environments
        device: Torch device
    
    Example:
        >>> from swarm import SwarmEnv
        >>> from swarm.lightning import PPOModule, EnvCollector
        >>> 
        >>> module = PPOModule()
        >>> collector = EnvCollector(
        ...     env_fn=lambda: SwarmEnv(task='concentration'),
        ...     module=module,
        ...     buffer_size=2048,
        ... )
        >>> 
        >>> # Collect rollout and get dataloader
        >>> dataloader = collector.collect()
    """
    
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        module: Any,  # Lightning module
        buffer_size: int = 2048,
        n_envs: int = 1,
        device: str = 'cpu',
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.env_fn = env_fn
        self.module = module
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Create environment(s)
        self.env = env_fn()
        
        # Infer shapes from env
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        
        self.observation_shape = obs_space.shape
        self.action_dim = act_space.shape[0] if hasattr(act_space, 'shape') else 1
        
        # Create buffer
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            observation_shape=self.observation_shape,
            action_dim=self.action_dim,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        
        # Current observation
        self._obs = None
        self._episode_reward = 0
        self._episode_rewards: List[float] = []
    
    @torch.no_grad()
    def collect(
        self, 
        batch_size: int = 64,
        deterministic: bool = False,
    ) -> DataLoader:
        """
        Collect a full rollout and return DataLoader.
        
        Args:
            batch_size: Batch size for DataLoader
            deterministic: Use deterministic policy
        
        Returns:
            DataLoader with collected transitions
        """
        self.buffer.reset()
        
        # Initialize if needed
        if self._obs is None:
            obs, _ = self.env.reset()
            self._obs = obs
        
        self.module.eval()
        
        for _ in range(self.buffer_size):
            # Convert to tensor
            obs_tensor = torch.tensor(
                self._obs, 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(0)
            
            # Get action from policy
            action, log_prob, value = self.module.get_action(obs_tensor, deterministic)
            
            # Step environment
            action_np = action.cpu().numpy().squeeze()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Store transition
            self.buffer.add(
                observation=self._obs,
                action=action_np,
                reward=reward,
                done=done,
                log_prob=log_prob.item(),
                value=value.item(),
            )
            
            # Track episode reward
            self._episode_reward += reward
            
            if done:
                self._episode_rewards.append(self._episode_reward)
                self._episode_reward = 0
                self._obs, _ = self.env.reset()
            else:
                self._obs = next_obs
        
        # Compute returns
        obs_tensor = torch.tensor(
            self._obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        _, _, last_value = self.module.get_action(obs_tensor, True)
        
        self.buffer.compute_returns(last_value.item())
        
        self.module.train()
        
        return self.buffer.get_dataloader(batch_size=batch_size)
    
    def get_episode_rewards(self) -> List[float]:
        """Get recent episode rewards."""
        return self._episode_rewards[-100:]
    
    def get_mean_reward(self) -> float:
        """Get mean episode reward."""
        if self._episode_rewards:
            return float(np.mean(self._episode_rewards[-100:]))
        return 0.0
    
    def close(self):
        """Close environment."""
        self.env.close()


class ReplayBuffer:
    """
    Replay buffer for off-policy algorithms (SAC, TD3).
    
    Args:
        buffer_size: Maximum buffer size
        observation_shape: Shape of observations
        action_dim: Action dimension
        device: Torch device
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        device: str = 'cpu',
    ):
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.device = device
        
        self.observations = np.zeros(
            (buffer_size,) + observation_shape, dtype=np.float32
        )
        self.next_observations = np.zeros(
            (buffer_size,) + observation_shape, dtype=np.float32
        )
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ):
        """Add transition."""
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = next_observation
        self.dones[self.pos] = float(done)
        
        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch."""
        max_idx = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, max_idx, size=batch_size)
        
        return {
            'observations': torch.tensor(
                self.observations[indices], device=self.device
            ),
            'actions': torch.tensor(
                self.actions[indices], device=self.device
            ),
            'rewards': torch.tensor(
                self.rewards[indices], device=self.device
            ),
            'next_observations': torch.tensor(
                self.next_observations[indices], device=self.device
            ),
            'dones': torch.tensor(
                self.dones[indices], device=self.device
            ),
        }
    
    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos
