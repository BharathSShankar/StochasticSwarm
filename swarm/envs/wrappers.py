"""
Environment Wrappers

Safety and normalization wrappers for stable training.
"""

import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any


class SafetyWrapper(gym.Wrapper):
    """
    Safety wrapper to prevent training failures.
    
    Features:
        - Action clipping to valid range
        - Reward clipping for stability
        - NaN/Inf detection and recovery
        - Divergence detection
    
    Args:
        env: Environment to wrap
        clip_actions: Clip actions to [-1, 1]
        clip_rewards: Maximum absolute reward value
        detect_divergence: Check for numerical issues
        divergence_threshold: Threshold for divergence detection
    """
    
    def __init__(
        self,
        env: gym.Env,
        clip_actions: bool = True,
        clip_rewards: float = 10.0,
        detect_divergence: bool = True,
        divergence_threshold: float = 100.0,
    ):
        super().__init__(env)
        self.clip_actions = clip_actions
        self.clip_rewards = clip_rewards
        self.detect_divergence = detect_divergence
        self.divergence_threshold = divergence_threshold
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Clip actions
        if self.clip_actions:
            action = np.clip(action, -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Detect divergence
        if self.detect_divergence:
            if np.isnan(obs).any() or np.isinf(obs).any():
                print("Warning: NaN/Inf in observation, terminating")
                terminated = True
                reward = -self.clip_rewards
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            elif obs.max() > self.divergence_threshold * self.env.num_particles:
                print("Warning: Divergence detected, terminating")
                terminated = True
                reward = -self.clip_rewards
        
        # Clip reward
        if self.clip_rewards > 0:
            info['raw_reward'] = reward
            reward = np.clip(reward, -self.clip_rewards, self.clip_rewards)
        
        return obs, float(reward), terminated, truncated, info


class NormalizeWrapper(gym.Wrapper):
    """
    Observation normalization wrapper using running statistics.
    
    Maintains running mean and std for observation normalization,
    which helps stabilize training.
    
    Args:
        env: Environment to wrap
        epsilon: Small value for numerical stability
    """
    
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        
        # Running statistics
        self._obs_mean = None
        self._obs_var = None
        self._obs_count = 0
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info
    
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        self._obs_count += 1
        
        if self._obs_mean is None:
            self._obs_mean = obs.copy()
            self._obs_var = np.ones_like(obs)
        else:
            # Welford's online algorithm for running statistics
            delta = obs - self._obs_mean
            self._obs_mean += delta / self._obs_count
            delta2 = obs - self._obs_mean
            self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count
        
        # Normalize
        std = np.sqrt(self._obs_var + self.epsilon)
        return ((obs - self._obs_mean) / std).astype(np.float32)
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """Get current normalization statistics."""
        return {
            'mean': self._obs_mean.copy() if self._obs_mean is not None else None,
            'var': self._obs_var.copy() if self._obs_var is not None else None,
            'count': self._obs_count,
        }
    
    def set_statistics(self, mean: np.ndarray, var: np.ndarray, count: int = 1000):
        """Set normalization statistics (for loading)."""
        self._obs_mean = mean.copy()
        self._obs_var = var.copy()
        self._obs_count = count


class RewardScaleWrapper(gym.Wrapper):
    """
    Wrapper to scale rewards.
    
    Args:
        env: Environment to wrap
        scale: Reward scaling factor
    """
    
    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['unscaled_reward'] = reward
        return obs, reward * self.scale, terminated, truncated, info


class TimeLimit(gym.Wrapper):
    """
    Time limit wrapper with custom max steps.
    
    Args:
        env: Environment to wrap
        max_steps: Maximum steps per episode
    """
    
    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self._max_steps = max_steps
        self._step_count = 0
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self._step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        
        if self._step_count >= self._max_steps:
            truncated = True
        
        info['step_count'] = self._step_count
        return obs, reward, terminated, truncated, info


class FlattenObservation(gym.ObservationWrapper):
    """
    Flatten 2D density grid to 1D vector.
    
    Useful when using policies that expect flat inputs.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.flatten()[0],
            high=env.observation_space.high.flatten()[0],
            shape=(np.prod(old_shape),),
            dtype=env.observation_space.dtype
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.flatten()


def wrap_env(
    env: gym.Env,
    safe: bool = True,
    normalize: bool = False,
    flatten: bool = False,
    reward_scale: Optional[float] = None,
    max_steps: Optional[int] = None,
) -> gym.Env:
    """
    Apply standard wrappers to environment.
    
    Args:
        env: Base environment
        safe: Apply SafetyWrapper
        normalize: Apply NormalizeWrapper
        flatten: Flatten observations
        reward_scale: Scale rewards
        max_steps: Override max steps
    
    Returns:
        Wrapped environment
    """
    if max_steps is not None:
        env = TimeLimit(env, max_steps)
    
    if safe:
        env = SafetyWrapper(env)
    
    if normalize:
        env = NormalizeWrapper(env)
    
    if reward_scale is not None:
        env = RewardScaleWrapper(env, reward_scale)
    
    if flatten:
        env = FlattenObservation(env)
    
    return env
