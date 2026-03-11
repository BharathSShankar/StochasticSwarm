"""
Task Definitions (Reward Functions)

Each task defines a reward function for the swarm environment.
Tasks are composable - you can create custom tasks by subclassing RewardFunction.
"""

import numpy as np
from typing import Optional, TYPE_CHECKING

# Re-use the single canonical RewardFunction ABC defined in base.
# base.py does NOT import tasks at module level, so this is safe.
from swarm.envs.base import RewardFunction  # noqa: F401

if TYPE_CHECKING:
    from swarm.envs.base import SwarmEnv


class ConcentrationTask(RewardFunction):
    """
    Task: Concentrate particles at domain center.
    
    Reward: Fraction of particles in center quarter of domain.
    """
    
    def __init__(self, threshold: float = 0.8):
        """
        Args:
            threshold: Fraction of particles in center for success
        """
        self.threshold = threshold
    
    def compute(self, density: np.ndarray, env: 'SwarmEnv') -> float:
        h, w = density.shape
        center = density[h//4:3*h//4, w//4:3*w//4]
        return float(center.sum() / env.num_particles)
    
    def check_success(self, density: np.ndarray, env: 'SwarmEnv') -> bool:
        reward = self.compute(density, env)
        return reward >= self.threshold


class DispersionTask(RewardFunction):
    """
    Task: Disperse particles uniformly across domain.
    
    Reward: Negative variance (uniform = high reward).
    """
    
    def __init__(self, scale: float = 1000.0):
        """
        Args:
            scale: Scaling factor for reward
        """
        self.scale = scale
    
    def compute(self, density: np.ndarray, env: 'SwarmEnv') -> float:
        # Normalize to distribution
        density_norm = density / (density.sum() + 1e-8)
        variance = np.var(density_norm)
        return float(-variance * self.scale)
    
    def check_success(self, density: np.ndarray, env: 'SwarmEnv') -> bool:
        density_norm = density / (density.sum() + 1e-8)
        variance = np.var(density_norm)
        # Success if variance is very low (uniform)
        expected_var = 1.0 / (density.shape[0] * density.shape[1])
        return variance < expected_var * 2


class CornerTask(RewardFunction):
    """
    Task: Drive particles to four corners of domain.
    
    Reward: Sum of particles in corner regions.
    """
    
    def __init__(self, corner_fraction: float = 0.25, threshold: float = 0.7):
        """
        Args:
            corner_fraction: Fraction of grid for each corner
            threshold: Fraction of particles in corners for success
        """
        self.corner_fraction = corner_fraction
        self.threshold = threshold
    
    def compute(self, density: np.ndarray, env: 'SwarmEnv') -> float:
        h, w = density.shape
        corner_size = max(1, int(h * self.corner_fraction))
        
        corners = [
            density[:corner_size, :corner_size],      # Top-left
            density[:corner_size, -corner_size:],     # Top-right
            density[-corner_size:, :corner_size],     # Bottom-left
            density[-corner_size:, -corner_size:],    # Bottom-right
        ]
        
        total_in_corners = sum(c.sum() for c in corners)
        return float(total_in_corners / env.num_particles)
    
    def check_success(self, density: np.ndarray, env: 'SwarmEnv') -> bool:
        return self.compute(density, env) >= self.threshold


class PatternTask(RewardFunction):
    """
    Task: Match a target density pattern.
    
    Reward: Based on error between current and target density.
    """
    
    def __init__(
        self,
        target: Optional[np.ndarray] = None,
        metric: str = 'mse',
        success_threshold: float = 0.85,
    ):
        """
        Args:
            target: Target density grid (uses env.target_density if None)
            metric: Error metric ('mse', 'mae', 'correlation')
            success_threshold: Correlation threshold for success
        """
        self.target = target
        self.metric = metric
        self.success_threshold = success_threshold
        self._best_error = float('inf')
    
    def _compute_error(self, current: np.ndarray, target: np.ndarray) -> float:
        """Compute error between densities."""
        curr_norm = current / (current.sum() + 1e-8)
        targ_norm = target / (target.sum() + 1e-8)
        
        if self.metric == 'mse':
            return float(np.mean((curr_norm - targ_norm) ** 2))
        elif self.metric == 'mae':
            return float(np.mean(np.abs(curr_norm - targ_norm)))
        elif self.metric == 'correlation':
            return float(np.corrcoef(curr_norm.flatten(), targ_norm.flatten())[0, 1])
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def compute(self, density: np.ndarray, env: 'SwarmEnv') -> float:
        target = self.target if self.target is not None else env.target_density
        
        if target is None:
            # Fall back to center concentration
            h, w = density.shape
            center = density[h//4:3*h//4, w//4:3*w//4]
            return float(center.sum() / env.num_particles)
        
        error = self._compute_error(density, target)
        
        if self.metric == 'correlation':
            # Higher correlation = better
            return float(error)
        else:
            # Lower error = better (return negative error)
            # Add improvement bonus
            reward = -error * 100
            if error < self._best_error:
                improvement = self._best_error - error
                reward += improvement * 500
                self._best_error = error
            return reward
    
    def check_success(self, density: np.ndarray, env: 'SwarmEnv') -> bool:
        target = self.target if self.target is not None else env.target_density
        if target is None:
            return False
        
        correlation = self._compute_error(density, target)
        if self.metric != 'correlation':
            # Compute correlation separately
            curr_norm = density / (density.sum() + 1e-8)
            targ_norm = target / (target.sum() + 1e-8)
            correlation = float(np.corrcoef(
                curr_norm.flatten(), 
                targ_norm.flatten()
            )[0, 1])
        
        return correlation >= self.success_threshold


class CustomTask(RewardFunction):
    """
    Create a custom task from a reward function.
    
    Example:
        >>> def my_reward(density, env):
        ...     return density[16, 16] / env.num_particles
        >>> task = CustomTask(my_reward)
        >>> env = SwarmEnv(task=task)
    """
    
    def __init__(
        self,
        reward_fn,
        success_fn=None,
    ):
        """
        Args:
            reward_fn: Callable(density, env) -> float
            success_fn: Optional callable(density, env) -> bool
        """
        self._reward_fn = reward_fn
        self._success_fn = success_fn
    
    def compute(self, density: np.ndarray, env: 'SwarmEnv') -> float:
        return float(self._reward_fn(density, env))
    
    def check_success(self, density: np.ndarray, env: 'SwarmEnv') -> bool:
        if self._success_fn is None:
            return False
        return bool(self._success_fn(density, env))


class KLDivergenceTask(RewardFunction):
    """
    Task: Shape swarm density to match a target distribution using KL Divergence.

    Reward = -D_KL(P || Q) where P is the current (normalized) swarm density
    and Q is the target distribution.

        D_KL(P || Q) = Σ P(x) · log( P(x) / Q(x) )

    A reward of 0 means perfect match. More negative = further from target.

    Args:
        target: Target density grid (will be normalized internally). If None,
                defaults to a centered Gaussian.
        epsilon: Smoothing term to avoid log(0). Default 1e-8.
        scale: Multiply reward by this factor (default 1.0).
        success_threshold: D_KL value below which the episode is considered
                           solved (default 0.01).
        improvement_bonus: Extra reward scaling for improvement steps (0 = off).

    Example:
        >>> from swarm.utils.density import create_target
        >>> target = create_target('ring', grid_resolution=32)
        >>> task = KLDivergenceTask(target=target, success_threshold=0.02)
        >>> env = SwarmEnv(task=task)
    """

    def __init__(
        self,
        target: Optional[np.ndarray] = None,
        epsilon: float = 1e-8,
        scale: float = 1.0,
        success_threshold: float = 0.01,
        improvement_bonus: float = 200.0,
    ):
        self.target = target
        self.epsilon = epsilon
        self.scale = scale
        self.success_threshold = success_threshold
        self.improvement_bonus = improvement_bonus
        self._best_kl = float('inf')

    def _get_target(self, env: 'SwarmEnv') -> np.ndarray:
        """Return the target distribution, falling back to env.target_density."""
        t = self.target if self.target is not None else env.target_density
        if t is None:
            # Default: centered Gaussian
            from swarm.utils.density import create_target
            t = create_target('gaussian', grid_resolution=env.grid_res)
        return t

    def _kl_divergence(self, current: np.ndarray, target: np.ndarray) -> float:
        """Compute D_KL(P || Q) with epsilon smoothing."""
        eps = self.epsilon
        # Normalize both to proper probability distributions
        P = current / (current.sum() + eps)
        Q = target / (target.sum() + eps)
        # Smooth to avoid log(0)
        P = np.clip(P, eps, None)
        Q = np.clip(Q, eps, None)
        kl = float(np.sum(P * np.log(P / Q)))
        return max(kl, 0.0)  # Numerical safety: KL is always >= 0

    def compute(self, density: np.ndarray, env: 'SwarmEnv') -> float:
        target = self._get_target(env)
        kl = self._kl_divergence(density, target)

        # Base reward: negative KL (0 = perfect, more negative = worse)
        reward = -kl * self.scale

        # Improvement bonus — initialise silently on first call to avoid ±inf
        if self.improvement_bonus > 0:
            if self._best_kl == float('inf'):
                self._best_kl = kl   # baseline: first observation, no bonus
            elif kl < self._best_kl:
                delta = self._best_kl - kl
                reward += delta * self.improvement_bonus
                self._best_kl = kl

        return float(reward)

    def check_success(self, density: np.ndarray, env: 'SwarmEnv') -> bool:
        target = self._get_target(env)
        kl = self._kl_divergence(density, target)
        return kl <= self.success_threshold

    def reset(self):
        """Reset best KL tracker (call between episodes if reusing task object)."""
        self._best_kl = float('inf')


class WassersteinTask(RewardFunction):
    """
    Task: Shape swarm density to match a target distribution using Wasserstein
    distance (Earth Mover's Distance).

    Uses the *Sliced Wasserstein Distance*: average of 1-D Wasserstein distances
    computed along random projections. This is O(N log N) per projection and
    provides a good approximation of the true 2-D W₁ distance without requiring
    a full OT solver.

    Reward = -W(P, Q)  (less negative = closer to target)

    Args:
        target: Target density grid. If None, defaults to centered Gaussian.
        num_projections: Number of 1-D projections for SWD (default 64).
        scale: Multiply reward by this factor (default 10.0).
        success_threshold: SWD value below which the episode is solved (default 0.005).
        improvement_bonus: Extra reward scaling for improvement steps (0 = off).

    Example:
        >>> from swarm.utils.density import create_target
        >>> target = create_target('ring', grid_resolution=32)
        >>> task = WassersteinTask(target=target)
        >>> env = SwarmEnv(task=task)
    """

    def __init__(
        self,
        target: Optional[np.ndarray] = None,
        num_projections: int = 64,
        scale: float = 10.0,
        success_threshold: float = 0.005,
        improvement_bonus: float = 200.0,
    ):
        self.target = target
        self.num_projections = num_projections
        self.scale = scale
        self.success_threshold = success_threshold
        self.improvement_bonus = improvement_bonus
        self._best_w = float('inf')
        # Pre-compute random projection directions (unit vectors)
        rng = np.random.RandomState(42)
        angles = rng.uniform(0, np.pi, num_projections)
        self._proj_dirs = np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )  # (num_projections, 2)

    def _get_target(self, env: 'SwarmEnv') -> np.ndarray:
        t = self.target if self.target is not None else env.target_density
        if t is None:
            from swarm.utils.density import create_target
            t = create_target('gaussian', grid_resolution=env.grid_res)
        return t

    def _sliced_wasserstein(
        self, current: np.ndarray, target: np.ndarray
    ) -> float:
        """
        Compute Sliced Wasserstein Distance between two 2-D density grids.

        Both grids are treated as discrete probability distributions over a
        regular grid.  Each density is expanded into a weighted point cloud,
        projected onto 1-D lines, and the 1-D Wasserstein distance (via CDFs)
        is averaged over all projections.
        """
        try:
            from scipy.stats import wasserstein_distance as w1d
        except ImportError:
            # Fallback: use KL divergence proxy
            eps = 1e-8
            P = current / (current.sum() + eps)
            Q = target / (target.sum() + eps)
            P = np.clip(P, eps, None)
            Q = np.clip(Q, eps, None)
            return float(np.sum(P * np.log(P / Q)))

        h, w = current.shape
        # Build coordinate grid (normalised to [0, 1])
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        x_coords = x_coords.astype(np.float32) / (w - 1)
        y_coords = y_coords.astype(np.float32) / (h - 1)
        coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)  # (N, 2)

        eps = 1e-8
        P = current.ravel() / (current.sum() + eps)
        Q = target.ravel() / (target.sum() + eps)
        P = np.clip(P, 0, None)
        Q = np.clip(Q, 0, None)

        distances = []
        for d in self._proj_dirs:
            proj = coords @ d           # (N,)
            distances.append(w1d(proj, proj, u_weights=P, v_weights=Q))

        return float(np.mean(distances))

    def compute(self, density: np.ndarray, env: 'SwarmEnv') -> float:
        target = self._get_target(env)
        w = self._sliced_wasserstein(density, target)

        reward = -w * self.scale

        # Improvement bonus — initialise silently on first call to avoid ±inf
        if self.improvement_bonus > 0:
            if self._best_w == float('inf'):
                self._best_w = w   # baseline: first observation, no bonus
            elif w < self._best_w:
                delta = self._best_w - w
                reward += delta * self.improvement_bonus
                self._best_w = w

        return float(reward)

    def check_success(self, density: np.ndarray, env: 'SwarmEnv') -> bool:
        target = self._get_target(env)
        w = self._sliced_wasserstein(density, target)
        return w <= self.success_threshold

    def reset(self):
        """Reset best distance tracker (call between episodes if reusing)."""
        self._best_w = float('inf')


# Aliases for backwards compatibility
ConcentrationReward = ConcentrationTask
DispersionReward = DispersionTask
CornerReward = CornerTask
PatternReward = PatternTask
KLDivergenceReward = KLDivergenceTask
WassersteinReward = WassersteinTask
