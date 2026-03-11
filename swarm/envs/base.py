"""
Unified Swarm Environment Base Class

Consolidates SwarmEnv and SwarmEnvV2 into a single, clean implementation
with all features:
- Learnable max_force with normalized action space [-1, 1]
- Temperature-modulated force scaling  
- Action smoothing
- Configurable reward shaping
- Task-based reward computation via composition
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple, Callable, List, Union
from abc import ABC, abstractmethod

try:
    import stochastic_swarm as ss
    SS_AVAILABLE = True
except ImportError:
    SS_AVAILABLE = False


class RewardFunction(ABC):
    """Abstract base class for reward functions (task definitions)."""
    
    @abstractmethod
    def compute(self, density: np.ndarray, env: 'SwarmEnv') -> float:
        """Compute reward from density grid."""
        pass
    
    def check_success(self, density: np.ndarray, env: 'SwarmEnv') -> bool:
        """Check if task is successfully completed (for early termination)."""
        return False


class SwarmEnv(gym.Env):
    """
    Unified Gymnasium environment for RL-controlled particle swarm.
    
    This is the canonical environment class that combines all features
    from the previous SwarmEnv and SwarmEnvV2 implementations.
    
    Features:
        - Normalized action space [-1, 1] with learnable max_force
        - Temperature-coupled force scaling
        - Action smoothing for stability
        - Pluggable reward functions via task parameter
        - Support for target pattern matching
    
    Args:
        task: Reward function or task name ('concentration', 'dispersion', etc.)
        num_particles: Number of particles in simulation
        temperature: Temperature for Brownian motion
        num_basis: Number of potential field basis functions
        grid_resolution: Size of observation grid (NxN)
        domain_size: Physical domain size
        physics_steps_per_action: Simulation steps per RL action
        max_steps: Maximum RL steps per episode
        initial_max_force: Initial maximum force magnitude
        learnable_max_force: Include max_force in action space
        temperature_force_coupling: Scale force by temperature
        action_smoothing: Exponential smoothing factor [0-1]
        reward_shaping: Reward shaping mode ('dense', 'sparse', 'simple')
        target_density: Optional target density for pattern matching
    
    Example:
        >>> env = SwarmEnv(task='concentration', num_particles=2000)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, term, trunc, info = env.step(action)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    # Add verbose parameter
    verbose: int = 0
    
    def __init__(
        self,
        task: Union[str, RewardFunction] = 'concentration',
        num_particles: int = 5000,
        temperature: float = 1.0,
        num_basis: int = 16,
        grid_resolution: int = 32,
        domain_size: float = 100.0,
        physics_steps_per_action: int = 10,
        max_steps: int = 100,
        # V2 features (normalized actions, learnable force)
        initial_max_force: float = 2000.0,
        learnable_max_force: bool = True,
        temperature_force_coupling: bool = True,
        action_smoothing: float = 0.0,
        reward_shaping: str = 'dense',
        # Target pattern for pattern matching tasks
        target_density: Optional[np.ndarray] = None,
        verbose: int = 0,
    ):
        super().__init__()
        self.verbose = verbose
        
        if not SS_AVAILABLE:
            raise ImportError(
                "stochastic_swarm C++ bindings not available. "
                "Build with: ./build.sh"
            )
        
        # Core parameters
        self.num_particles = num_particles
        self.temperature = temperature
        self.num_basis = num_basis
        self.grid_res = grid_resolution
        self.domain_size = domain_size
        self.physics_steps = physics_steps_per_action
        self.max_steps = max_steps
        
        # Force scaling
        self.initial_max_force = initial_max_force
        self.max_force = initial_max_force
        self.learnable_max_force = learnable_max_force
        self.temperature_force_coupling = temperature_force_coupling
        self.action_smoothing = action_smoothing
        self.reward_shaping = reward_shaping
        
        # Target pattern (for pattern matching tasks)
        self.target_density = target_density
        
        # Internal state
        self._prev_action = None
        self._episode_return = 0.0
        self._best_error = float('inf')
        
        # Setup reward function from task
        self._reward_fn = self._create_reward_function(task)
        
        # Create C++ simulation
        self.sim = ss.ParticleSystem(
            num_particles=num_particles,
            temperature=temperature,
            num_basis=num_basis,
            grid_res=grid_resolution
        )
        
        # Observation space: density grid
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=float('inf'),
            shape=(grid_resolution, grid_resolution),
            dtype=np.float32
        )
        
        # Action space: normalized [-1, 1]
        action_dim = num_basis + 1 if learnable_max_force else num_basis
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        self.current_step = 0
    
    def _create_reward_function(self, task) -> RewardFunction:
        """Create reward function from task specification."""
        if isinstance(task, RewardFunction):
            return task
        
        # Import task classes here to avoid circular imports
        from swarm.envs.tasks import (
            ConcentrationTask,
            DispersionTask,
            CornerTask,
            PatternTask,
            KLDivergenceTask,
            WassersteinTask,
        )
        
        task_map = {
            'concentration': ConcentrationTask,
            'dispersion': DispersionTask,
            'corner': CornerTask,
            'corners': CornerTask,
            'pattern': PatternTask,
            # Week 4: distribution matching
            'kl': KLDivergenceTask,
            'kl_divergence': KLDivergenceTask,
            'wasserstein': WassersteinTask,
            'emd': WassersteinTask,
        }
        
        task_lower = task.lower()
        if task_lower not in task_map:
            raise ValueError(
                f"Unknown task: {task}. "
                f"Available: {list(task_map.keys())}"
            )
        
        return task_map[task_lower]()
    
    def _scale_action(self, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """Convert normalized action to actual force values."""
        if self.learnable_max_force:
            force_action = action[:-1]
            force_scale_raw = action[-1]
            # Sigmoid transform: [-1, 1] -> (0.1, 2.0)
            force_scale = 0.1 + 1.9 / (1 + np.exp(-3 * force_scale_raw))
            self.max_force = self.initial_max_force * force_scale
        else:
            force_action = action
            force_scale = 1.0
        
        # Temperature coupling
        if self.temperature_force_coupling:
            effective_max_force = self.max_force * np.sqrt(self.temperature)
        else:
            effective_max_force = self.max_force
        
        # Scale to actual forces
        scaled = force_action * effective_max_force
        
        # Action smoothing
        if self.action_smoothing > 0 and self._prev_action is not None:
            alpha = self.action_smoothing
            scaled = alpha * self._prev_action + (1 - alpha) * scaled
        
        self._prev_action = scaled.copy()
        return scaled, force_scale
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset simulation
        self.sim.initialize_random(self.domain_size)
        self.sim.set_potential_params([0.0] * self.num_basis)
        
        # Reset state
        self._prev_action = None
        self.max_force = self.initial_max_force
        self._episode_return = 0.0
        self._best_error = float('inf')
        self.current_step = 0
        
        # Get observation
        self.sim.update_density_grid()
        obs = self.sim.get_density_grid().get_grid().copy().astype(np.float32)
        
        info = {
            'max_force': self.max_force,
            'step': 0,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return observation."""
        # Scale action
        scaled_forces, force_scale = self._scale_action(action)
        
        # Apply to simulation
        self.sim.set_potential_params(scaled_forces.tolist())
        
        # Run physics
        for _ in range(self.physics_steps):
            self.sim.step()
        
        # Get observation
        self.sim.update_density_grid()
        obs = self.sim.get_density_grid().get_grid().copy().astype(np.float32)
        
        # Compute reward
        reward = self._reward_fn.compute(obs, self)
        self._episode_return += reward
        
        # Check termination
        self.current_step += 1
        terminated = self._reward_fn.check_success(obs, self)
        truncated = self.current_step >= self.max_steps
        
        info = {
            'step': self.current_step,
            'max_force': self.max_force,
            'force_scale': force_scale,
            'physics_steps': self.current_step * self.physics_steps,
            'episode_return': self._episode_return,
        }
        
        return obs, float(reward), terminated, truncated, info
    
    def get_particle_positions(self) -> np.ndarray:
        """Get current particle positions as Nx2 array."""
        x = np.array(self.sim.get_x())
        y = np.array(self.sim.get_y())
        return np.column_stack((x, y))
    
    def set_target_density(self, target: np.ndarray):
        """Set target density for pattern matching tasks."""
        self.target_density = target.astype(np.float32)
        self._best_error = float('inf')
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get current environment state for visualization.
        
        Returns:
            Dictionary containing density, positions, and metadata
        """
        self.sim.update_density_grid()
        density = self.sim.get_density_grid().get_grid().copy().astype(np.float32)
        positions = self.get_particle_positions()
        
        return {
            'density': density,
            'positions': positions,
            'timestep': self.current_step,
            'max_force': self.max_force,
            'temperature': self.temperature,
            'episode_return': self._episode_return,
        }
    
    def visualize(
        self,
        mode: str = 'combined',
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Visualize current state.
        
        Args:
            mode: Visualization mode ('density', 'particles', 'combined')
            save_path: Optional path to save image
            title: Optional custom title
            
        Returns:
            RGB image array if save_path is None, else None
        """
        try:
            from swarm.utils.visualization import render_combined, save_snapshot
            
            self.sim.update_density_grid()
            density = self.sim.get_density_grid().get_grid().copy()
            positions = self.get_particle_positions()
            
            if title is None:
                title = f'Step {self.current_step} | Force: {self.max_force:.0f}'
            
            if save_path:
                save_snapshot(
                    density,
                    save_path,
                    positions=positions if mode != 'density' else None,
                    domain_size=self.domain_size,
                    title=title,
                    mode=mode,
                )
                return None
            else:
                return render_combined(
                    density,
                    positions=positions if mode != 'density' else None,
                    domain_size=self.domain_size,
                    title=title,
                )
        except ImportError:
            if self.verbose > 0:
                print("Visualization utilities not available")
            return None
    
    def create_gif(
        self,
        frames: List[Dict[str, Any]],
        output_path: str,
        fps: int = 10,
        mode: str = 'combined',
    ) -> Optional[str]:
        """
        Create a GIF from a list of recorded frames.
        
        Args:
            frames: List of frame dicts from get_state_dict()
            output_path: Path to save GIF
            fps: Frames per second
            mode: Visualization mode
            
        Returns:
            Path to saved GIF or None if failed
        """
        try:
            from swarm.utils.visualization import create_gif
            return create_gif(
                frames,
                output_path,
                fps=fps,
                mode=mode,
                domain_size=self.domain_size,
                verbose=True,
            )
        except ImportError:
            print("Visualization utilities not available")
            return None
    
    def render(self, mode: str = 'human'):
        """Render current state."""
        if mode not in ['human', 'rgb_array']:
            return None
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            self.sim.update_density_grid()
            density = self.sim.get_density_grid().get_grid()
            
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(density, cmap='hot', origin='lower')
            plt.colorbar(im, ax=ax, label='Density')
            ax.set_title(f'Step {self.current_step} | Force: {self.max_force:.0f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            if mode == 'rgb_array':
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                w, h = canvas.get_width_height()
                rgb = buf.reshape(h, w, 4)[:, :, :3]
                plt.close(fig)
                return rgb
            else:
                plt.pause(0.01)
                plt.close(fig)
        except ImportError:
            pass
        
        return None
    
    def close(self):
        """Clean up resources."""
        pass


# Convenience factory function
def make_env(
    task: str = 'concentration',
    safe: bool = True,
    normalize: bool = False,
    **kwargs
) -> gym.Env:
    """
    Create a swarm environment with optional wrappers.
    
    Args:
        task: Task name or RewardFunction
        safe: Apply SafetyWrapper
        normalize: Apply NormalizeWrapper
        **kwargs: Additional SwarmEnv arguments
    
    Returns:
        Configured environment
    """
    env = SwarmEnv(task=task, **kwargs)
    
    if safe:
        from swarm.envs.wrappers import SafetyWrapper
        env = SafetyWrapper(env)
    
    if normalize:
        from swarm.envs.wrappers import NormalizeWrapper
        env = NormalizeWrapper(env)
    
    return env
