"""
Training Callbacks

Unified callbacks for logging, visualization, and checkpointing.
Consolidates callbacks from rl_template.py, long_training.py, algo_sweep.py.
"""

import os
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
from io import BytesIO

try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    # Create stub for type hints
    class BaseCallback:
        pass


class VisualizationCallback(BaseCallback):
    """
    Callback for logging visualizations to TensorBoard and generating GIFs.
    
    Logs density maps and particle positions during training, and can
    automatically generate GIF animations at the end of training.
    
    Args:
        log_dir: Directory for logs
        viz_freq: Visualization frequency
        max_frames: Maximum frames to store
        save_gif: Whether to save GIF at end of training
        gif_mode: GIF visualization mode ('density', 'particles', 'combined')
        gif_fps: GIF frames per second
        domain_size: Physical domain size for particle rendering
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        log_dir: str = './runs',
        viz_freq: int = 5000,
        max_frames: int = 200,
        save_gif: bool = True,
        gif_mode: str = 'combined',
        gif_fps: int = 10,
        domain_size: float = 100.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.viz_freq = viz_freq
        self.max_frames = max_frames
        self.save_gif = save_gif
        self.gif_mode = gif_mode
        self.gif_fps = gif_fps
        self.domain_size = domain_size
        
        # Storage for GIF generation
        self.frames: List[Dict[str, Any]] = []
        
        # TensorBoard writer
        self._writer = None
    
    def _init_writer(self):
        """Initialize TensorBoard writer."""
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(self.log_dir)
            except ImportError:
                pass
    
    def _on_training_start(self):
        self._init_writer()
    
    def _on_step(self) -> bool:
        if self.n_calls % self.viz_freq == 0:
            self._record_frame()
        return True
    
    def _record_frame(self):
        """Record current state."""
        try:
            obs = self.locals.get('new_obs', [None])[0]
            if obs is None:
                return
            
            # Get particle positions if available
            positions = None
            try:
                env = self.training_env.envs[0]
                if hasattr(env, 'unwrapped'):
                    env = env.unwrapped
                if hasattr(env, 'get_particle_positions'):
                    positions = env.get_particle_positions().copy()
            except (AttributeError, IndexError):
                pass
            
            # Store frame
            if len(self.frames) < self.max_frames:
                self.frames.append({
                    'timestep': self.num_timesteps,
                    'density': obs.copy() if hasattr(obs, 'copy') else np.array(obs),
                    'positions': positions,
                })
            
            # Log to TensorBoard
            if self._writer is not None and len(obs.shape) == 2:
                self._log_density(obs)
                if positions is not None:
                    self._log_particles(positions)
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not record frame: {e}")
    
    def _log_density(self, density: np.ndarray):
        """Log density map to TensorBoard."""
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(density, cmap='hot', origin='lower')
            ax.set_title(f'Step {self.num_timesteps}')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
            buf.seek(0)
            img = np.array(Image.open(buf))
            plt.close(fig)
            
            self._writer.add_image(
                'visualization/density', 
                img, 
                self.num_timesteps, 
                dataformats='HWC'
            )
        except Exception:
            pass
    
    def _log_particles(self, positions: np.ndarray):
        """Log particle positions to TensorBoard."""
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.5, c='blue')
            ax.set_xlim(0, self.domain_size)
            ax.set_ylim(0, self.domain_size)
            ax.set_aspect('equal')
            ax.set_title(f'Step {self.num_timesteps} (N={len(positions)})')
            ax.grid(True, alpha=0.3)
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
            buf.seek(0)
            img = np.array(Image.open(buf))
            plt.close(fig)
            
            self._writer.add_image(
                'visualization/particles',
                img,
                self.num_timesteps,
                dataformats='HWC'
            )
        except Exception:
            pass
    
    def get_frames(self) -> List[Dict[str, Any]]:
        """Get recorded frames for GIF generation."""
        return self.frames.copy()
    
    def save_gif_now(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Manually save GIF from recorded frames.
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to saved GIF or None if failed
        """
        if not self.frames:
            if self.verbose > 0:
                print("No frames recorded for GIF generation")
            return None
        
        try:
            from swarm.utils.visualization import create_gif
            
            if output_path is None:
                output_path = os.path.join(self.log_dir, f'training_evolution.gif')
            
            path = create_gif(
                self.frames,
                output_path,
                fps=self.gif_fps,
                mode=self.gif_mode,
                domain_size=self.domain_size,
                verbose=self.verbose > 0,
            )
            
            if self.verbose > 0:
                print(f"GIF saved to: {path}")
            
            return path
        except Exception as e:
            if self.verbose > 0:
                print(f"Failed to create GIF: {e}")
            return None
    
    def _on_training_end(self):
        # Save GIF if enabled
        if self.save_gif and self.frames:
            if self.verbose > 0:
                print("\nGenerating training evolution GIF...")
            self.save_gif_now()
        
        if self._writer is not None:
            self._writer.close()


class ProgressCallback(BaseCallback):
    """
    Callback for detailed progress tracking.
    
    Provides ETA, performance statistics, and progress logging.
    
    Args:
        total_timesteps: Total training timesteps
        log_freq: Progress logging frequency
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        total_timesteps: int,
        log_freq: int = 10_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_freq = log_freq
        
        self.start_time = None
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.best_mean_reward = -float('inf')
    
    def _on_training_start(self):
        self.start_time = datetime.now()
        if self.verbose > 0:
            print(f"\nTraining started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total timesteps: {self.total_timesteps:,}")
    
    def _on_step(self) -> bool:
        # Track episodes
        infos = self.locals.get('infos', [{}])
        if infos and 'episode' in infos[0]:
            ep = infos[0]['episode']
            self.episode_rewards.append(ep['r'])
            self.episode_lengths.append(ep['l'])
        
        # Log progress
        if self.n_calls % self.log_freq == 0:
            self._log_progress()
        
        return True
    
    def _log_progress(self):
        """Log progress information."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress = self.num_timesteps / self.total_timesteps
        
        # ETA
        if progress > 0:
            remaining = elapsed / progress - elapsed
            eta = self._format_time(remaining)
        else:
            eta = "..."
        
        # Statistics
        if self.episode_rewards:
            window = min(100, len(self.episode_rewards))
            mean_reward = np.mean(self.episode_rewards[-window:])
            std_reward = np.std(self.episode_rewards[-window:])
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_marker = " *"
            else:
                best_marker = ""
        else:
            mean_reward = 0
            std_reward = 0
            best_marker = ""
        
        sps = self.num_timesteps / elapsed if elapsed > 0 else 0
        
        if self.verbose > 0:
            print(
                f"Step {self.num_timesteps:>10,}/{self.total_timesteps:,} "
                f"({progress*100:5.1f}%) | "
                f"Reward: {mean_reward:>7.2f}±{std_reward:.2f}{best_marker} | "
                f"SPS: {sps:>5.0f} | ETA: {eta}"
            )
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human readable."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        
        if h > 0:
            return f"{h}h{m}m"
        elif m > 0:
            return f"{m}m{s}s"
        else:
            return f"{s}s"
    
    def _on_training_end(self):
        if self.verbose > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            print(f"\nTraining complete!")
            print(f"Total time: {self._format_time(elapsed)}")
            print(f"Best mean reward: {self.best_mean_reward:.2f}")


class CheckpointCallback(BaseCallback):
    """
    Enhanced checkpoint management.
    
    Saves regular checkpoints and best model.
    
    Args:
        save_path: Directory for checkpoints
        save_freq: Checkpoint frequency
        keep_n: Number of checkpoints to keep
        save_best: Save best model separately
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        save_path: str,
        save_freq: int = 10_000,
        keep_n: int = 5,
        save_best: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.keep_n = keep_n
        self.save_best = save_best
        
        self.checkpoints: List[str] = []
        self.best_mean_reward = -float('inf')
        self.episode_rewards: List[float] = []
        
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Track rewards
        infos = self.locals.get('infos', [{}])
        if infos and 'episode' in infos[0]:
            self.episode_rewards.append(infos[0]['episode']['r'])
        
        # Save checkpoint
        if self.n_calls % self.save_freq == 0:
            self._save_checkpoint()
        
        # Check for best
        if self.save_best and len(self.episode_rewards) >= 10:
            mean_reward = np.mean(self.episode_rewards[-100:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self._save_best()
        
        return True
    
    def _save_checkpoint(self):
        """Save checkpoint."""
        name = f"checkpoint_{self.num_timesteps}"
        path = os.path.join(self.save_path, name)
        
        self.model.save(path)
        self.checkpoints.append(path)
        
        if self.verbose > 0:
            print(f"  Saved checkpoint: {name}")
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.keep_n:
            old = self.checkpoints.pop(0)
            try:
                os.remove(f"{old}.zip")
            except OSError:
                pass
    
    def _save_best(self):
        """Save best model."""
        path = os.path.join(self.save_path, "best_model")
        self.model.save(path)
        
        if self.verbose > 0:
            print(f"  New best model! (reward: {self.best_mean_reward:.2f})")


class MetricsCallback(BaseCallback):
    """
    Callback for logging custom metrics.
    
    Logs metrics from environment info dict to TensorBoard.
    
    Args:
        metric_keys: List of keys to log from info
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        metric_keys: Optional[List[str]] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.metric_keys = metric_keys or []
        
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self._current_reward = 0.0
        self._current_length = 0
    
    def _on_step(self) -> bool:
        # Track episode
        reward = self.locals.get('rewards', [0])[0]
        self._current_reward += reward
        self._current_length += 1
        
        # Log custom metrics
        infos = self.locals.get('infos', [{}])
        if infos:
            for key in self.metric_keys:
                if key in infos[0]:
                    self.logger.record(f'custom/{key}', infos[0][key])
        
        # Episode end
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            
            self.logger.record('rollout/ep_reward', self._current_reward)
            self.logger.record('rollout/ep_length', self._current_length)
            
            if len(self.episode_rewards) >= 10:
                window = min(100, len(self.episode_rewards))
                self.logger.record(
                    'rollout/ep_reward_mean', 
                    np.mean(self.episode_rewards[-window:])
                )
            
            self._current_reward = 0.0
            self._current_length = 0
        
        return True
