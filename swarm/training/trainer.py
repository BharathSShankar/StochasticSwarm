"""
Unified Trainer

Single trainer class that works with any algorithm and configuration.
Consolidates RLTrainer and train_long functionality.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Callable, Dict, Any, Optional, List
import gymnasium as gym

from swarm.training.config import TrainingConfig

# Check for stable-baselines3
try:
    from stable_baselines3 import PPO, A2C, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class Trainer:
    """
    Unified RL trainer with TensorBoard logging and checkpointing.
    
    Consolidates RLTrainer, train_long, and AlgorithmSweep functionality
    into a single, clean interface.
    
    Args:
        env_fn: Function that creates environment
        config: Training configuration
        algorithm: Override config algorithm
        seed: Random seed
    
    Example:
        >>> from swarm import SwarmEnv, Trainer, TrainingConfig
        >>> 
        >>> trainer = Trainer(
        ...     env_fn=lambda: SwarmEnv(task='concentration'),
        ...     config=TrainingConfig.medium()
        ... )
        >>> trainer.train()
        >>> trainer.evaluate()
        >>> trainer.save('my_model')
    """
    
    ALGORITHMS = {
        'PPO': PPO if SB3_AVAILABLE else None,
        'A2C': A2C if SB3_AVAILABLE else None,
        'SAC': SAC if SB3_AVAILABLE else None,
        'TD3': TD3 if SB3_AVAILABLE else None,
    }
    
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        config: Optional[TrainingConfig] = None,
        algorithm: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 required. Install: pip install stable-baselines3"
            )
        
        self.env_fn = env_fn
        self.config = config or TrainingConfig()
        
        if algorithm:
            self.config.algorithm = algorithm
        if seed:
            self.config.seed = seed
        
        # Setup directories
        self._setup_directories()
        
        # Create environments
        self.env = self._create_env(self.config.n_envs)
        self.eval_env = self._create_env(1) if self.config.eval_freq > 0 else None
        
        # Create model
        self.model = self._create_model()
        
        # Print info
        if self.config.verbose > 0:
            self._print_init_info()
    
    def _setup_directories(self):
        """Setup output directories."""
        if self.config.experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.config.experiment_name = f"{self.config.algorithm}_{timestamp}"
        
        self.log_dir = os.path.join(
            self.config.log_dir, 
            self.config.experiment_name
        )
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def _create_env(self, n_envs: int):
        """Create vectorized environment."""
        if n_envs == 1 or not self.config.use_subproc:
            return DummyVecEnv([self.env_fn for _ in range(n_envs)])
        else:
            return SubprocVecEnv([self.env_fn for _ in range(n_envs)])
    
    def _create_model(self):
        """Create RL model."""
        AlgorithmClass = self.ALGORITHMS.get(self.config.algorithm)
        if AlgorithmClass is None:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
        
        kwargs = self.config.get_algorithm_kwargs()
        tb_log = self.log_dir if self.config.tensorboard else None
        
        return AlgorithmClass(
            self.config.policy,
            self.env,
            seed=self.config.seed,
            tensorboard_log=tb_log,
            **kwargs
        )
    
    def _create_callbacks(self) -> List:
        """Create training callbacks."""
        from swarm.training.callbacks import (
            VisualizationCallback,
            ProgressCallback,
            CheckpointCallback,
            MetricsCallback,
        )
        
        callbacks = []
        
        # Progress tracking
        callbacks.append(ProgressCallback(
            total_timesteps=self.config.total_timesteps,
            log_freq=self.config.log_interval,
            verbose=self.config.verbose,
        ))
        
        # Checkpointing
        callbacks.append(CheckpointCallback(
            save_path=self.checkpoint_dir,
            save_freq=self.config.checkpoint_freq,
            keep_n=self.config.keep_checkpoints,
            save_best=self.config.save_best,
            verbose=self.config.verbose,
        ))
        
        # Visualization
        if self.config.visualize:
            callbacks.append(VisualizationCallback(
                log_dir=self.log_dir,
                viz_freq=self.config.viz_freq,
                verbose=self.config.verbose,
            ))
        
        # Metrics
        callbacks.append(MetricsCallback(verbose=self.config.verbose))
        
        # Evaluation
        if self.eval_env is not None:
            eval_cb = EvalCallback(
                self.eval_env,
                best_model_save_path=self.checkpoint_dir if self.config.save_best else None,
                log_path=self.log_dir,
                eval_freq=max(1, self.config.eval_freq // self.config.n_envs),
                n_eval_episodes=self.config.n_eval_episodes,
                deterministic=True,
            )
            callbacks.append(eval_cb)
        
        return callbacks
    
    def _print_init_info(self):
        """Print initialization information."""
        print("\n" + "="*60)
        print("Trainer Initialized")
        print("="*60)
        print(f"Algorithm: {self.config.algorithm}")
        print(f"Policy: {self.config.policy}")
        print(f"Timesteps: {self.config.total_timesteps:,}")
        print(f"Environments: {self.config.n_envs}")
        print(f"Log directory: {self.log_dir}")
        print("="*60 + "\n")
    
    def train(self, progress_bar: bool = True) -> 'Trainer':
        """
        Train the model.
        
        Args:
            progress_bar: Show progress bar
        
        Returns:
            self (for chaining)
        """
        if self.config.verbose > 0:
            print(f"\nTraining for {self.config.total_timesteps:,} timesteps...")
            print(f"TensorBoard: tensorboard --logdir={self.config.log_dir}\n")
        
        callbacks = self._create_callbacks()
        
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=progress_bar,
        )
        
        # Save final model
        final_path = os.path.join(self.checkpoint_dir, 'final_model')
        self.model.save(final_path)
        
        if self.config.verbose > 0:
            print(f"\n✓ Training complete! Final model: {final_path}.zip")
        
        return self
    
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions
            render: Render environment
        
        Returns:
            Dictionary with evaluation statistics
        """
        episode_rewards = []
        episode_lengths = []

        # Use a single-env for evaluation to avoid VecEnv array ambiguity
        eval_env = self._create_env(1)

        for _ in range(n_episodes):
            obs = eval_env.reset()
            done = np.array([False])
            ep_reward = 0.0
            ep_length = 0

            while not bool(done[0]):
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = eval_env.step(action)
                ep_reward += float(reward[0])
                ep_length += 1

                if render:
                    eval_env.render()

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

        eval_env.close()
        
        stats = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
        }
        
        if self.config.verbose > 0:
            print(f"\nEvaluation ({n_episodes} episodes):")
            print(f"  Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"  Range:  [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        
        return stats
    
    def save(self, path: Optional[str] = None):
        """Save model to path."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'model')
        self.model.save(path)
        if self.config.verbose > 0:
            print(f"✓ Saved: {path}.zip")
    
    def load(self, path: str):
        """Load model from path."""
        AlgorithmClass = self.ALGORITHMS[self.config.algorithm]
        self.model = AlgorithmClass.load(path, env=self.env)
        if self.config.verbose > 0:
            print(f"✓ Loaded: {path}")
    
    def close(self):
        """Clean up environments."""
        self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()


def quick_train(
    env_fn: Callable[[], gym.Env],
    timesteps: int = 50_000,
    algorithm: str = 'PPO',
    **kwargs
) -> Trainer:
    """
    Quick training helper for rapid experimentation.
    
    Args:
        env_fn: Environment factory function
        timesteps: Training timesteps
        algorithm: RL algorithm
        **kwargs: Additional TrainingConfig args
    
    Returns:
        Trained Trainer instance
    """
    config = TrainingConfig(
        total_timesteps=timesteps,
        algorithm=algorithm,
        **kwargs
    )
    
    trainer = Trainer(env_fn=env_fn, config=config)
    trainer.train()
    trainer.evaluate()
    
    return trainer
