"""
Training Configuration

Unified configuration for all training scenarios.
Consolidates hyperparams from rl_template.py, long_training.py, algo_sweep.py.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable


@dataclass
class TrainingConfig:
    """
    Unified training configuration.
    
    Consolidates settings from rl_template.py, long_training.py, and algo_sweep.py.
    
    Args:
        total_timesteps: Total training timesteps
        algorithm: RL algorithm ('PPO', 'A2C', 'SAC', 'TD3')
        policy: Policy architecture ('MlpPolicy', 'CnnPolicy')
        learning_rate: Learning rate or schedule
        batch_size: Batch size for updates
        n_envs: Number of parallel environments
        log_dir: Directory for logs
        experiment_name: Name for this experiment
        checkpoint_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        n_eval_episodes: Episodes per evaluation
        log_interval: Log metrics every N steps
        tensorboard: Enable TensorBoard logging
        visualize: Enable visualization logging
        viz_freq: Visualization frequency
    
    Example:
        >>> config = TrainingConfig.medium()
        >>> trainer = Trainer(env, config=config)
    """
    
    # Core training
    total_timesteps: int = 100_000
    algorithm: str = 'PPO'
    policy: str = 'MlpPolicy'
    
    # Learning rate (can be float or schedule string)
    learning_rate: float = 3e-4
    learning_rate_end: float = 1e-5
    lr_schedule: str = 'constant'  # 'constant', 'linear', 'cosine'
    
    # Batch/update settings
    batch_size: int = 64
    n_steps: int = 2048  # For PPO/A2C
    n_epochs: int = 10   # For PPO
    
    # Parallelization
    n_envs: int = 1
    use_subproc: bool = False
    
    # Discount and GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # PPO specific
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Off-policy specific (SAC, TD3)
    buffer_size: int = 100_000
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    learning_starts: int = 1000
    
    # Logging
    log_dir: str = './runs'
    experiment_name: Optional[str] = None
    log_interval: int = 1000
    tensorboard: bool = True
    
    # Checkpointing
    checkpoint_freq: int = 10_000
    keep_checkpoints: int = 5
    save_best: bool = True
    
    # Evaluation
    eval_freq: int = 5_000
    n_eval_episodes: int = 5
    
    # Visualization
    visualize: bool = True
    viz_freq: int = 5_000
    
    # Misc
    seed: Optional[int] = None
    verbose: int = 1
    
    def get_algorithm_kwargs(self) -> Dict[str, Any]:
        """Get algorithm-specific kwargs."""
        base = {
            'learning_rate': self.get_lr_schedule(),
            'verbose': self.verbose,
        }
        
        if self.algorithm == 'PPO':
            return {
                **base,
                'n_steps': self.n_steps,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'max_grad_norm': self.max_grad_norm,
            }
        
        elif self.algorithm == 'A2C':
            return {
                **base,
                'n_steps': min(self.n_steps, 32),
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'ent_coef': max(self.ent_coef, 0.01),
                'vf_coef': self.vf_coef,
            }
        
        elif self.algorithm == 'SAC':
            return {
                **base,
                'buffer_size': self.buffer_size,
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'tau': self.tau,
                'train_freq': self.train_freq,
                'gradient_steps': self.gradient_steps,
                'learning_starts': self.learning_starts,
                'ent_coef': 'auto',
            }
        
        elif self.algorithm == 'TD3':
            return {
                **base,
                'buffer_size': self.buffer_size,
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'tau': self.tau,
                'train_freq': self.train_freq,
                'gradient_steps': self.gradient_steps,
                'learning_starts': self.learning_starts,
                'policy_delay': 2,
            }
        
        return base
    
    def get_lr_schedule(self) -> Callable[[float], float]:
        """Get learning rate schedule function."""
        if self.lr_schedule == 'constant':
            return lambda _: self.learning_rate
        
        elif self.lr_schedule == 'linear':
            def linear(progress: float) -> float:
                return self.learning_rate_end + progress * (
                    self.learning_rate - self.learning_rate_end
                )
            return linear
        
        elif self.lr_schedule == 'cosine':
            def cosine(progress: float) -> float:
                return self.learning_rate_end + 0.5 * (
                    self.learning_rate - self.learning_rate_end
                ) * (1 + np.cos(np.pi * (1 - progress)))
            return cosine
        
        return lambda _: self.learning_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_timesteps': self.total_timesteps,
            'algorithm': self.algorithm,
            'policy': self.policy,
            'learning_rate': self.learning_rate,
            'learning_rate_end': self.learning_rate_end,
            'lr_schedule': self.lr_schedule,
            'batch_size': self.batch_size,
            'n_envs': self.n_envs,
            'gamma': self.gamma,
            'checkpoint_freq': self.checkpoint_freq,
            'eval_freq': self.eval_freq,
        }
    
    # Preset constructors
    @classmethod
    def quick(cls) -> 'TrainingConfig':
        """Quick config for testing (50K steps)."""
        return cls(
            total_timesteps=50_000,
            checkpoint_freq=10_000,
            eval_freq=5_000,
            n_envs=1,
            viz_freq=2_500,
        )
    
    @classmethod
    def medium(cls) -> 'TrainingConfig':
        """Medium training (500K steps)."""
        return cls(
            total_timesteps=500_000,
            checkpoint_freq=50_000,
            eval_freq=25_000,
            n_envs=4,
            lr_schedule='linear',
        )
    
    @classmethod
    def long(cls) -> 'TrainingConfig':
        """Long training (1M steps)."""
        return cls(
            total_timesteps=1_000_000,
            checkpoint_freq=100_000,
            eval_freq=50_000,
            n_envs=4,
            lr_schedule='linear',
            learning_rate_end=1e-5,
        )
    
    @classmethod
    def massive(cls) -> 'TrainingConfig':
        """Massive training (10M steps)."""
        return cls(
            total_timesteps=10_000_000,
            checkpoint_freq=500_000,
            eval_freq=250_000,
            n_envs=8,
            lr_schedule='cosine',
            learning_rate_end=1e-6,
            keep_checkpoints=10,
        )


# Preset configurations
PRESETS: Dict[str, TrainingConfig] = {
    'quick': TrainingConfig.quick(),
    'medium': TrainingConfig.medium(),
    'long': TrainingConfig.long(),
    'massive': TrainingConfig.massive(),
}


# Algorithm-specific hyperparameter variants
ALGORITHM_VARIANTS = {
    'PPO': {
        'default': dict(learning_rate=3e-4, clip_range=0.2, n_epochs=10),
        'high_lr': dict(learning_rate=1e-3, clip_range=0.2, ent_coef=0.01),
        'conservative': dict(learning_rate=1e-4, clip_range=0.1, n_epochs=5),
        'large_batch': dict(batch_size=256, n_epochs=20, clip_range=0.1),
    },
    'A2C': {
        'default': dict(learning_rate=7e-4, ent_coef=0.01),
        'high_entropy': dict(learning_rate=5e-4, ent_coef=0.05),
        'stable': dict(learning_rate=3e-4, ent_coef=0.005),
    },
    'SAC': {
        'default': dict(learning_rate=3e-4, buffer_size=100_000),
        'fast': dict(learning_rate=1e-3, buffer_size=50_000, tau=0.01),
        'stable': dict(learning_rate=1e-4, buffer_size=500_000, tau=0.002),
    },
    'TD3': {
        'default': dict(learning_rate=3e-4, buffer_size=100_000),
        'aggressive': dict(learning_rate=1e-3, buffer_size=50_000),
        'stable': dict(learning_rate=1e-4, buffer_size=200_000),
    },
}


def get_variant_config(
    algorithm: str,
    variant: str = 'default',
    base_config: Optional[TrainingConfig] = None,
) -> TrainingConfig:
    """
    Get configuration with algorithm variant.
    
    Args:
        algorithm: Algorithm name
        variant: Variant name ('default', 'fast', 'stable', etc.)
        base_config: Base config to modify
    
    Returns:
        TrainingConfig with variant settings
    """
    config = base_config or TrainingConfig()
    config.algorithm = algorithm
    
    if algorithm in ALGORITHM_VARIANTS:
        if variant in ALGORITHM_VARIANTS[algorithm]:
            for key, value in ALGORITHM_VARIANTS[algorithm][variant].items():
                setattr(config, key, value)
    
    return config
