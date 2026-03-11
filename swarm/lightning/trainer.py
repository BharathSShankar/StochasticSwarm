"""
PyTorch Lightning Trainer for Swarm RL

High-level trainer that combines the Lightning module with
environment interaction for complete RL training.
"""

import os
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime
import numpy as np

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    PL_AVAILABLE = True
except ImportError:
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping  
        from lightning.pytorch.loggers import TensorBoardLogger
        PL_AVAILABLE = True
    except ImportError:
        PL_AVAILABLE = False

import gymnasium as gym
from swarm.lightning.data import EnvCollector, RolloutBuffer


class LightningTrainer:
    """
    High-level trainer for RL with PyTorch Lightning.
    
    Handles the training loop that alternates between:
    1. Collecting environment rollouts
    2. Updating the policy with Lightning
    
    Args:
        module: Lightning module (PPOModule, ActorCriticModule, etc.)
        env_fn: Function to create environment
        rollout_steps: Steps per rollout
        n_epochs: PPO epochs per rollout
        batch_size: Mini-batch size
        max_iterations: Maximum training iterations
        eval_freq: Evaluation frequency (iterations)
        eval_episodes: Episodes per evaluation
        log_dir: Directory for logs
        experiment_name: Name for experiment
        device: Torch device ('cpu', 'cuda', 'mps')
        accelerator: Lightning accelerator
    
    Example:
        >>> from swarm import SwarmEnv
        >>> from swarm.lightning import PPOModule, LightningTrainer
        >>> 
        >>> # Create module with custom network
        >>> module = PPOModule(
        ...     observation_shape=(32, 32),
        ...     action_dim=17,
        ... )
        >>> 
        >>> # Create trainer
        >>> trainer = LightningTrainer(
        ...     module=module,
        ...     env_fn=lambda: SwarmEnv(task='concentration'),
        ...     max_iterations=100,
        ... )
        >>> 
        >>> # Train
        >>> trainer.train()
        >>> 
        >>> # Evaluate
        >>> trainer.evaluate(render=True)
    """
    
    def __init__(
        self,
        module,  # Lightning module
        env_fn: Callable[[], gym.Env],
        rollout_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        max_iterations: int = 100,
        eval_freq: int = 10,
        eval_episodes: int = 5,
        log_dir: str = './lightning_runs',
        experiment_name: Optional[str] = None,
        device: str = 'cpu',
        accelerator: str = 'auto',
        **kwargs
    ):
        if not PL_AVAILABLE:
            raise ImportError(
                "PyTorch Lightning required. Install: pip install pytorch-lightning"
            )
        
        self.module = module
        self.env_fn = env_fn
        self.rollout_steps = rollout_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.log_dir = log_dir
        self.device = device
        self.accelerator = accelerator
        
        # Setup experiment
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"swarm_ppo_{timestamp}"
        self.experiment_name = experiment_name
        
        # Create output directory
        self.output_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create collector
        self.collector = EnvCollector(
            env_fn=env_fn,
            module=module,
            buffer_size=rollout_steps,
            device=device,
        )
        
        # Training state
        self.iteration = 0
        self.total_steps = 0
        self.best_mean_reward = -float('inf')
        self.training_history: List[Dict] = []
    
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the full training loop.
        
        Args:
            verbose: Print progress
        
        Returns:
            Training statistics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Lightning RL Training: {self.experiment_name}")
            print(f"{'='*60}")
            print(f"Iterations: {self.max_iterations}")
            print(f"Rollout steps: {self.rollout_steps}")
            print(f"Epochs per rollout: {self.n_epochs}")
            print(f"Output: {self.output_dir}")
            print(f"{'='*60}\n")
        
        # Setup Lightning trainer
        logger = TensorBoardLogger(
            save_dir=self.log_dir,
            name=self.experiment_name,
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.output_dir, 'checkpoints'),
            filename='model-{epoch:02d}',
            save_top_k=3,
            monitor='train/loss',
            mode='min',
        )
        
        # Training loop
        for iteration in range(self.max_iterations):
            self.iteration = iteration
            
            # Collect rollout
            dataloader = self.collector.collect(batch_size=self.batch_size)
            self.total_steps += self.rollout_steps
            
            # Train on rollout
            pl_trainer = pl.Trainer(
                max_epochs=self.n_epochs,
                accelerator=self.accelerator,
                devices=1,
                logger=logger,
                callbacks=[checkpoint_callback],
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            
            pl_trainer.fit(self.module, dataloader)
            
            # Log progress
            mean_reward = self.collector.get_mean_reward()
            
            self.training_history.append({
                'iteration': iteration,
                'total_steps': self.total_steps,
                'mean_reward': mean_reward,
            })
            
            if verbose:
                best_marker = ""
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    best_marker = " *"
                    self._save_best()
                
                print(
                    f"Iter {iteration+1:4d}/{self.max_iterations} | "
                    f"Steps: {self.total_steps:>8,} | "
                    f"Reward: {mean_reward:>8.2f}{best_marker}"
                )
            
            # Evaluation
            if (iteration + 1) % self.eval_freq == 0:
                eval_stats = self.evaluate(n_episodes=self.eval_episodes)
                if verbose:
                    print(f"  Eval: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        
        # Final save
        self._save_final()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"Total steps: {self.total_steps:,}")
            print(f"Best mean reward: {self.best_mean_reward:.2f}")
            print(f"{'='*60}\n")
        
        return {
            'total_steps': self.total_steps,
            'best_mean_reward': self.best_mean_reward,
            'history': self.training_history,
        }
    
    def evaluate(
        self, 
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions
            render: Render environment
        
        Returns:
            Evaluation statistics
        """
        import torch
        
        self.module.eval()
        env = self.env_fn()
        
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            
            while not done:
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                
                with torch.no_grad():
                    action, _, _ = self.module.get_action(obs_tensor, deterministic)
                
                action_np = action.cpu().numpy().squeeze()
                obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                
                ep_reward += float(reward)
                ep_length += 1
                
                if render:
                    env.render()
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
        
        env.close()
        self.module.train()
        
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
        }
    
    def _save_best(self):
        """Save best model."""
        import torch
        path = os.path.join(self.output_dir, 'best_model.pt')
        torch.save(self.module.state_dict(), path)
    
    def _save_final(self):
        """Save final model."""
        import torch
        path = os.path.join(self.output_dir, 'final_model.pt')
        torch.save(self.module.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights."""
        import torch
        self.module.load_state_dict(torch.load(path))
    
    def close(self):
        """Clean up resources."""
        self.collector.close()


def quick_lightning_train(
    env_fn: Callable[[], gym.Env],
    iterations: int = 50,
    network_type: str = 'cnn',
    device: str = 'cpu',
    **kwargs
):
    """
    Quick training helper for Lightning RL.
    
    Args:
        env_fn: Environment factory function
        iterations: Number of training iterations
        network_type: Network architecture ('mlp', 'cnn', 'attention')
        device: Torch device
        **kwargs: Additional PPOModule args
    
    Returns:
        Trained LightningTrainer
    """
    from swarm.lightning.module import PPOModule
    from swarm.lightning.networks import ActorCritic
    
    # Get env info
    env = env_fn()
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    env.close()
    
    # Create network
    network = ActorCritic(
        observation_shape=obs_shape,
        action_dim=action_dim,
        network_type=network_type,
    )
    
    # Create module
    module = PPOModule(
        observation_shape=obs_shape,
        action_dim=action_dim,
        network=network,
        **kwargs
    )
    
    # Create trainer
    trainer = LightningTrainer(
        module=module,
        env_fn=env_fn,
        max_iterations=iterations,
        device=device,
    )
    
    # Train
    trainer.train()
    
    return trainer
