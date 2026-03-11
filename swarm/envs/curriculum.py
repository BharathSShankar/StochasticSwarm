"""
Curriculum Learning Environment

Progressive difficulty environment that automatically advances
through stages based on performance.
"""

import numpy as np
from typing import Optional, Dict, Any, List
import gymnasium as gym

from swarm.envs.base import SwarmEnv
from swarm.envs.tasks import PatternTask
from swarm.utils.density import create_pattern


class CurriculumEnv(SwarmEnv):
    """
    Environment with curriculum learning for progressive difficulty.
    
    Features:
        - Multiple curriculum stages with increasing difficulty
        - Automatic progression based on performance
        - Configurable difficulty parameters per stage
    
    Args:
        curriculum_stages: List of stage configurations
        auto_progress: Automatically progress through curriculum
        progress_threshold: Success rate required to progress
        progress_window: Number of episodes to evaluate
        **kwargs: SwarmEnv arguments
    
    Example:
        >>> stages = [
        ...     {'name': 'easy', 'pattern': 'center', 'difficulty': 0.3},
        ...     {'name': 'hard', 'pattern': 'ring', 'difficulty': 0.8},
        ... ]
        >>> env = CurriculumEnv(curriculum_stages=stages)
    """
    
    DEFAULT_CURRICULUM = [
        {
            'name': 'center_easy',
            'pattern': 'center',
            'difficulty': 0.2,
            'temperature_scale': 0.5,
            'max_steps_scale': 1.5,
            'success_threshold': 0.7,
        },
        {
            'name': 'center_normal',
            'pattern': 'center',
            'difficulty': 0.4,
            'temperature_scale': 1.0,
            'max_steps_scale': 1.0,
            'success_threshold': 0.8,
        },
        {
            'name': 'corners',
            'pattern': 'corners',
            'difficulty': 0.5,
            'temperature_scale': 1.0,
            'max_steps_scale': 1.0,
            'success_threshold': 0.75,
        },
        {
            'name': 'ring',
            'pattern': 'ring',
            'difficulty': 0.7,
            'temperature_scale': 1.0,
            'max_steps_scale': 1.0,
            'success_threshold': 0.8,
        },
        {
            'name': 'spiral',
            'pattern': 'spiral',
            'difficulty': 0.9,
            'temperature_scale': 1.0,
            'max_steps_scale': 0.8,
            'success_threshold': 0.75,
        },
    ]
    
    def __init__(
        self,
        curriculum_stages: Optional[List[Dict[str, Any]]] = None,
        auto_progress: bool = True,
        progress_threshold: float = 0.7,
        progress_window: int = 20,
        **kwargs
    ):
        # Initialize with pattern task
        kwargs.setdefault('task', 'pattern')
        super().__init__(**kwargs)
        
        # Curriculum config
        self.curriculum_stages = curriculum_stages or self.DEFAULT_CURRICULUM
        self.auto_progress = auto_progress
        self.progress_threshold = progress_threshold
        self.progress_window = progress_window
        
        # Curriculum state
        self.current_stage = 0
        self._episode_successes: List[bool] = []
        self._total_episodes = 0
        self._base_temperature = self.temperature
        self._base_max_steps = self.max_steps
        
        # Stage-specific state
        self._stage_max_steps = self.max_steps
        self._stage_name = ''
        self._current_difficulty = 0.0
        
        # Apply initial stage
        self._apply_stage(0)
    
    def _apply_stage(self, stage_idx: int):
        """Apply curriculum stage configuration."""
        stage_idx = min(stage_idx, len(self.curriculum_stages) - 1)
        stage = self.curriculum_stages[stage_idx]
        self.current_stage = stage_idx
        
        # Set target pattern
        pattern = stage.get('pattern', 'center')
        self.target_density = create_pattern(
            pattern, 
            self.grid_res, 
            self.num_particles
        )
        
        # Update reward function with new target
        if hasattr(self._reward_fn, 'target'):
            self._reward_fn.target = self.target_density
            self._reward_fn._best_error = float('inf')
        
        # Adjust temperature
        temp_scale = stage.get('temperature_scale', 1.0)
        self.temperature = self._base_temperature * temp_scale
        
        # Adjust max steps
        steps_scale = stage.get('max_steps_scale', 1.0)
        self._stage_max_steps = int(self._base_max_steps * steps_scale)
        
        # Update threshold
        if isinstance(self._reward_fn, PatternTask):
            self._reward_fn.success_threshold = stage.get('success_threshold', 0.8)
        
        # Store stage info
        self._current_difficulty = stage.get('difficulty', 0.5)
        self._stage_name = stage.get('name', f'stage_{stage_idx}')
        self._best_error = float('inf')
        
        print(f"Curriculum: Stage {stage_idx} - {self._stage_name} "
              f"(difficulty: {self._current_difficulty:.1f})")
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ):
        """Reset with curriculum tracking."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Add curriculum info
        info.update({
            'curriculum_stage': self.current_stage,
            'stage_name': self._stage_name,
            'difficulty': self._current_difficulty,
            'total_episodes': self._total_episodes,
        })
        
        return obs, info
    
    def step(self, action: np.ndarray):
        """Step with curriculum max_steps and progression."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Use stage-specific max steps
        truncated = self.current_step >= self._stage_max_steps
        
        # Track episode completion
        if terminated or truncated:
            self._episode_successes.append(terminated)
            self._total_episodes += 1
            
            # Check for progression
            if self.auto_progress and self._should_progress():
                self._progress_to_next_stage()
        
        # Add curriculum info
        info.update({
            'curriculum_stage': self.current_stage,
            'stage_name': self._stage_name,
            'difficulty': self._current_difficulty,
            'success_rate': self._get_recent_success_rate(),
        })
        
        return obs, reward, terminated, truncated, info
    
    def _should_progress(self) -> bool:
        """Check if agent should progress."""
        if len(self._episode_successes) < self.progress_window:
            return False
        
        recent = self._episode_successes[-self.progress_window:]
        success_rate = sum(recent) / len(recent)
        return success_rate >= self.progress_threshold
    
    def _progress_to_next_stage(self):
        """Progress to next curriculum stage."""
        if self.current_stage < len(self.curriculum_stages) - 1:
            self._episode_successes = []
            self._apply_stage(self.current_stage + 1)
    
    def _get_recent_success_rate(self) -> float:
        """Get success rate over recent episodes."""
        if not self._episode_successes:
            return 0.0
        window = min(self.progress_window, len(self._episode_successes))
        return sum(self._episode_successes[-window:]) / window
    
    def set_stage(self, stage_idx: int):
        """Manually set curriculum stage."""
        self._apply_stage(stage_idx)
        self._episode_successes = []
    
    def get_progress(self) -> Dict[str, Any]:
        """Get curriculum progress information."""
        return {
            'current_stage': self.current_stage,
            'total_stages': len(self.curriculum_stages),
            'stage_name': self._stage_name,
            'difficulty': self._current_difficulty,
            'success_rate': self._get_recent_success_rate(),
            'total_episodes': self._total_episodes,
            'progress_threshold': self.progress_threshold,
        }
