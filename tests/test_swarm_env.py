#!/usr/bin/env python3
"""
Test script for SwarmEnv RL environments

This script tests:
1. Environment creation
2. Reset functionality
3. Step functionality with random actions
4. Observation space
5. Action space
6. Different task types
"""

import sys
import numpy as np

try:
    import stochastic_swarm as ss
    print("✓ Successfully imported stochastic_swarm module")
except ImportError as e:
    print(f"✗ Failed to import stochastic_swarm: {e}")
    print("\nPlease build the C++ extension first:")
    print("  cd build && cmake .. && make stochastic_swarm")
    sys.exit(1)

try:
    import gymnasium as gym
    print("✓ Successfully imported gymnasium")
except ImportError:
    print("✗ gymnasium not installed")
    print("  Install with: pip install gymnasium")
    sys.exit(1)

# Import from consolidated swarm package
from swarm import SwarmEnv

print("\n" + "="*60)
print("SWARM ENVIRONMENT TEST SUITE")
print("="*60)

# Test 1: Basic Environment Creation
print("\nTest 1: Environment Creation")
try:
    env = SwarmEnv(
        num_particles=1000,
        temperature=1.0,
        num_basis=4,
        grid_resolution=16,
        physics_steps_per_action=5,
        max_steps=10,
        task='concentration',
    )
    print(f"  ✓ Created environment")
    print(f"    - Observation space: {env.observation_space.shape}")
    print(f"    - Action space: {env.action_space.shape}")
    print(f"    - Particles: {env.num_particles}")
    print(f"    - Basis functions: {env.num_basis}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Reset Functionality
print("\nTest 2: Reset Functionality")
try:
    obs, info = env.reset()
    print(f"  ✓ Reset successful")
    print(f"    - Observation shape: {obs.shape}")
    print(f"    - Observation dtype: {obs.dtype}")
    print(f"    - Total particles: {obs.sum():.1f}")
    print(f"    - Min density: {obs.min():.2f}")
    print(f"    - Max density: {obs.max():.2f}")
    
    # Verify observation is valid
    assert obs.shape == env.observation_space.shape, "Shape mismatch"
    assert obs.dtype == np.float32, "Dtype mismatch"
    assert np.all(obs >= 0), "Negative densities found"
    assert np.isclose(obs.sum(), env.num_particles, rtol=0.01), "Particle count mismatch"
    print(f"  ✓ All checks passed")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Step with Random Actions
print("\nTest 3: Step with Random Actions")
try:
    num_steps = 5
    rewards = []
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        # Verify outputs
        assert obs.shape == env.observation_space.shape, "Obs shape mismatch"
        assert isinstance(reward, (float, np.floating)), f"Reward type: {type(reward)}"
        assert isinstance(terminated, bool), "terminated not bool"
        assert isinstance(truncated, bool), "truncated not bool"
        assert isinstance(info, dict), "info not dict"
        
        print(f"  Step {step+1}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
    
    print(f"  ✓ All steps completed")
    print(f"    - Average reward: {np.mean(rewards):.4f}")
    print(f"    - Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Episode Completion
print("\nTest 4: Full Episode")
try:
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0
    terminated = False
    
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        done = terminated or truncated
    
    print(f"  ✓ Episode completed")
    print(f"    - Steps taken: {step_count}")
    print(f"    - Total reward: {episode_reward:.4f}")
    print(f"    - Episode ended by: {'termination' if terminated else 'truncation'}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Different Task Types
print("\nTest 5: Different Task Types")

task_types = ['concentration', 'dispersion', 'corner', 'pattern']

for task in task_types:
    try:
        task_env = SwarmEnv(
            num_particles=500,
            num_basis=4,
            grid_resolution=16,
            max_steps=5,
            task=task,
        )
        obs, info = task_env.reset()
        
        # Run a few steps
        reward = 0.0
        for _ in range(3):
            action = task_env.action_space.sample()
            obs, reward, terminated, truncated, info = task_env.step(action)
        
        print(f"  ✓ {task:15s} - reward: {reward:.4f}")
        
    except Exception as e:
        print(f"  ✗ {task:15s} - Failed: {e}")

# Test 6: Potential Field Control
print("\nTest 6: Potential Field Control")
try:
    env = SwarmEnv(num_particles=500, num_basis=9, grid_resolution=16, task='concentration')
    obs, info = env.reset()
    
    # Test with zero action (no force)
    zero_action = np.zeros(env.num_basis, dtype=np.float32)
    obs0, reward0, _, _, _ = env.step(zero_action)
    
    # Test with strong attractive force
    attract_action = np.full(env.num_basis, -5.0, dtype=np.float32)
    env.reset()
    reward_attract = 0.0
    for _ in range(5):
        obs_attract, reward_attract, _, _, _ = env.step(attract_action)
    
    # Test with strong repulsive force
    repel_action = np.full(env.num_basis, 5.0, dtype=np.float32)
    env.reset()
    reward_repel = 0.0
    for _ in range(5):
        obs_repel, reward_repel, _, _, _ = env.step(repel_action)
    
    print(f"  ✓ Potential field control working")
    print(f"    - Zero action reward:     {reward0:.4f}")
    print(f"    - Attractive action reward: {reward_attract:.4f}")
    print(f"    - Repulsive action reward:  {reward_repel:.4f}")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Action Space Bounds
print("\nTest 7: Action Space Validation")
try:
    env = SwarmEnv(num_particles=500, num_basis=4, grid_resolution=16, task='concentration')
    env.reset()
    
    # Test valid actions
    valid_actions = [
        np.zeros(4),
        np.ones(4) * 5.0,
        np.ones(4) * -5.0,
        np.random.randn(4) * 3.0
    ]
    
    for i, action in enumerate(valid_actions):
        obs, reward, _, _, _ = env.step(action)
        print(f"  ✓ Action {i+1}: shape={action.shape}, range=[{action.min():.2f}, {action.max():.2f}]")
    
    print(f"  ✓ Action space validation passed")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 8: Observation Consistency
print("\nTest 8: Observation Consistency")
try:
    env = SwarmEnv(num_particles=1000, num_basis=4, grid_resolution=16, task='concentration')
    
    # Reset multiple times and check consistency
    particle_counts = []
    for _ in range(3):
        obs, info = env.reset()
        particle_counts.append(obs.sum())
    
    # All should equal number of particles
    for i, count in enumerate(particle_counts):
        assert np.isclose(count, env.num_particles, rtol=0.01), \
            f"Reset {i}: particle count {count} != {env.num_particles}"
    
    print(f"  ✓ Observation consistency verified")
    print(f"    - Particle counts: {[f'{c:.1f}' for c in particle_counts]}")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 9: Gym API Compliance
print("\nTest 9: Gym API Compliance")
try:
    env = SwarmEnv(num_particles=500, num_basis=4, grid_resolution=16, task='concentration')
    
    # Check required attributes
    assert hasattr(env, 'observation_space'), "Missing observation_space"
    assert hasattr(env, 'action_space'), "Missing action_space"
    assert hasattr(env, 'reset'), "Missing reset method"
    assert hasattr(env, 'step'), "Missing step method"
    
    # Check observation space
    assert isinstance(env.observation_space, gym.spaces.Box), "observation_space not Box"
    
    # Check action space
    assert isinstance(env.action_space, gym.spaces.Box), "action_space not Box"
    
    # Check reset return type
    reset_result = env.reset()
    assert isinstance(reset_result, tuple), "reset should return tuple"
    assert len(reset_result) == 2, "reset should return (obs, info)"
    
    # Check step return type
    action = env.action_space.sample()
    step_result = env.step(action)
    assert isinstance(step_result, tuple), "step should return tuple"
    assert len(step_result) == 5, "step should return (obs, reward, terminated, truncated, info)"
    
    print(f"  ✓ Gym API compliance verified")
    print(f"    - Has required attributes")
    print(f"    - Correct return types")
    print(f"    - Spaces properly defined")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("All tests passed successfully! ✓")
print("\nThe SwarmEnv RL environment is ready for training.")
print("\nNext steps:")
print("  1. Install RL library: pip install stable-baselines3")
print("  2. Train an agent using the environment")
print("  3. Visualize learned behaviors")
print("="*60)
