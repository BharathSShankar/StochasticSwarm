"""
Quick RL Training Demo
Demonstrates rapid training with stable-baselines3 PPO
"""

import numpy as np
from swarm import SwarmEnv

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("stable-baselines3 not installed. Install with: pip install stable-baselines3")


def main():
    if not SB3_AVAILABLE:
        print("Please install stable-baselines3: pip install stable-baselines3")
        return
        
    print('='*70)
    print('QUICK RL TRAINING DEMO')
    print('='*70)

    # Create environment
    print('\nCreating environment...')
    env = SwarmEnv(
        num_particles=1000,
        num_basis=9,
        grid_resolution=32,
        physics_steps_per_action=10,
        task='concentration',
    )

    # Test environment
    print('Testing environment...')
    obs, info = env.reset()
    print(f'  Observation shape: {obs.shape}')
    print(f'  Action space: {env.action_space}')
    print(f'  Observation space: {env.observation_space}')

    # Test random policy first
    print('\nTesting random policy (baseline)...')
    random_rewards = []
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        random_rewards.append(episode_reward)
    print(f'  Random policy mean reward: {np.mean(random_rewards):.3f} ± {np.std(random_rewards):.3f}')

    # Create PPO agent
    print('\nCreating PPO agent...')
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=0, 
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        n_epochs=10
    )

    # Quick training
    print('\nTraining for 10,000 timesteps...')
    model.learn(total_timesteps=10000, progress_bar=True)
    print('✓ Training complete!')

    # Evaluate trained policy
    print('\nEvaluating trained policy...')
    trained_rewards = []
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        for step in range(20):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        trained_rewards.append(episode_reward)
        print(f'  Episode {episode+1}: {episode_reward:.3f}')
    
    print(f'\n  Trained policy mean reward: {np.mean(trained_rewards):.3f} ± {np.std(trained_rewards):.3f}')

    # Comparison
    print('\n' + '='*70)
    print('RESULTS COMPARISON')
    print('='*70)
    print(f'Random policy:  {np.mean(random_rewards):7.3f} ± {np.std(random_rewards):.3f}')
    print(f'Trained policy: {np.mean(trained_rewards):7.3f} ± {np.std(trained_rewards):.3f}')
    improvement = ((np.mean(trained_rewards) - np.mean(random_rewards)) / np.mean(random_rewards)) * 100
    print(f'Improvement:    {improvement:7.1f}%')
    print('='*70)

    # Save model
    print('\nSaving model...')
    model.save('quick_demo_model')
    print('✓ Model saved to quick_demo_model.zip')

    print('\n✓ RL training demo successful!')


if __name__ == '__main__':
    main()
