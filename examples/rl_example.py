"""
Example: Train RL agent to control particle swarm

This example demonstrates how to train an RL agent to concentrate particles
using the SwarmEnv environment. Two training approaches are shown:

1. Manual training loop (simple, no dependencies)
2. Stable-baselines3 training (advanced, requires stable-baselines3)

Note: For stable-baselines3, install with:
    pip install stable-baselines3
"""

import numpy as np
from swarm import SwarmEnv


def manual_random_search(num_episodes=100, steps_per_episode=50):
    """
    Simple random search: Try random actions and track best performance
    
    This is a baseline approach that doesn't use neural networks.
    Useful for understanding the environment before using complex RL algorithms.
    """
    print("\n" + "="*70)
    print("MANUAL RANDOM SEARCH TRAINING")
    print("="*70)
    
    env = SwarmEnv(
        num_particles=2000,
        temperature=1.0,
        num_basis=9,
        grid_resolution=32,
        physics_steps_per_action=10,
        task='concentration',
    )
    
    best_reward = -float('inf')
    best_action = None
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        
        # Try same random action for entire episode
        action = env.action_space.sample()
        
        for step in range(steps_per_episode):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        
        # Track best action
        if total_reward > best_reward:
            best_reward = total_reward
            best_action = action.copy()
        
        if (episode + 1) % 10 == 0:
            recent_mean = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1:3d}: "
                  f"Total Reward = {total_reward:7.2f}, "
                  f"Recent Mean = {recent_mean:7.2f}, "
                  f"Best = {best_reward:7.2f}")
    
    print(f"\nBest performing action: {best_action}")
    print(f"Best total reward: {best_reward:.2f}")
    
    # Demonstrate best action
    print("\nDemonstrating best action...")
    obs, info = env.reset()
    for step in range(steps_per_episode):
        obs, reward, terminated, truncated, info = env.step(best_action)
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}: Reward = {reward:.4f}")
        if terminated or truncated:
            break
    
    return episode_rewards, best_action


def train_with_stable_baselines3(total_timesteps=50000):
    """
    Train using stable-baselines3 PPO algorithm
    
    Requires: pip install stable-baselines3
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("\n⚠ stable-baselines3 not installed!")
        print("Install with: pip install stable-baselines3")
        return None
    
    print("\n" + "="*70)
    print("STABLE-BASELINES3 PPO TRAINING")
    print("="*70)
    
    # Create environment
    def make_env():
        return SwarmEnv(
            num_particles=2000,
            temperature=1.0,
            num_basis=9,
            grid_resolution=32,
            physics_steps_per_action=10,
            task='concentration',
        )
    
    env = DummyVecEnv([make_env])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    
    # Create callback for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=1000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Create PPO agent
    print("\nCreating PPO agent...")
    model = PPO(
        'MlpPolicy',  # Multi-layer perceptron for flattened grid observation
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Train
    print(f"\nTraining for {total_timesteps} timesteps...")
    print("(This may take several minutes...)\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )
    
    # Save final model
    model.save("swarm_ppo_final")
    print("\n✓ Model saved to 'swarm_ppo_final.zip'")
    
    # Evaluate trained policy
    print("\nEvaluating trained policy...")
    obs = env.reset()
    episode_rewards = []
    current_episode_reward = 0
    
    for step in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        current_episode_reward += reward[0]
        
        if done[0]:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            obs = env.reset()
            if len(episode_rewards) >= 10:
                break
    
    print(f"\nEvaluation results (10 episodes):")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"  Std reward:  {np.std(episode_rewards):.2f}")
    print(f"  Min reward:  {np.min(episode_rewards):.2f}")
    print(f"  Max reward:  {np.max(episode_rewards):.2f}")
    
    return model


def compare_policies():
    """
    Compare random policy vs trained policy
    """
    print("\n" + "="*70)
    print("POLICY COMPARISON")
    print("="*70)
    
    env = SwarmEnv(num_particles=2000, num_basis=9, task='concentration')
    
    # Test random policy
    print("\n1. Testing random policy...")
    obs, info = env.reset()
    random_rewards = []
    for _ in range(10):
        obs, info = env.reset()
        episode_reward = 0
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        random_rewards.append(episode_reward)
    
    print(f"   Random policy mean reward: {np.mean(random_rewards):.2f}")
    
    # Test zero action (no control)
    print("\n2. Testing zero action (no control)...")
    zero_rewards = []
    for _ in range(10):
        obs, info = env.reset()
        episode_reward = 0
        action = np.zeros(9)
        for step in range(50):
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        zero_rewards.append(episode_reward)
    
    print(f"   Zero action mean reward: {np.mean(zero_rewards):.2f}")
    
    # Test attractive action (all negative = attractive wells)
    print("\n3. Testing attractive action (hand-crafted)...")
    attractive_rewards = []
    for _ in range(10):
        obs, info = env.reset()
        episode_reward = 0
        action = np.ones(9) * -3.0  # All attractive wells
        for step in range(50):
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        attractive_rewards.append(episode_reward)
    
    print(f"   Attractive action mean reward: {np.mean(attractive_rewards):.2f}")
    
    # Summary
    print("\n" + "-"*70)
    print("SUMMARY")
    print("-"*70)
    print(f"Random policy:     {np.mean(random_rewards):7.2f} ± {np.std(random_rewards):.2f}")
    print(f"Zero action:       {np.mean(zero_rewards):7.2f} ± {np.std(zero_rewards):.2f}")
    print(f"Attractive action: {np.mean(attractive_rewards):7.2f} ± {np.std(attractive_rewards):.2f}")
    print("-"*70)


def visualize_learned_policy(model_path='swarm_ppo_final.zip'):
    """
    Visualize a learned policy in action
    
    Requires: stable-baselines3 and matplotlib
    """
    try:
        from stable_baselines3 import PPO
        import matplotlib.pyplot as plt
    except ImportError:
        print("Requires stable-baselines3 and matplotlib")
        return
    
    print("\n" + "="*70)
    print("VISUALIZING LEARNED POLICY")
    print("="*70)
    
    # Load model
    try:
        model = PPO.load(model_path)
        print(f"✓ Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model not found: {model_path}")
        print("Train a model first using train_with_stable_baselines3()")
        return
    
    # Create environment
    env = SwarmEnv(num_particles=2000, num_basis=9, grid_resolution=32, task='concentration')
    
    # Run episode and collect data
    obs, info = env.reset()
    observations = [obs.copy()]
    actions = []
    rewards = []
    
    for step in range(50):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        observations.append(obs.copy())
        actions.append(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot density evolution
    for i, idx in enumerate([0, len(observations)//2, len(observations)-1]):
        ax = axes[0, i]
        ax.imshow(observations[idx], cmap='hot', interpolation='nearest', origin='lower')
        ax.set_title(f'Density at Step {idx}')
        ax.set_xlabel('x (grid)')
        ax.set_ylabel('y (grid)')
    
    # Plot rewards
    axes[1, 0].plot(rewards, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Reward Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot actions (heatmap)
    if len(actions) > 0:
        action_array = np.array(actions)
        im = axes[1, 1].imshow(action_array.T, cmap='RdBu', aspect='auto', 
                              vmin=-5, vmax=5, interpolation='nearest')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Basis Function')
        axes[1, 1].set_title('Actions Over Time')
        plt.colorbar(im, ax=axes[1, 1], label='Strength')
    
    # Plot statistics
    axes[1, 2].axis('off')
    stats_text = f"""
Episode Statistics

Total Steps: {len(rewards)}
Total Reward: {sum(rewards):.2f}
Mean Reward: {np.mean(rewards):.4f}
Final Reward: {rewards[-1]:.4f}

Action Stats:
  Mean: {np.mean(actions):.3f}
  Std:  {np.std(actions):.3f}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, 
                   verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig('learned_policy_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to 'learned_policy_visualization.png'")
    plt.show()


def main():
    """
    Main function: Run different training/evaluation modes
    """
    print("\n" + "="*70)
    print("STOCHASTIC SWARM RL TRAINING EXAMPLES")
    print("="*70)
    print("\nChoose an option:")
    print("  1. Manual random search (simple, no dependencies)")
    print("  2. Train with stable-baselines3 PPO (requires stable-baselines3)")
    print("  3. Compare different policies")
    print("  4. Visualize learned policy (requires trained model)")
    print("  5. Run all examples")
    
    choice = input("\nEnter choice (1-5), or press Enter for option 1: ").strip()
    if not choice:
        choice = '1'
    
    if choice == '1':
        manual_random_search(num_episodes=100, steps_per_episode=50)
    
    elif choice == '2':
        train_with_stable_baselines3(total_timesteps=50000)
    
    elif choice == '3':
        compare_policies()
    
    elif choice == '4':
        visualize_learned_policy()
    
    elif choice == '5':
        print("\nRunning all examples...\n")
        manual_random_search(num_episodes=50, steps_per_episode=30)
        compare_policies()
        
        response = input("\nTrain with stable-baselines3? (takes several minutes) [y/N]: ")
        if response.lower() == 'y':
            model = train_with_stable_baselines3(total_timesteps=50000)
            if model is not None:
                visualize_learned_policy()
    
    else:
        print("Invalid choice!")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
