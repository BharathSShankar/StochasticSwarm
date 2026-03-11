"""
Visualization and GIF Generation Demo

Demonstrates how to easily get visualizations and GIFs during training
or inference using the swarm environment.
"""

import numpy as np
from pathlib import Path

# Import swarm environment and utilities
from swarm.envs import SwarmEnv
from swarm.utils import create_gif, save_snapshot


def demo_manual_recording():
    """Demo: Manually recording frames and creating a GIF."""
    print("\n" + "="*70)
    print("Demo 1: Manual Frame Recording and GIF Generation")
    print("="*70)
    
    # Create environment
    env = SwarmEnv(
        task='concentration',
        num_particles=2000,
        max_steps=50,
        verbose=1,
    )
    
    # Record frames manually
    frames = []
    obs, info = env.reset()
    
    print("\nRunning simulation and recording frames...")
    for step in range(30):
        # Record current state
        state = env.get_state_dict()
        frames.append(state)
        
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            print(f"  Step {step}/{30} - Reward: {reward:.4f}")
        
        if terminated or truncated:
            break
    
    print(f"\nRecorded {len(frames)} frames")
    
    # Create GIF using utility function
    output_dir = Path("outputs/viz_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gif_path = create_gif(
        frames,
        output_path=output_dir / "manual_recording.gif",
        fps=10,
        mode='combined',
        domain_size=env.domain_size,
        verbose=True,
    )
    
    print(f"\n✓ GIF saved to: {gif_path}")
    env.close()


def demo_env_methods():
    """Demo: Using environment's built-in visualization methods."""
    print("\n" + "="*70)
    print("Demo 2: Environment Built-in Visualization Methods")
    print("="*70)
    
    env = SwarmEnv(
        task='corner',
        num_particles=1500,
        max_steps=100,
        verbose=1,
    )
    
    obs, info = env.reset()
    
    # Run for a few steps
    print("\nRunning simulation...")
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nCurrent step: {env.current_step}")
    
    # Save snapshot using environment method
    output_dir = Path("outputs/viz_demo")
    snapshot_path = output_dir / "snapshot_combined.png"
    
    print(f"\nSaving snapshot to {snapshot_path}...")
    env.visualize(mode='combined', save_path=str(snapshot_path))
    print("✓ Snapshot saved")
    
    # Get state dict for custom processing
    state = env.get_state_dict()
    print(f"\nState dict keys: {list(state.keys())}")
    print(f"  Density shape: {state['density'].shape}")
    print(f"  Particle count: {len(state['positions'])}")
    print(f"  Timestep: {state['timestep']}")
    print(f"  Episode return: {state['episode_return']:.4f}")
    
    env.close()


def demo_training_with_callback():
    """Demo: Using VisualizationCallback during training."""
    print("\n" + "="*70)
    print("Demo 3: Automatic Visualization During Training")
    print("="*70)
    
    try:
        from stable_baselines3 import PPO
        from swarm.training.callbacks import VisualizationCallback
    except ImportError:
        print("\n⚠ Stable-Baselines3 not available. Skipping training demo.")
        return
    
    # Create environment
    env = SwarmEnv(
        task='concentration',
        num_particles=1000,
        max_steps=50,
    )
    
    # Setup visualization callback
    output_dir = Path("outputs/viz_demo")
    callback = VisualizationCallback(
        log_dir=str(output_dir / "training_logs"),
        viz_freq=1000,      # Visualize every 1000 steps
        max_frames=50,      # Store 50 frames max
        save_gif=True,      # Auto-save GIF at end
        gif_mode='combined', # Show both density and particles
        gif_fps=10,
        domain_size=env.domain_size,
        verbose=1,
    )
    
    # Create and train model
    print("\nTraining PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device='cpu',
    )
    
    model.learn(
        total_timesteps=5000,
        callback=callback,
        progress_bar=True,
    )
    
    print("\n✓ Training complete!")
    print(f"✓ GIF automatically saved to: {output_dir / 'training_logs'}")
    print(f"✓ Recorded {len(callback.get_frames())} frames")
    
    env.close()


def demo_inference_visualization():
    """Demo: Visualizing a trained policy or random actions."""
    print("\n" + "="*70)
    print("Demo 4: Inference Visualization")
    print("="*70)
    
    env = SwarmEnv(
        task='dispersion',
        num_particles=2000,
        max_steps=100,
        verbose=1,
    )
    
    # Collect frames during inference
    frames = []
    obs, info = env.reset()
    
    print("\nRunning inference and collecting frames...")
    done = False
    step = 0
    
    while not done and step < 40:
        # Record every 2nd frame to reduce GIF size
        if step % 2 == 0:
            frames.append(env.get_state_dict())
        
        # Use a simple heuristic policy (negative gradient to disperse)
        action_dim = env.action_space.shape[0] if env.action_space.shape else 16
        action = np.random.randn(action_dim) * 0.5
        obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        step += 1
    
    print(f"\nCompleted {step} steps, recorded {len(frames)} frames")
    
    # Create GIF from inference
    output_dir = Path("outputs/viz_demo")
    
    # Try density-only mode
    gif_path_density = env.create_gif(
        frames,
        output_path=str(output_dir / "inference_density.gif"),
        fps=5,
        mode='density',
    )
    print(f"✓ Density GIF saved to: {gif_path_density}")
    
    # Try particles-only mode
    gif_path_particles = env.create_gif(
        frames,
        output_path=str(output_dir / "inference_particles.gif"),
        fps=5,
        mode='particles',
    )
    print(f"✓ Particles GIF saved to: {gif_path_particles}")
    
    env.close()


def demo_snapshot_saving():
    """Demo: Saving individual snapshots."""
    print("\n" + "="*70)
    print("Demo 5: Saving Individual Snapshots")
    print("="*70)
    
    env = SwarmEnv(
        task='concentration',
        num_particles=3000,
        max_steps=100,
    )
    
    obs, info = env.reset()
    output_dir = Path("outputs/viz_demo/snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCapturing snapshots at different stages...")
    
    # Initial state
    env.visualize(mode='combined', save_path=str(output_dir / "step_000.png"))
    print("  ✓ Saved initial state")
    
    # After 20 steps
    for _ in range(20):
        action = env.action_space.sample()
        env.step(action)
    env.visualize(mode='combined', save_path=str(output_dir / "step_020.png"))
    print("  ✓ Saved step 20")
    
    # After 50 steps
    for _ in range(30):
        action = env.action_space.sample()
        env.step(action)
    env.visualize(mode='combined', save_path=str(output_dir / "step_050.png"))
    print("  ✓ Saved step 50")
    
    # Using save_snapshot directly for more control
    state = env.get_state_dict()
    save_snapshot(
        density=state['density'],
        positions=state['positions'],
        output_path=output_dir / "custom_snapshot.png",
        domain_size=env.domain_size,
        title=f"Custom Snapshot - Step {state['timestep']}",
        mode='combined',
        dpi=200,  # Higher quality
    )
    print("  ✓ Saved custom high-quality snapshot")
    
    print(f"\n✓ All snapshots saved to: {output_dir}")
    env.close()


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("SWARM VISUALIZATION AND GIF GENERATION DEMO")
    print("="*70)
    print("\nThis demo shows how to:")
    print("  1. Manually record frames and create GIFs")
    print("  2. Use environment's built-in visualization methods")
    print("  3. Automatically visualize during training")
    print("  4. Create GIFs from inference")
    print("  5. Save individual snapshots")
    
    # Run demos
    demo_manual_recording()
    demo_env_methods()
    demo_snapshot_saving()
    demo_inference_visualization()
    demo_training_with_callback()  # This one requires SB3
    
    print("\n" + "="*70)
    print("ALL DEMOS COMPLETE!")
    print("="*70)
    print("\nCheck the 'outputs/viz_demo' directory for generated files:")
    print("  - *.gif files (animations)")
    print("  - *.png files (snapshots)")
    print("  - training_logs/ (TensorBoard logs if training ran)")
    print("\nTo view TensorBoard logs:")
    print("  tensorboard --logdir=outputs/viz_demo/training_logs")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
