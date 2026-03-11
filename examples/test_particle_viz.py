"""
Quick test to verify particle position visualization works
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from python.rl_template import RLTrainer, TensorBoardConfig
from python.swarm_env import SwarmConcentrationEnv

# Test environment has get_particle_positions method
env = SwarmConcentrationEnv(num_particles=500)
obs, info = env.reset()
print(f"✓ Environment created: {env}")

# Test get_particle_positions
positions = env.get_particle_positions()
print(f"✓ Particle positions shape: {positions.shape}")
print(f"✓ Position sample: {positions[:3]}")

# Test quick training with visualization
print("\n" + "="*70)
print("Testing Particle Visualization in TensorBoard")
print("="*70)

config = TensorBoardConfig(
    experiment_name='particle_viz_test',
    log_visualizations=True,
    visualization_freq=1000,  # Log every 1000 steps
    checkpoint_freq=0,  # No checkpoints for quick test
    eval_freq=0  # No evaluation for quick test
)

trainer = RLTrainer(
    env_fn=lambda: SwarmConcentrationEnv(num_particles=800, num_basis=9),
    algorithm='PPO',
    tb_config=config,
    verbose=1
)

print("\nTraining for 3000 steps (will log visualizations at steps 1000, 2000, 3000)...")
trainer.train(total_timesteps=3000, progress_bar=True)

trainer.close()

print("\n" + "="*70)
print("✓ Test Complete!")
print("="*70)
print("\nView particle evolution in TensorBoard:")
print("  1. Run: tensorboard --logdir=./runs")
print("  2. Open: http://localhost:6006")
print("  3. Go to IMAGES tab")
print("  4. Look for:")
print("     - visualization/density_map (heatmaps)")
print("     - visualization/particle_positions (scatter plots)")
print("  5. Use the slider to see evolution over time!")
print("="*70)
