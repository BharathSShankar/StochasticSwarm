"""
Week 4 Visualizer: Watch the swarm form a ring (or any target distribution).

This script:
  1. Optionally trains a fresh agent (or loads an existing model).
  2. Runs a rollout episode using the deterministic policy.
  3. Saves a 3-panel animated GIF:
       [Current density]  |  [Target distribution]  |  [Difference]
     with KL divergence and step number on each frame.

Usage
-----
# Visualize using the existing saved model (runs training if model not found)
python examples/week4_visualize.py

# Specify a saved model zip
python examples/week4_visualize.py --model week4_kl_ring_gaussian.zip

# Choose a different target shape
python examples/week4_visualize.py --target double_gaussian --model week4_kl_ring_gaussian.zip

# Train from scratch then visualize (200K steps)
python examples/week4_visualize.py --train --steps 200000 --target ring_gaussian
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — works in any terminal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Week 4 — Distribution Matching Visualizer')
    p.add_argument('--model',   default='week4_kl_ring_gaussian.zip',
                   help='Path to saved SB3 model zip (default: week4_kl_ring_gaussian.zip)')
    p.add_argument('--target',  default='ring_gaussian',
                   choices=['gaussian','double_gaussian','ring','ring_gaussian',
                            'corners','cross','uniform'],
                   help='Target distribution shape (default: ring_gaussian)')
    p.add_argument('--task',    default='kl', choices=['kl','wasserstein'],
                   help='Reward metric (default: kl)')
    p.add_argument('--particles', type=int, default=2000)
    p.add_argument('--steps',   type=int, default=200_000,
                   help='Training steps if --train is set')
    p.add_argument('--n-envs',  type=int, default=4)
    p.add_argument('--train',   action='store_true',
                   help='Force a fresh training run before visualizing')
    p.add_argument('--episode-steps', type=int, default=150,
                   help='Steps to run for the recorded episode')
    p.add_argument('--fps',     type=int, default=8)
    p.add_argument('--output',  default='ring_forming.gif',
                   help='Output GIF path (default: ring_forming.gif)')
    p.add_argument('--dpi',     type=int, default=120)
    return p.parse_args()


# ── Environment factory ───────────────────────────────────────────────────────

def make_env_fn(task_name, target, num_particles):
    from swarm import SwarmEnv, KLDivergenceTask, WassersteinTask

    def _make():
        if task_name == 'kl':
            task = KLDivergenceTask(target=target.copy(), success_threshold=0.02, improvement_bonus=300.0)
        else:
            task = WassersteinTask(target=target.copy(), num_projections=64, success_threshold=0.005, improvement_bonus=300.0)
        return SwarmEnv(
            task=task,
            num_particles=num_particles,
            temperature=1.0,
            num_basis=16,
            grid_resolution=32,
            physics_steps_per_action=10,
            max_steps=150,
            learnable_max_force=True,
            action_smoothing=0.1,
        )
    return _make


# ── Training ─────────────────────────────────────────────────────────────────

def train(args, target):
    from swarm import Trainer, TrainingConfig

    print(f'[train] {args.steps:,} steps  task={args.task}  target={args.target}')
    config = TrainingConfig(
        total_timesteps=args.steps,
        algorithm='PPO',
        lr_schedule='cosine',
        n_envs=args.n_envs,
        batch_size=128,
        n_epochs=10,
        checkpoint_freq=50_000,
        eval_freq=25_000,
        tensorboard=True,
        visualize=False,
        experiment_name=f'week4_{args.task}_{args.target}',
        verbose=1,
    )
    trainer = Trainer(env_fn=make_env_fn(args.task, target, args.particles), config=config)
    trainer.train()
    save_path = f'week4_{args.task}_{args.target}'
    trainer.save(save_path)
    print(f'[train] saved → {save_path}.zip')
    return save_path + '.zip'


# ── Rollout recording ─────────────────────────────────────────────────────────

def record_episode(model, env_fn, episode_steps):
    """Run one deterministic episode and collect per-step density snapshots."""
    from swarm.utils.density import kl_divergence, wasserstein_distance_2d

    env = env_fn()
    obs, _ = env.reset()

    frames = []
    kl_history = []
    task = env._reward_fn

    for step in range(episode_steps):
        density = env.sim.get_density_grid().get_grid().copy().astype(np.float32)
        target  = task._get_target(env)
        kl      = kl_divergence(density, target)
        kl_history.append(kl)

        frames.append({
            'density': density.copy(),
            'target':  target.copy(),
            'step':    step,
            'kl':      kl,
        })

        action, _ = model.predict(obs[np.newaxis], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])

        if terminated or truncated:
            print(f'  episode ended at step {step}')
            break

    env.close()
    return frames, kl_history


# ── Frame rendering ───────────────────────────────────────────────────────────

def build_frame(frame_data, vmax_current, vmax_target, fig, axes, kl_history):
    """Update the figure axes for a single frame."""
    ax_current, ax_target, ax_diff, ax_kl = axes

    density = frame_data['density']
    target  = frame_data['target']
    step    = frame_data['step']
    kl      = frame_data['kl']
    diff    = density / (density.sum() + 1e-8) - target / (target.sum() + 1e-8)

    # Panel 1 — current density
    ax_current.clear()
    ax_current.imshow(density, cmap='hot', origin='lower', vmin=0, vmax=vmax_current)
    ax_current.set_title(f'Swarm Density\n(step {step})', fontsize=9)
    ax_current.axis('off')

    # Panel 2 — target distribution
    ax_target.clear()
    ax_target.imshow(target, cmap='plasma', origin='lower', vmin=0, vmax=vmax_target)
    ax_target.set_title('Target Distribution\n(ring_gaussian)', fontsize=9)
    ax_target.axis('off')

    # Panel 3 — normalized difference
    ax_diff.clear()
    vabs = max(abs(diff.min()), abs(diff.max()), 1e-6)
    ax_diff.imshow(diff, cmap='coolwarm', origin='lower', vmin=-vabs, vmax=vabs)
    ax_diff.set_title(f'Difference\nKL = {kl:.3f}', fontsize=9)
    ax_diff.axis('off')

    # Panel 4 — KL over time
    ax_kl.clear()
    ax_kl.plot(kl_history[:step + 1], color='steelblue', linewidth=1.5)
    ax_kl.axhline(0, color='green', linestyle='--', linewidth=0.8, alpha=0.5, label='Perfect (0)')
    ax_kl.set_xlim(0, max(len(kl_history) - 1, 1))
    ax_kl.set_ylim(bottom=0)
    ax_kl.set_xlabel('Step', fontsize=8)
    ax_kl.set_ylabel('D_KL(P‖Q)', fontsize=8)
    ax_kl.set_title('KL Divergence', fontsize=9)
    ax_kl.tick_params(labelsize=7)
    ax_kl.grid(True, alpha=0.3)

    fig.suptitle(
        f'Week 4 Distribution Matching — Ring Gaussian Target'
        f'\nStep {step:3d}  |  KL = {kl:.4f}  |  PPO (200K steps)',
        fontsize=10,
    )


# ── GIF generator ─────────────────────────────────────────────────────────────

def save_gif(frames, kl_history, output_path, fps, dpi):
    from matplotlib.animation import PillowWriter, FuncAnimation

    print(f'[gif] rendering {len(frames)} frames → {output_path}')

    vmax_c = max(f['density'].max() for f in frames) + 1e-6
    vmax_t = frames[0]['target'].max() + 1e-6

    fig = plt.figure(figsize=(14, 4), dpi=dpi)
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)
    ax_current = fig.add_subplot(gs[0, 0])
    ax_target  = fig.add_subplot(gs[0, 1])
    ax_diff    = fig.add_subplot(gs[0, 2])
    ax_kl      = fig.add_subplot(gs[0, 3])
    axes = (ax_current, ax_target, ax_diff, ax_kl)

    def update(i):
        build_frame(frames[i], vmax_c, vmax_t, fig, axes, kl_history)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f'[gif] saved: {output_path}  ({os.path.getsize(output_path) / 1024:.1f} KB)')


# ── Static comparison PNG ─────────────────────────────────────────────────────

def save_comparison_png(frames, output_path='ring_comparison.png', dpi=150):
    """Save a 3-row snapshot: start | middle | end."""
    indices = [0, len(frames) // 2, len(frames) - 1]
    labels  = ['Start (random)', 'Midpoint', 'Final']

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for row, (idx, label) in enumerate(zip(indices, labels)):
        f = frames[idx]
        density = f['density']
        target  = f['target']
        diff    = (density / (density.sum() + 1e-8)
                   - target / (target.sum() + 1e-8))

        axes[row, 0].imshow(density, cmap='hot',     origin='lower')
        axes[row, 0].set_title(f'{label} — Swarm', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(target,  cmap='plasma',  origin='lower')
        axes[row, 1].set_title(f'{label} — Target', fontsize=10)
        axes[row, 1].axis('off')

        vabs = max(abs(diff.min()), abs(diff.max()), 1e-6)
        axes[row, 2].imshow(diff, cmap='coolwarm', origin='lower', vmin=-vabs, vmax=vabs)
        axes[row, 2].set_title(f'{label} — Diff   KL={f["kl"]:.3f}', fontsize=10)
        axes[row, 2].axis('off')

    plt.suptitle('Week 4: KL Distribution Matching — Ring Gaussian', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'[png] saved: {output_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    from swarm.utils.density import create_target, kl_divergence

    target = create_target(args.target, grid_resolution=32, total_particles=args.particles)
    print(f'[target] {args.target}: sum={target.sum():.0f}  max={target.max():.3f}')

    # ── load or train model ──────────────────────────────────────────────────
    model_path = args.model
    if args.train or not Path(model_path).exists():
        if not Path(model_path).exists():
            print(f'[info] model not found: {model_path} — training now')
        model_path = train(args, target)

    from stable_baselines3 import PPO
    print(f'[load] {model_path}')
    model = PPO.load(model_path)

    # ── record episode ───────────────────────────────────────────────────────
    print(f'[record] {args.episode_steps}-step deterministic episode')
    env_fn = make_env_fn(args.task, target, args.particles)
    frames, kl_history = record_episode(model, env_fn, args.episode_steps)

    kl_start = kl_history[0]
    kl_end   = kl_history[-1]
    print(f'[result] KL start={kl_start:.4f}  end={kl_end:.4f}  '
          f'improvement={kl_start - kl_end:.4f} ({100*(kl_start-kl_end)/kl_start:.1f}%)')

    # ── save outputs ─────────────────────────────────────────────────────────
    save_gif(frames, kl_history, args.output, fps=args.fps, dpi=args.dpi)
    save_comparison_png(frames, output_path=args.output.replace('.gif', '_comparison.png'))

    print()
    print('=' * 55)
    print(f'  GIF saved  : {args.output}')
    print(f'  PNG saved  : {args.output.replace(".gif","_comparison.png")}')
    print(f'  KL start   : {kl_start:.4f}')
    print(f'  KL final   : {kl_end:.4f}')
    print('=' * 55)


if __name__ == '__main__':
    main()
