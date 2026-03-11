"""
Week 4 Example: Distribution Matching with KL Divergence & Wasserstein Distance

Goal: Train a PPO agent to shape the stochastic swarm so its empirical density
matches a chosen target distribution by minimising KL Divergence (or Wasserstein
Earth Mover's Distance).

Usage
-----
# KL divergence, ring target (default)
python examples/week4_distribution_matching.py

# Wasserstein distance, double-Gaussian target, 500K steps
python examples/week4_distribution_matching.py --task wasserstein \
       --target double_gaussian --steps 500000

# Custom image target
python examples/week4_distribution_matching.py --image path/to/logo.png
"""

import argparse
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Week 4 – Distribution Matching")
    parser.add_argument(
        "--task", choices=["kl", "wasserstein"], default="kl",
        help="Reward metric (default: kl)"
    )
    parser.add_argument(
        "--target",
        choices=["gaussian", "double_gaussian", "ring", "ring_gaussian",
                 "corners", "cross", "uniform"],
        default="ring",
        help="Target distribution shape (default: ring)"
    )
    parser.add_argument(
        "--image", default=None,
        help="Path to an image to use as target density (overrides --target)"
    )
    parser.add_argument("--steps",    type=int, default=200_000, help="Training timesteps")
    parser.add_argument("--particles", type=int, default=3000,   help="Number of particles")
    parser.add_argument("--grid",      type=int, default=32,     help="Grid resolution")
    parser.add_argument("--n-envs",    type=int, default=4,      help="Parallel envs")
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--no-visualize", dest="visualize", action="store_false")
    parser.add_argument("--save-gif",  action="store_true", default=False)
    return parser.parse_args()


def build_target(args) -> np.ndarray:
    """Return a float32 target density grid."""
    from swarm.utils.density import create_target, image_to_density

    if args.image is not None:
        print(f"[target] loading image: {args.image}")
        target = image_to_density(
            args.image,
            grid_resolution=args.grid,
            total_particles=args.particles,
        )
    else:
        print(f"[target] generating '{args.target}' distribution")
        target = create_target(
            args.target,
            grid_resolution=args.grid,
            total_particles=args.particles,
        )
    return target


def show_target(target: np.ndarray, shape_name: str):
    """Pretty-print target stats and optionally show a plot."""
    print(f"  shape   : {target.shape}")
    print(f"  sum     : {target.sum():.1f}")
    print(f"  max     : {target.max():.4f}")
    print(f"  nonzero : {(target > 0).sum()} / {target.size} cells")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 4))
        plt.imshow(target, cmap="hot", origin="lower")
        plt.colorbar(label="density")
        plt.title(f"Target: {shape_name}")
        plt.tight_layout()
        plt.savefig(f"target_{shape_name}.png", dpi=120)
        print(f"  saved  : target_{shape_name}.png")
        plt.close()
    except ImportError:
        pass


# ── environment factory ───────────────────────────────────────────────────────

def make_env_fn(task_name: str, target: np.ndarray, num_particles: int,
                grid_resolution: int):
    """Return a callable that creates a fresh SwarmEnv for SB3 VecEnv."""
    def _make():
        from swarm import SwarmEnv, KLDivergenceTask, WassersteinTask

        if task_name == "kl":
            task = KLDivergenceTask(
                target=target.copy(),
                success_threshold=0.02,
                improvement_bonus=300.0,
            )
        else:
            task = WassersteinTask(
                target=target.copy(),
                num_projections=64,
                success_threshold=0.005,
                improvement_bonus=300.0,
            )

        env = SwarmEnv(
            task=task,
            num_particles=num_particles,
            temperature=1.0,
            num_basis=16,
            grid_resolution=grid_resolution,
            physics_steps_per_action=10,
            max_steps=150,
            learnable_max_force=True,
            action_smoothing=0.1,
        )
        return env
    return _make


# ── evaluation helper ─────────────────────────────────────────────────────────

def evaluate_and_report(trainer, target: np.ndarray, task_name: str, n_episodes: int = 5):
    """Run n_episodes and print KL/Wasserstein of the final state."""
    from swarm.utils.density import kl_divergence, wasserstein_distance_2d

    print("\n── Evaluation ──────────────────────────────────────────────")
    stats = trainer.evaluate(n_episodes=n_episodes)
    print(f"  mean reward : {stats.get('mean_reward', float('nan')):.4f}")
    print(f"  std  reward : {stats.get('std_reward',  float('nan')):.4f}")

    # Run one extra episode to get the final density for metric reporting
    env = trainer.env_fn() if hasattr(trainer, 'env_fn') else None
    if env is not None:
        obs, _ = env.reset()
        for _ in range(150):
            action, _ = trainer.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        final_density = obs  # (32, 32) float32
        kl   = kl_divergence(final_density, target)
        swd  = wasserstein_distance_2d(final_density, target)
        print(f"  final KL    : {kl:.6f}  (0 = perfect)")
        print(f"  final SWD   : {swd:.6f}  (0 = perfect)")
        env.close()

    print("────────────────────────────────────────────────────────────\n")
    return stats


# ── GIF recording ─────────────────────────────────────────────────────────────

def record_gif(trainer, target: np.ndarray, task_name: str, output: str = "week4_result.gif"):
    """Record a 100-step episode and save as GIF."""
    print(f"[gif] recording episode → {output}")
    env = trainer.env_fn()
    obs, _ = env.reset()
    frames = [env.get_state_dict()]

    for _ in range(100):
        action, _ = trainer.model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.get_state_dict())
        if terminated or truncated:
            break

    path = env.create_gif(frames, output, fps=8, mode="combined")
    env.close()
    if path:
        print(f"[gif] saved: {path}")
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  Week 4 — Distribution Matching")
    print(f"  task     : {args.task.upper()} Divergence")
    print(f"  target   : {args.image or args.target}")
    print(f"  steps    : {args.steps:,}")
    print(f"  particles: {args.particles:,}")
    print(f"  n_envs   : {args.n_envs}")
    print("=" * 60)

    # 1. Build target distribution
    target = build_target(args)
    show_target(target, args.image or args.target)

    # 2. Visualise target + compute baseline
    from swarm.utils.density import kl_divergence, wasserstein_distance_2d
    uniform = np.ones_like(target)
    kl_base  = kl_divergence(uniform, target)
    swd_base = wasserstein_distance_2d(uniform, target)
    print(f"\n[baseline] KL (uniform vs target)  : {kl_base:.4f}")
    print(f"[baseline] SWD (uniform vs target) : {swd_base:.4f}")

    # 3. Training config
    from swarm import TrainingConfig, Trainer

    config = TrainingConfig(
        total_timesteps=args.steps,
        algorithm="PPO",
        policy="MlpPolicy",
        learning_rate=3e-4,
        lr_schedule="cosine",
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        n_envs=args.n_envs,
        checkpoint_freq=50_000,
        eval_freq=25_000,
        tensorboard=True,
        visualize=args.visualize,
        viz_freq=10_000,
        experiment_name=f"week4_{args.task}_{args.image or args.target}",
        verbose=1,
    )

    env_fn = make_env_fn(args.task, target, args.particles, args.grid)

    trainer = Trainer(env_fn=env_fn, config=config)

    # 4. Train
    print("\n[train] starting …  (tensorboard: tensorboard --logdir=./runs)\n")
    trainer.train()
    trainer.save(f"week4_{args.task}_{args.image or args.target}")

    # 5. Evaluate
    evaluate_and_report(trainer, target, args.task)

    # 6. Optional GIF
    if args.save_gif:
        record_gif(trainer, target, args.task)

    print("Done! ✔")


if __name__ == "__main__":
    main()
