"""
tests/benchmark_env.py
======================
Performance benchmarks for the SwarmEnv Gymnasium environment and the
underlying C++ Python bindings (stochastic_swarm).

Sections
--------
1. C++ binding micro-benchmarks
   - ParticleSystem.step()         throughput  (particles/sec)
   - DensityGrid.update()          throughput  (particles/sec)
   - PotentialField.set_strengths  overhead
2. SwarmEnv gym interface
   - env.reset()                   latency
   - env.step()                    latency  &  stepped-particles/sec
3. Scaling sweeps
   - num_particles   : 100 -> 10 000
   - grid_resolution : 16  -> 128
   - physics_steps_per_action : 1, 5, 10, 20
   - num_basis       : 4   -> 64
4. Task reward computation
   - KLDivergenceTask vs WassersteinTask vs ConcentrationTask

Usage
-----
    python tests/benchmark_env.py              # full suite
    python tests/benchmark_env.py --quick      # reduced iterations
    python tests/benchmark_env.py --section 2  # only section 2
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, cast

import numpy as np

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

SEP_WIDE = "=" * 72
SEP_THIN = "-" * 72


def _hdr(title: str) -> None:
    print(f"\n{SEP_WIDE}\n  {title}\n{SEP_WIDE}")


def _sep() -> None:
    print(SEP_THIN)


def _col(value: Any, width: int, right: bool = True) -> str:
    s = str(value)
    return s.rjust(width) if right else s.ljust(width)


def _row(*cols: Any, widths: Tuple[int, ...] = (32, 14, 14, 14)) -> None:
    parts = [_col(c, w, right=(i > 0)) for i, (c, w) in enumerate(zip(cols, widths))]
    print("".join(parts))


def _warm_and_time(fn: Callable[[], Any], warmup: int, iters: int) -> float:
    """Warm up then time iters calls; returns average seconds-per-call."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


# ---------------------------------------------------------------------------
# Optional dependency imports – safe at module level
# ---------------------------------------------------------------------------

try:
    import stochastic_swarm as _ss_mod
    SS_OK: bool = True
except ImportError:
    _ss_mod = None  # type: ignore[assignment]
    SS_OK = False
    print(
        "[WARNING] stochastic_swarm C++ bindings not found. "
        "C++ micro-benchmarks will be skipped.\n"
        "Build with:  ./scripts/build.sh  (or cmake)."
    )

try:
    from swarm.envs.base import SwarmEnv as _SwarmEnv
    from swarm.envs.tasks import (
        ConcentrationTask as _ConcentrationTask,
        DispersionTask as _DispersionTask,
        KLDivergenceTask as _KLDivergenceTask,
        WassersteinTask as _WassersteinTask,
    )
    from swarm.utils.density import create_target as _create_target
    ENV_OK: bool = True
except ImportError as _e:
    _SwarmEnv = None  # type: ignore[assignment,misc]
    _ConcentrationTask = None  # type: ignore[assignment,misc]
    _DispersionTask = None  # type: ignore[assignment,misc]
    _KLDivergenceTask = None  # type: ignore[assignment,misc]
    _WassersteinTask = None  # type: ignore[assignment,misc]
    _create_target = None  # type: ignore[assignment]
    ENV_OK = False
    print(f"[WARNING] Could not import swarm package: {_e}")


# ===========================================================================
# Section 1 – C++ binding micro-benchmarks
# ===========================================================================

def bench_cpp_bindings(quick: bool = False) -> None:
    if not SS_OK or _ss_mod is None:
        print("[SKIP] Section 1: C++ bindings not available.")
        return

    ss = _ss_mod  # local alias – guaranteed non-None inside this branch

    _hdr("SECTION 1 · C++ Binding Micro-benchmarks")

    DOMAIN = 100.0
    ITERS  = 200 if not quick else 30
    WARMUP = 20  if not quick else 5

    # ── 1a  ParticleSystem.step() ────────────────────────────────────────
    print("\n[1a]  ParticleSystem.step() throughput\n")
    _row("Particles", "Step (µs)", "Throughput (M p/s)", "", widths=(14, 16, 22, 14))
    _sep()

    for N in [500, 1_000, 2_000, 5_000, 10_000]:
        ps = ss.ParticleSystem(num_particles=N, temperature=1.0,
                               num_basis=0, grid_res=32)
        ps.initialize_random(DOMAIN)
        spc = _warm_and_time(ps.step, WARMUP, ITERS)
        tp  = N / spc / 1e6
        _row(N, f"{spc*1e6:.2f}", f"{tp:.3f}", "", widths=(14, 16, 22, 14))

    # ── 1b  DensityGrid.update() ─────────────────────────────────────────
    print("\n[1b]  DensityGrid.update() throughput\n")
    _row("Particles", "Grid", "Update (µs)", "Throughput (M p/s)", widths=(12, 10, 14, 22))
    _sep()

    rng = np.random.default_rng(0)
    for N in [1_000, 5_000, 10_000]:
        x_list = rng.uniform(0, DOMAIN, N).tolist()
        y_list = rng.uniform(0, DOMAIN, N).tolist()
        for g in [16, 32, 64]:
            grid = ss.DensityGrid(g, g, DOMAIN)

            def _update(g_=grid, x_=x_list, y_=y_list) -> None:
                g_.clear()
                g_.update(x_, y_)

            spc = _warm_and_time(_update, WARMUP, ITERS)
            tp  = N / spc / 1e6
            _row(N, f"{g}×{g}", f"{spc*1e6:.2f}", f"{tp:.3f}", widths=(12, 10, 14, 22))
        _sep()

    # ── 1c  set_potential_params overhead ────────────────────────────────
    print("\n[1c]  set_potential_params() overhead\n")
    _row("Particles", "Basis fns", "Overhead (µs)", "", widths=(12, 12, 16, 12))
    _sep()

    N = 2_000
    for nb in [9, 16, 25, 36, 64]:
        ps = ss.ParticleSystem(num_particles=N, temperature=1.0,
                               num_basis=nb, grid_res=32)
        ps.initialize_random(DOMAIN)
        params = rng.uniform(-500.0, 500.0, nb).tolist()

        def _set(p_=ps, pr_=params) -> None:
            p_.set_potential_params(pr_)

        spc = _warm_and_time(_set, WARMUP, ITERS * 5)
        _row(N, nb, f"{spc*1e6:.2f}", "", widths=(12, 12, 16, 12))


# ===========================================================================
# Section 2 – SwarmEnv gym interface
# ===========================================================================

@dataclass
class _EnvCfg:
    num_particles:           int
    num_basis:               int
    grid_resolution:         int
    physics_steps_per_action: int


def bench_gym_interface(quick: bool = False) -> None:
    if not ENV_OK or _SwarmEnv is None:
        print("[SKIP] Section 2: swarm package not available.")
        return

    _hdr("SECTION 2 · SwarmEnv Gym Interface (reset / step)")

    ITERS_RESET = 20  if not quick else 5
    ITERS_STEP  = 100 if not quick else 20
    WARMUP      = 5   if not quick else 2

    configs = [
        _EnvCfg(num_particles=500,   num_basis=16, grid_resolution=32, physics_steps_per_action=5),
        _EnvCfg(num_particles=1_000, num_basis=16, grid_resolution=32, physics_steps_per_action=5),
        _EnvCfg(num_particles=2_000, num_basis=25, grid_resolution=32, physics_steps_per_action=10),
        _EnvCfg(num_particles=5_000, num_basis=25, grid_resolution=32, physics_steps_per_action=10),
    ]

    # ── 2a  reset latency ────────────────────────────────────────────────
    print("\n[2a]  env.reset() latency\n")
    _row("N part", "Basis", "Grid", "Reset (ms)")
    _sep()

    for cfg in configs:
        env = _SwarmEnv(
            num_particles=cfg.num_particles,
            num_basis=cfg.num_basis,
            grid_resolution=cfg.grid_resolution,
            physics_steps_per_action=cfg.physics_steps_per_action,
        )
        spc = _warm_and_time(lambda e=env: e.reset(), WARMUP, ITERS_RESET)
        _row(cfg.num_particles, cfg.num_basis,
             f"{cfg.grid_resolution}×{cfg.grid_resolution}",
             f"{spc*1e3:.2f} ms")
        env.close()

    # ── 2b  step latency ─────────────────────────────────────────────────
    print("\n[2b]  env.step() latency\n")
    _row("N part", "Phys steps", "Step (ms)", "Steps/sec")
    _sep()

    for cfg in configs:
        env = _SwarmEnv(
            num_particles=cfg.num_particles,
            num_basis=cfg.num_basis,
            grid_resolution=cfg.grid_resolution,
            physics_steps_per_action=cfg.physics_steps_per_action,
        )
        env.reset()
        action = env.action_space.sample()

        def _step(e=env, a=action):
            obs, r, term, trunc, info = e.step(a)
            if term or trunc:
                e.reset()

        spc = _warm_and_time(_step, WARMUP, ITERS_STEP)
        _row(cfg.num_particles,
             cfg.physics_steps_per_action,
             f"{spc*1e3:.2f} ms",
             f"{1.0/spc:.1f}")
        env.close()


# ===========================================================================
# Section 3 – Scaling sweeps
# ===========================================================================

def _time_env_step(
    env: Any,
    iters: int,
    warmup: int,
) -> Tuple[float, float]:
    """Returns (step_ms, steps_per_sec)."""
    env.reset()
    action = env.action_space.sample()

    def _step(e=env, a=action):
        obs, r, term, trunc, info = e.step(a)
        if term or trunc:
            e.reset()

    spc = _warm_and_time(_step, warmup, iters)
    return spc * 1e3, 1.0 / spc


def bench_scaling(quick: bool = False) -> None:
    if not ENV_OK or _SwarmEnv is None:
        print("[SKIP] Section 3: swarm package not available.")
        return

    _hdr("SECTION 3 · Scaling Sweeps")

    ITERS  = 60  if not quick else 15
    WARMUP = 5   if not quick else 2

    # ── 3a  num_particles sweep ──────────────────────────────────────────
    print("\n[3a]  num_particles sweep  (basis=16, grid=32, phys=10)\n")
    _row("Particles", "Step (ms)", "Steps/sec", "Sim-steps/s")
    _sep()

    PHYS = 10
    for N in [100, 500, 1_000, 2_000, 5_000, 10_000]:
        env = _SwarmEnv(num_particles=N, num_basis=16,
                        grid_resolution=32, physics_steps_per_action=PHYS)
        ms, sps = _time_env_step(env, ITERS, WARMUP)
        _row(N, f"{ms:.2f}", f"{sps:.1f}", f"{sps*PHYS:.1f}")
        env.close()

    # ── 3b  grid_resolution sweep ────────────────────────────────────────
    print("\n[3b]  grid_resolution sweep  (N=2000, basis=16, phys=10)\n")
    _row("Grid", "Step (ms)", "Steps/sec", "")
    _sep()

    for g in [16, 32, 64, 128]:
        env = _SwarmEnv(num_particles=2_000, num_basis=16,
                        grid_resolution=g, physics_steps_per_action=10)
        ms, sps = _time_env_step(env, ITERS, WARMUP)
        _row(f"{g}×{g}", f"{ms:.2f}", f"{sps:.1f}", "")
        env.close()

    # ── 3c  physics_steps_per_action sweep ──────────────────────────────
    print("\n[3c]  physics_steps_per_action sweep  (N=2000, basis=16, grid=32)\n")
    _row("Phys steps", "Step (ms)", "Steps/sec", "Sim-steps/s")
    _sep()

    for p in [1, 2, 5, 10, 20, 50]:
        env = _SwarmEnv(num_particles=2_000, num_basis=16,
                        grid_resolution=32, physics_steps_per_action=p)
        ms, sps = _time_env_step(env, ITERS, WARMUP)
        _row(p, f"{ms:.2f}", f"{sps:.1f}", f"{sps*p:.1f}")
        env.close()

    # ── 3d  num_basis sweep ──────────────────────────────────────────────
    print("\n[3d]  num_basis sweep  (N=2000, grid=32, phys=10)\n")
    _row("Basis fns", "Step (ms)", "Steps/sec", "")
    _sep()

    for nb in [4, 9, 16, 25, 36, 49, 64]:
        env = _SwarmEnv(num_particles=2_000, num_basis=nb,
                        grid_resolution=32, physics_steps_per_action=10)
        ms, sps = _time_env_step(env, ITERS, WARMUP)
        _row(nb, f"{ms:.2f}", f"{sps:.1f}", "")
        env.close()


# ===========================================================================
# Section 4 – Task reward computation overhead
# ===========================================================================

def bench_task_rewards(quick: bool = False) -> None:
    # Import tasks locally so Pylance/mypy can narrow their types here.
    try:
        from swarm.envs.tasks import (  # noqa: PLC0415
            ConcentrationTask,
            DispersionTask,
            KLDivergenceTask,
            WassersteinTask,
        )
        from swarm.utils.density import create_target  # noqa: PLC0415
    except ImportError:
        print("[SKIP] Section 4: swarm package not available.")
        return

    _hdr("SECTION 4 · Task Reward Computation Overhead")

    ITERS  = 200 if not quick else 40
    WARMUP = 20  if not quick else 5

    rng = np.random.default_rng(42)
    G   = 32
    N   = 2_000

    density = rng.random((G, G)).astype(np.float32)
    density /= density.sum()

    target = create_target("ring_gaussian", grid_resolution=G, total_particles=N)

    # Dummy env exposing only the attrs that tasks inspect
    class _DummyEnv:
        num_particles = N
        grid_res      = G

    dummy = _DummyEnv()

    tasks: Dict[str, Any] = {
        "ConcentrationTask" : ConcentrationTask(),
        "DispersionTask"    : DispersionTask(),
        "KLDivergenceTask"  : KLDivergenceTask(target=target.copy()),
        "WassersteinTask"   : WassersteinTask(target=target.copy()),
    }

    print("\n")
    _row("Task", "Per call (µs)", "Calls/sec", "", widths=(24, 16, 16, 10))
    _sep()

    for name, task in tasks.items():
        def _call(t=task, d=density, e=dummy):
            t.compute(d, e)

        spc  = _warm_and_time(_call, WARMUP, ITERS)
        cps  = 1.0 / spc
        _row(name, f"{spc*1e6:.2f}", f"{cps:.1f}", "", widths=(24, 16, 16, 10))


# ===========================================================================
# System info
# ===========================================================================

def _print_system_info() -> None:
    import platform
    print("\n" + SEP_WIDE)
    print("  System information")
    print(SEP_THIN)
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  OS        : {platform.platform()}")
    print(f"  CPU       : {platform.processor() or 'n/a'}")
    print(f"  NumPy     : {np.__version__}")
    try:
        import stable_baselines3 as sb3
        print(f"  SB3       : {sb3.__version__}")
    except ImportError:
        pass
    print(f"  C++ ext   : {'stochastic_swarm ✓' if SS_OK else 'stochastic_swarm ✗  (not built)'}")
    print(SEP_WIDE + "\n")


# ===========================================================================
# Entry point
# ===========================================================================

_DESCRIPTION = """\
SwarmEnv performance benchmarks.

Sections:
  1  C++ binding micro-benchmarks  (ParticleSystem, DensityGrid)
  2  SwarmEnv gym interface        (reset / step latency)
  3  Scaling sweeps                (particles, grid, phys-steps, basis)
  4  Task reward overhead          (KL, Wasserstein, Concentration)
"""


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=_DESCRIPTION)
    parser.add_argument("--quick", action="store_true", help="Reduce iteration counts for a fast smoke-test run.")
    parser.add_argument("--section", type=int, choices=[1, 2, 3, 4], default=None, help="Run only the specified section (1-4).")
    args = parser.parse_args()

    _print_system_info()

    runners: Dict[int, Callable[[bool], None]] = {
        1: bench_cpp_bindings,
        2: bench_gym_interface,
        3: bench_scaling,
        4: bench_task_rewards,
    }

    if args.section is not None:
        runners[args.section](args.quick)
    else:
        for fn in runners.values():
            fn(args.quick)

    print(f"\n{SEP_WIDE}")
    print("  All benchmarks complete.")
    print(SEP_WIDE + "\n")


if __name__ == "__main__":
    main()
