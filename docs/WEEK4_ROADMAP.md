# Week 4 Roadmap — Distribution Matching (KL Divergence)

**Status: ✅ Implemented**

---

## 🎯 Goal

Train a PPO agent to control a Langevin-dynamics particle swarm so that its
empirical probability density function (PDF) converges to an arbitrary target
distribution by **minimising an information-theoretic divergence**.

---

## 📐 The Math

### Discrete 2-D Probability Distribution

The swarm occupies a 32 × 32 density grid.  Normalising it gives a discrete PDF:

```
P(x, y) = count(x, y) / N_particles
```

### KL Divergence (Forward)

```
D_KL(P ‖ Q) = Σ_{x,y}  P(x,y) · log[ P(x,y) / Q(x,y) ]
```

- **P** = current (empirical) swarm density  
- **Q** = target distribution  
- `D_KL ≥ 0`, with equality iff P = Q  
- **Reward** = `−D_KL(P ‖ Q)` (higher = better)

#### Forward vs Reverse KL

| Direction | Formula | Behaviour |
|-----------|---------|-----------|
| Forward `D_KL(P‖Q)` | Σ P log(P/Q) | **Mean-seeking** — agent spreads to cover all modes in Q |
| Reverse `D_KL(Q‖P)` | Σ Q log(Q/P) | **Mode-seeking** — agent collapses to single mode of Q |
| Symmetric | (fwd + rev) / 2 | Balance between both |

### Wasserstein / Earth Mover's Distance (EMD)

For each random projection direction **d** ∈ S¹:

```
SWD(P, Q) = E_d[ W₁(P_d, Q_d) ]
```

where `P_d`, `Q_d` are 1-D marginals along **d** and `W₁` is the 1-D
Wasserstein distance (area between CDFs).

- Uses 64 random projections (**Sliced Wasserstein Distance**, O(K·N log N))  
- More robust than KL when distributions have disjoint support  
- **Reward** = `−SWD(P, Q)`

---

## ✅ Tasks Implemented

### 1. `KLDivergenceTask` ([`swarm/envs/tasks.py`](../swarm/envs/tasks.py))

```python
from swarm import KLDivergenceTask, SwarmEnv
from swarm.utils.density import create_target

# Ring target
target = create_target('ring', grid_resolution=32, total_particles=3000)

task = KLDivergenceTask(
    target=target,
    epsilon=1e-8,           # smoothing (avoids log 0)
    scale=1.0,              # reward multiplier
    success_threshold=0.02, # D_KL ≤ this → episode solved
    improvement_bonus=300., # extra reward on new best
)

env = SwarmEnv(task=task, num_particles=3000)
```

### 2. `WassersteinTask` ([`swarm/envs/tasks.py`](../swarm/envs/tasks.py))

```python
from swarm import WassersteinTask, SwarmEnv

task = WassersteinTask(
    target=target,
    num_projections=64,      # SWD projections
    scale=10.0,              # reward multiplier
    success_threshold=0.005, # SWD ≤ this → episode solved
    improvement_bonus=300.,
)

env = SwarmEnv(task=task, num_particles=3000)
```

Or use the short-hand string names:

```python
env = SwarmEnv(task='kl',         target_density=target)  # KLDivergenceTask
env = SwarmEnv(task='wasserstein', target_density=target)  # WassersteinTask
env = SwarmEnv(task='emd',         target_density=target)  # alias
```

---

## 🎨 Target Distribution Factory

[`swarm.utils.density.create_target`](../swarm/utils/density.py) provides
named distributions:

```python
from swarm.utils.density import create_target

shapes = ['gaussian', 'double_gaussian', 'ring', 'ring_gaussian',
          'corners', 'cross', 'checkerboard', 'stripes', 'uniform']

target = create_target('ring_gaussian', grid_resolution=32, total_particles=3000)
```

| Shape | Description |
|-------|-------------|
| `gaussian` | Single centred Gaussian blob |
| `double_gaussian` | Two Gaussians separated horizontally |
| `ring` | Hard-edged annulus |
| `ring_gaussian` | Smooth Gaussian-profiled ring (recommended for training) |
| `corners` | Four equal blobs at corners |
| `cross` | Symmetric cross |
| `checkerboard` | Alternating cell pattern |
| `stripes` | Horizontal (or vertical) stripes |
| `uniform` | Flat / maximum-entropy distribution |

Load a custom shape from an image:

```python
from swarm.utils.density import image_to_density

target = image_to_density('logo.png', grid_resolution=32, total_particles=3000)
```

---

## 📐 Utility Functions

```python
from swarm.utils.density import (
    kl_divergence,          # D_KL(P ‖ Q) for 2-D grids
    symmetric_kl,           # (fwd + rev) / 2
    wasserstein_distance_2d # Sliced Wasserstein Distance
)

kl  = kl_divergence(current, target)          # float ≥ 0
skl = symmetric_kl(current, target)           # float ≥ 0
swd = wasserstein_distance_2d(current, target) # float ≥ 0
```

---

## 🚀 Training

### Quick start

```bash
python examples/week4_distribution_matching.py \
    --task kl --target ring --steps 200000
```

### Long run (recommended)

```bash
python examples/week4_distribution_matching.py \
    --task kl --target ring_gaussian \
    --steps 1000000 --n-envs 8 \
    --save-gif
```

### From image

```bash
python examples/week4_distribution_matching.py \
    --image assets/smiley.png --task wasserstein \
    --steps 500000
```

### Python API

```python
from swarm import SwarmEnv, KLDivergenceTask, Trainer, TrainingConfig
from swarm.utils.density import create_target

target = create_target('double_gaussian', grid_resolution=32, total_particles=3000)

env_fn = lambda: SwarmEnv(
    task=KLDivergenceTask(target=target, success_threshold=0.02),
    num_particles=3000,
    temperature=1.0,
    num_basis=16,
    physics_steps_per_action=10,
    max_steps=150,
    learnable_max_force=True,
    action_smoothing=0.1,
)

config = TrainingConfig(
    total_timesteps=500_000,
    algorithm='PPO',
    lr_schedule='cosine',
    n_envs=4,
    tensorboard=True,
    visualize=True,
    experiment_name='week4_kl_double_gaussian',
)

trainer = Trainer(env_fn=env_fn, config=config)
trainer.train()
trainer.save('week4_model')
```

---

## 📊 Expected Results

| Task | Target | Steps | Expected Final KL |
|------|--------|-------|-------------------|
| KL | Gaussian | 200k | < 0.05 |
| KL | Ring | 500k | < 0.10 |
| KL | Double Gaussian | 500k | < 0.08 |
| Wasserstein | Ring | 500k | < 0.010 |

---

## 🧪 Verification Checklist

- [ ] KL divergence decreases monotonically (smoothed)  
- [ ] SWD decreases monotonically (smoothed)  
- [ ] Agent learns to concentrate at the correct spatial location  
- [ ] Ring task: density peaks at radius ≈ 35 % of grid, not at centre  
- [ ] Double Gaussian: two distinct density blobs appear  
- [ ] Episode success (early termination) is triggered  
- [ ] TensorBoard shows reward increasing and KL decreasing  

---

## 🔗 References

- **KL Divergence (forward/reverse)**: Bishop (2006) *Pattern Recognition and ML*, §1.6  
- **Wasserstein / EMD**: Villani (2009) *Optimal Transport: Old and New*  
- **Sliced Wasserstein Distance**: Rabin et al. (2011), NeurIPS  
- **Reinforcement learning + density matching**: Vicsek model, Active Matter RL  
