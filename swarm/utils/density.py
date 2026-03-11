"""
Density Grid Utilities

Functions for creating and manipulating particle density grids.
Consolidated from python/image_to_density.py.

Week 4 additions:
  - create_target()          unified target distribution factory
  - kl_divergence()          D_KL(P || Q) for discrete 2-D grids
  - symmetric_kl()           (D_KL(P||Q) + D_KL(Q||P)) / 2
  - wasserstein_distance_2d() Sliced Wasserstein distance
"""

import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def image_to_density(
    source: Union[str, Path, np.ndarray],
    grid_resolution: int = 32,
    total_particles: int = 5000,
    invert: bool = False,
    threshold: float = 0.0,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """
    Convert image to density grid for RL target matching.
    
    Args:
        source: Image path or numpy array
        grid_resolution: Output grid size (NxN)
        total_particles: Total particles to distribute
        invert: If True, dark regions have high density
        threshold: Minimum brightness threshold (0-1)
        blur_sigma: Gaussian blur sigma
    
    Returns:
        Density grid as float32 array
    """
    # Load image
    if isinstance(source, (str, Path)):
        if not PIL_AVAILABLE:
            raise ImportError("PIL required. Install: pip install Pillow")
        
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        img = Image.open(path)
        if img.mode != 'L':
            img = img.convert('L')
        image = np.array(img)
    else:
        image = np.array(source)
    
    # Ensure 2D grayscale
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    # Resize
    if PIL_AVAILABLE:
        img_pil = Image.fromarray(image.astype(np.uint8))
        img_resized = img_pil.resize(
            (grid_resolution, grid_resolution), 
            Image.Resampling.BILINEAR
        )
        density = np.array(img_resized, dtype=np.float32)
    else:
        # Simple resize
        from scipy.ndimage import zoom
        zoom_factor = grid_resolution / image.shape[0]
        density = zoom(image.astype(np.float32), zoom_factor)
    
    # Normalize to [0, 1]
    density = density / 255.0
    
    # Invert if requested
    if invert:
        density = 1.0 - density
    
    # Apply threshold
    density = np.where(density > threshold, density - threshold, 0)
    
    # Blur if requested
    if blur_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            density = gaussian_filter(density, sigma=blur_sigma)
        except ImportError:
            pass
    
    # Normalize to total particles
    if density.sum() > 0:
        density = density * (total_particles / density.sum())
    else:
        density = np.ones_like(density) * (total_particles / grid_resolution**2)
    
    return density.astype(np.float32)


def create_pattern(
    pattern_type: str,
    grid_resolution: int = 32,
    total_particles: int = 5000,
    **kwargs
) -> np.ndarray:
    """
    Create predefined target pattern.
    
    Args:
        pattern_type: Pattern name ('center', 'ring', 'corners', 'cross',
                     'stripes', 'checkerboard', 'spiral', 'uniform', 'gaussian')
        grid_resolution: Grid size
        total_particles: Total particles
        **kwargs: Pattern-specific parameters
    
    Returns:
        Density grid as float32 array
    """
    center = grid_resolution / 2
    y, x = np.ogrid[:grid_resolution, :grid_resolution]
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    
    if pattern_type == 'center':
        radius = kwargs.get('radius', grid_resolution * 0.2)
        pattern = (dist <= radius).astype(np.float32)
    
    elif pattern_type == 'ring':
        radius_outer = kwargs.get('radius_outer', grid_resolution * 0.4)
        radius_inner = kwargs.get('radius_inner', grid_resolution * 0.3)
        pattern = ((dist >= radius_inner) & (dist <= radius_outer)).astype(np.float32)
    
    elif pattern_type == 'gaussian':
        sigma = kwargs.get('sigma', grid_resolution * 0.15)
        pattern = np.exp(-dist**2 / (2 * sigma**2))
    
    elif pattern_type in ('corners', 'corner'):
        corner_size = kwargs.get('corner_size', grid_resolution // 4)
        pattern = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        pattern[:corner_size, :corner_size] = 1
        pattern[:corner_size, -corner_size:] = 1
        pattern[-corner_size:, :corner_size] = 1
        pattern[-corner_size:, -corner_size:] = 1
    
    elif pattern_type == 'cross':
        width = kwargs.get('width', grid_resolution // 4)
        pattern = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        half_w = width // 2
        c = grid_resolution // 2
        pattern[c - half_w:c + half_w, :] = 1
        pattern[:, c - half_w:c + half_w] = 1
    
    elif pattern_type == 'stripes':
        num_stripes = kwargs.get('num_stripes', 4)
        horizontal = kwargs.get('horizontal', True)
        stripe_width = grid_resolution // (num_stripes * 2)
        pattern = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        for i in range(num_stripes):
            start = i * 2 * stripe_width
            end = start + stripe_width
            if horizontal:
                pattern[start:end, :] = 1
            else:
                pattern[:, start:end] = 1
    
    elif pattern_type == 'checkerboard':
        cell_size = kwargs.get('cell_size', grid_resolution // 4)
        pattern = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                if ((i // cell_size) + (j // cell_size)) % 2 == 0:
                    pattern[i, j] = 1
    
    elif pattern_type == 'spiral':
        pattern = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        for theta in np.linspace(0, 6 * np.pi, 1000):
            r = theta * grid_resolution / (6 * np.pi * 2)
            xi = int(center + r * np.cos(theta))
            yi = int(center + r * np.sin(theta))
            if 0 <= xi < grid_resolution and 0 <= yi < grid_resolution:
                pattern[yi, xi] = 1
        # Dilate
        try:
            from scipy.ndimage import binary_dilation
            pattern = binary_dilation(pattern, iterations=2).astype(np.float32)
        except ImportError:
            pass
    
    elif pattern_type == 'uniform':
        pattern = np.ones((grid_resolution, grid_resolution), dtype=np.float32)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern_type}")
    
    # Normalize to total particles
    if pattern.sum() > 0:
        pattern = pattern * (total_particles / pattern.sum())
    
    return pattern.astype(np.float32)


def compute_error(
    current: np.ndarray,
    target: np.ndarray,
    metric: str = 'mse'
) -> float:
    """
    Compute error between density grids.
    
    Args:
        current: Current density grid
        target: Target density grid
        metric: Error metric ('mse', 'mae', 'correlation', 'kl')
    
    Returns:
        Error value (lower is better for mse/mae, higher for correlation)
    """
    # Normalize to distributions
    curr = current / (current.sum() + 1e-8)
    targ = target / (target.sum() + 1e-8)
    
    if metric == 'mse':
        return float(np.mean((curr - targ)**2))
    
    elif metric == 'mae':
        return float(np.mean(np.abs(curr - targ)))
    
    elif metric == 'correlation':
        return float(np.corrcoef(curr.flatten(), targ.flatten())[0, 1])
    
    elif metric in ('kl', 'kl_divergence'):
        eps = 1e-8
        kl = np.sum(targ * np.log((targ + eps) / (curr + eps)))
        return float(kl)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def visualize_comparison(
    current: np.ndarray,
    target: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Visualize current vs target density.
    
    Args:
        current: Current density grid
        target: Target density grid
        save_path: Optional path to save figure
        show: Whether to display figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Current
    im1 = axes[0].imshow(current, cmap='hot', origin='lower')
    axes[0].set_title('Current Density')
    plt.colorbar(im1, ax=axes[0])
    
    # Target
    im2 = axes[1].imshow(target, cmap='hot', origin='lower')
    axes[1].set_title('Target Density')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    diff = current - target
    max_abs = max(abs(diff.min()), abs(diff.max()))
    im3 = axes[2].imshow(diff, cmap='coolwarm', origin='lower',
                         vmin=-max_abs, vmax=max_abs)
    axes[2].set_title('Difference')
    plt.colorbar(im3, ax=axes[2])
    
    # Metrics
    mse = compute_error(current, target, 'mse')
    corr = compute_error(current, target, 'correlation')
    fig.suptitle(f'MSE: {mse:.6f}, Correlation: {corr:.4f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


# ---------------------------------------------------------------------------
# Week 4: KL Divergence & Wasserstein utilities
# ---------------------------------------------------------------------------

def create_target(
    shape: str,
    grid_resolution: int = 32,
    total_particles: int = 5000,
    **kwargs,
) -> np.ndarray:
    """
    Create a normalised 2-D target density for distribution-matching tasks.

    This is the canonical Week 4 factory — it wraps ``create_pattern`` and
    adds shapes that are meaningful as probability distributions.

    Args:
        shape: One of:
            'gaussian'         — single centred Gaussian (default σ = 15 % grid)
            'double_gaussian'  — two equal Gaussians side-by-side
            'ring'             — thin annulus
            'ring_gaussian'    — Gaussian-profiled ring (smooth edges)
            'center'           — filled circle at centre
            'corners'          — four equal corner blobs
            'cross'            — symmetric cross
            'checkerboard'     — equal cells alternating on/off
            'stripes'          — horizontal stripes
            'uniform'          — uniform (flat) distribution
        grid_resolution: Grid side length N (output is N × N).
        total_particles:  Total particle count the density sums to.
        **kwargs: Shape-specific overrides (see table below).

    Keyword args per shape
    ----------------------
    gaussian:         sigma  (default 0.15 * grid_resolution)
    double_gaussian:  sigma (default 0.10 * grid_resolution),
                      separation (default 0.30 * grid_resolution)
    ring:             radius_outer (0.40 * N), radius_inner (0.28 * N)
    ring_gaussian:    radius (0.35 * N), sigma_r (0.04 * N)
    center:           radius (0.20 * N)
    corners:          corner_size (N // 4)
    cross:            width (N // 4)
    checkerboard:     cell_size (N // 4)
    stripes:          num_stripes (4), horizontal (True)
    uniform:          —

    Returns:
        float32 array of shape (grid_resolution, grid_resolution) summing to
        approximately *total_particles*.

    Example:
        >>> target = create_target('ring', grid_resolution=32)
        >>> target.sum()   # ≈ 5000
    """
    N = grid_resolution
    center = N / 2.0
    y, x = np.ogrid[:N, :N]
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    if shape == 'gaussian':
        sigma = kwargs.get('sigma', N * 0.15)
        pattern = np.exp(-dist ** 2 / (2 * sigma ** 2)).astype(np.float32)

    elif shape == 'double_gaussian':
        sigma = kwargs.get('sigma', N * 0.10)
        sep = kwargs.get('separation', N * 0.30)
        cx1, cy1 = center - sep / 2, center
        cx2, cy2 = center + sep / 2, center
        d1 = np.sqrt((x - cx1) ** 2 + (y - cy1) ** 2)
        d2 = np.sqrt((x - cx2) ** 2 + (y - cy2) ** 2)
        pattern = (
            np.exp(-d1 ** 2 / (2 * sigma ** 2))
            + np.exp(-d2 ** 2 / (2 * sigma ** 2))
        ).astype(np.float32)

    elif shape == 'ring':
        r_outer = kwargs.get('radius_outer', N * 0.40)
        r_inner = kwargs.get('radius_inner', N * 0.28)
        pattern = ((dist >= r_inner) & (dist <= r_outer)).astype(np.float32)

    elif shape == 'ring_gaussian':
        radius = kwargs.get('radius', N * 0.35)
        sigma_r = kwargs.get('sigma_r', N * 0.04)
        pattern = np.exp(-((dist - radius) ** 2) / (2 * sigma_r ** 2)).astype(
            np.float32
        )

    else:
        # Delegate to the existing create_pattern for the remaining shapes
        return create_pattern(shape, grid_resolution=N, total_particles=total_particles, **kwargs)

    # Normalise to total_particles
    s = pattern.sum()
    if s > 0:
        pattern = pattern * (total_particles / s)
    else:
        pattern = np.ones((N, N), dtype=np.float32) * (total_particles / N ** 2)

    return pattern.astype(np.float32)


def kl_divergence(
    current: np.ndarray,
    target: np.ndarray,
    epsilon: float = 1e-8,
    direction: str = 'forward',
) -> float:
    """
    Compute discrete KL Divergence between two 2-D density grids.

    Both grids are normalised to probability distributions before comparison.

    Args:
        current:   Current swarm density  (P).
        target:    Target density         (Q).
        epsilon:   Small smoothing term to avoid log(0). Default 1e-8.
        direction: ``'forward'``  → D_KL(P || Q)  (standard, shape-seeking)
                   ``'reverse'``  → D_KL(Q || P)  (mode-covering)
                   ``'symmetric'``→ (D_KL(P||Q) + D_KL(Q||P)) / 2

    Returns:
        KL divergence value  ≥ 0  (0 means identical distributions).

    Math:
        D_KL(P || Q) = Σ_x  P(x) · log( P(x) / Q(x) )
    """
    eps = epsilon
    P = current / (current.sum() + eps)
    Q = target / (target.sum() + eps)
    P = np.clip(P, eps, None)
    Q = np.clip(Q, eps, None)

    if direction == 'forward':
        return float(max(np.sum(P * np.log(P / Q)), 0.0))
    elif direction == 'reverse':
        return float(max(np.sum(Q * np.log(Q / P)), 0.0))
    elif direction == 'symmetric':
        fwd = float(max(np.sum(P * np.log(P / Q)), 0.0))
        rev = float(max(np.sum(Q * np.log(Q / P)), 0.0))
        return (fwd + rev) / 2.0
    else:
        raise ValueError(f"Unknown direction: {direction!r}. Use 'forward', 'reverse', or 'symmetric'.")


def symmetric_kl(
    current: np.ndarray,
    target: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """
    Symmetric KL Divergence: (D_KL(P||Q) + D_KL(Q||P)) / 2.

    Convenience wrapper around :func:`kl_divergence`.
    """
    return kl_divergence(current, target, epsilon=epsilon, direction='symmetric')


def wasserstein_distance_2d(
    current: np.ndarray,
    target: np.ndarray,
    num_projections: int = 64,
    seed: int = 42,
) -> float:
    """
    Sliced Wasserstein Distance (SWD) between two 2-D density grids.

    The SWD approximates the true Wasserstein-1 distance by averaging the
    exact 1-D Wasserstein distance over *num_projections* random unit-vector
    projections of the 2-D domain.  Complexity is O(num_projections · N log N)
    where N is the number of grid cells.

    Requires ``scipy >= 1.0``.  Falls back to :func:`kl_divergence` if scipy
    is not installed.

    Args:
        current:         Current density grid.
        target:          Target density grid (same shape as *current*).
        num_projections: Number of 1-D projections. More = more accurate but
                         slower. 64 is a good default for 32×32 grids.
        seed:            Random seed for reproducible projections.

    Returns:
        Non-negative float.  0 means identical distributions.

    References:
        Rabin et al. (2011) "Wasserstein Barycenter and its Application to
        Texture Mixing."
    """
    try:
        from scipy.stats import wasserstein_distance as w1d
    except ImportError:
        # Graceful fallback
        return kl_divergence(current, target)

    eps = 1e-8
    h, w = current.shape

    # Coordinate grid normalised to [0, 1] × [0, 1]
    y_c, x_c = np.mgrid[0:h, 0:w]
    x_c = x_c.astype(np.float32) / max(w - 1, 1)
    y_c = y_c.astype(np.float32) / max(h - 1, 1)
    coords = np.stack([x_c.ravel(), y_c.ravel()], axis=1)  # (N, 2)

    P = current.ravel().astype(np.float64)
    Q = target.ravel().astype(np.float64)
    P = np.clip(P / (P.sum() + eps), 0, None)
    Q = np.clip(Q / (Q.sum() + eps), 0, None)

    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, np.pi, num_projections)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (K, 2)

    distances = [
        w1d(coords @ d, coords @ d, u_weights=P, v_weights=Q)
        for d in dirs
    ]
    return float(np.mean(distances))
