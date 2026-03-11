"""Utility subpackage."""
from swarm.utils.density import (
    image_to_density,
    create_pattern,
    compute_error,
    visualize_comparison,
    # Week 4
    create_target,
    kl_divergence,
    symmetric_kl,
    wasserstein_distance_2d,
)
from swarm.utils.visualization import (
    render_density,
    render_particles,
    render_combined,
    create_gif,
    create_matplotlib_animation,
    save_snapshot,
)

__all__ = [
    # Density utilities
    'image_to_density',
    'create_pattern',
    'compute_error',
    'visualize_comparison',
    # Week 4: distribution matching
    'create_target',
    'kl_divergence',
    'symmetric_kl',
    'wasserstein_distance_2d',
    # Visualization utilities
    'render_density',
    'render_particles',
    'render_combined',
    'create_gif',
    'create_matplotlib_animation',
    'save_snapshot',
]
