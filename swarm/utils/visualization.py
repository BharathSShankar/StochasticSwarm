"""
<br/>Visualization and GIF Generation Utilities

Provides easy-to-use functions for visualizing swarm environments
and generating GIFs during training or inference.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
from io import BytesIO


def render_density(
    density: np.ndarray,
    title: str = "Density Map",
    cmap: str = 'hot',
    figsize: Tuple[int, int] = (6, 6),
    show_colorbar: bool = True,
    ax: Optional[Axes] = None,
) -> np.ndarray:
    """
    Render a density grid as an image array.
    
    Args:
        density: 2D density grid
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
        show_colorbar: Whether to show colorbar
        ax: Optional existing axes to plot on
        
    Returns:
        RGB image array (H, W, 3)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_figure = True
    else:
        fig = ax.figure
        own_figure = False
    
    im = ax.imshow(density, cmap=cmap, origin='lower')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if show_colorbar and own_figure:
        plt.colorbar(im, ax=ax, label='Density')
    
    # Convert to RGB array
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    
    from PIL import Image
    img = np.array(Image.open(buf).convert('RGB'))
    
    if own_figure:
        plt.close(fig)  # type: ignore
    
    return img


def render_particles(
    positions: np.ndarray,
    domain_size: float = 100.0,
    title: str = "Particle Positions",
    figsize: Tuple[int, int] = (6, 6),
    alpha: float = 0.5,
    s: float = 1.0,
    color: str = 'blue',
    ax: Optional[Axes] = None,
) -> np.ndarray:
    """
    Render particle positions as an image array.
    
    Args:
        positions: Nx2 array of particle positions
        domain_size: Physical domain size
        title: Plot title
        figsize: Figure size
        alpha: Point transparency
        s: Point size
        color: Point color
        ax: Optional existing axes to plot on
        
    Returns:
        RGB image array (H, W, 3)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_figure = True
    else:
        fig = ax.figure
        own_figure = False
    
    ax.scatter(positions[:, 0], positions[:, 1], s=s, alpha=alpha, c=color)
    ax.set_xlim(0, domain_size)
    ax.set_ylim(0, domain_size)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    # Convert to RGB array
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    
    from PIL import Image
    img = np.array(Image.open(buf).convert('RGB'))
    
    if own_figure:
        plt.close(fig)  # type: ignore
    
    return img


def render_combined(
    density: np.ndarray,
    positions: Optional[np.ndarray] = None,
    domain_size: float = 100.0,
    title: str = "Swarm State",
    figsize: Tuple[int, int] = (12, 5),
) -> np.ndarray:
    """
    Render both density and particle positions side-by-side.
    
    Args:
        density: 2D density grid
        positions: Optional Nx2 array of particle positions
        domain_size: Physical domain size
        title: Overall title
        figsize: Figure size
        
    Returns:
        RGB image array (H, W, 3)
    """
    if positions is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Density
        im = ax1.imshow(density, cmap='hot', origin='lower')
        ax1.set_title('Density Map')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im, ax=ax1, label='Density')
        
        # Particles
        ax2.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.5, c='blue')
        ax2.set_xlim(0, domain_size)
        ax2.set_ylim(0, domain_size)
        ax2.set_aspect('equal')
        ax2.set_title(f'Particles (N={len(positions)})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14)
    else:
        fig, ax1 = plt.subplots(figsize=(6, 6))
        im = ax1.imshow(density, cmap='hot', origin='lower')
        ax1.set_title(title)
        plt.colorbar(im, ax=ax1, label='Density')
    
    # Convert to RGB array
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    
    from PIL import Image
    img = np.array(Image.open(buf).convert('RGB'))
    plt.close(fig)
    
    return img


def create_gif(
    frames: List[Dict[str, Any]],
    output_path: Union[str, Path],
    fps: int = 10,
    mode: str = 'density',
    domain_size: float = 100.0,
    loop: int = 0,
    verbose: bool = True,
) -> str:
    """
    Create a GIF from recorded frames.
    
    Args:
        frames: List of frame dictionaries with 'density' and optionally 'positions', 'timestep'
        output_path: Path to save GIF
        fps: Frames per second
        mode: Visualization mode ('density', 'particles', 'combined')
        domain_size: Physical domain size
        loop: Number of loops (0 = infinite)
        verbose: Print progress
        
    Returns:
        Path to saved GIF
    """
    from PIL import Image
    
    if not frames:
        raise ValueError("No frames to create GIF")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    images = []
    
    for i, frame in enumerate(frames):
        if verbose and i % 10 == 0:
            print(f"  Rendering frame {i+1}/{len(frames)}...")
        
        density = frame.get('density')
        positions = frame.get('positions')
        timestep = frame.get('timestep', i)
        
        if density is None:
            continue
        
        if mode == 'density':
            img_array = render_density(
                density,
                title=f'Step {timestep}',
                figsize=(6, 6),
            )
        elif mode == 'particles' and positions is not None:
            img_array = render_particles(
                positions,
                domain_size=domain_size,
                title=f'Step {timestep} (N={len(positions)})',
                figsize=(6, 6),
            )
        elif mode == 'combined':
            img_array = render_combined(
                density,
                positions=positions,
                domain_size=domain_size,
                title=f'Step {timestep}',
                figsize=(12, 5),
            )
        else:
            # Fallback to density
            img_array = render_density(density, title=f'Step {timestep}')
        
        images.append(Image.fromarray(img_array))
    
    if not images:
        raise ValueError("No valid images created from frames")
    
    # Save GIF
    duration = int(1000 / fps)  # milliseconds per frame
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=False,
    )
    
    if verbose:
        print(f"  Saved GIF to {output_path} ({len(images)} frames, {fps} fps)")
    
    return str(output_path)


def create_matplotlib_animation(
    frames: List[Dict[str, Any]],
    output_path: Union[str, Path],
    fps: int = 10,
    mode: str = 'density',
    domain_size: float = 100.0,
    dpi: int = 100,
    verbose: bool = True,
) -> str:
    """
    Create animation using matplotlib's FuncAnimation (supports MP4, GIF).
    
    Args:
        frames: List of frame dictionaries
        output_path: Path to save animation
        fps: Frames per second
        mode: Visualization mode
        domain_size: Physical domain size
        dpi: DPI for saved animation
        verbose: Print progress
        
    Returns:
        Path to saved animation
    """
    if not frames:
        raise ValueError("No frames to create animation")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format from extension
    suffix = output_path.suffix.lower()
    if suffix == '.gif':
        writer = 'pillow'
    elif suffix in ['.mp4', '.avi']:
        writer = 'ffmpeg'
    else:
        writer = 'pillow'
    
    # Setup figure
    if mode == 'combined':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        axes: Tuple[Axes, ...] = (ax1, ax2)
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        axes = (ax,)
    
    def init():
        """Initialize animation."""
        for ax in axes:
            ax.clear()
        return axes
    
    def update(frame_idx):
        """Update function for animation."""
        frame = frames[frame_idx]
        density = frame.get('density')
        positions = frame.get('positions')
        timestep = frame.get('timestep', frame_idx)
        
        for ax in axes:
            ax.clear()
        
        if mode == 'density' or mode == 'combined':
            ax = axes[0]
            im = ax.imshow(density, cmap='hot', origin='lower')
            ax.set_title(f'Density - Step {timestep}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        
        if mode == 'particles' and positions is not None:
            ax = axes[0]
            ax.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.5, c='blue')
            ax.set_xlim(0, domain_size)
            ax.set_ylim(0, domain_size)
            ax.set_aspect('equal')
            ax.set_title(f'Particles - Step {timestep}')
            ax.grid(True, alpha=0.3)
        
        if mode == 'combined' and positions is not None:
            ax2 = axes[1]
            ax2.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.5, c='blue')
            ax2.set_xlim(0, domain_size)
            ax2.set_ylim(0, domain_size)
            ax2.set_aspect('equal')
            ax2.set_title(f'Particles (N={len(positions)})')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.grid(True, alpha=0.3)
        
        return axes
    
    if verbose:
        print(f"  Creating animation with {len(frames)} frames...")
    
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=1000//fps,
        blit=False,
        repeat=True,
    )
    
    anim.save(output_path, writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    
    if verbose:
        print(f"  Saved animation to {output_path}")
    
    return str(output_path)


def save_snapshot(
    density: np.ndarray,
    output_path: Union[str, Path],
    positions: Optional[np.ndarray] = None,
    domain_size: float = 100.0,
    title: str = "Snapshot",
    mode: str = 'combined',
    dpi: int = 150,
) -> str:
    """
    Save a single snapshot image.
    
    Args:
        density: 2D density grid
        output_path: Path to save image
        positions: Optional particle positions
        domain_size: Physical domain size
        title: Image title
        mode: Visualization mode
        dpi: DPI for saved image
        
    Returns:
        Path to saved image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if mode == 'combined' and positions is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        im = ax1.imshow(density, cmap='hot', origin='lower')
        ax1.set_title('Density Map')
        plt.colorbar(im, ax=ax1, label='Density')
        
        ax2.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.5, c='blue')
        ax2.set_xlim(0, domain_size)
        ax2.set_ylim(0, domain_size)
        ax2.set_aspect('equal')
        ax2.set_title(f'Particles (N={len(positions)})')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14)
    elif mode == 'particles' and positions is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.5, c='blue')
        ax.set_xlim(0, domain_size)
        ax.set_ylim(0, domain_size)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(density, cmap='hot', origin='lower')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Density')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return str(output_path)
