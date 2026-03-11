#!/usr/bin/env python3
"""
Visualization script for StochasticSwarm particle simulation
Reads CSV files from output/<temperature>/ directory and creates animations/plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse


def get_temp_dir(temperature):
    """Get the directory name for a given temperature"""
    return f"T_{temperature:.2f}"


def list_available_temperatures():
    """List all available temperature runs"""
    output_dir = Path("output")
    if not output_dir.exists():
        return []
    
    temps = []
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("T_"):
            try:
                temp = float(d.name[2:])
                temps.append(temp)
            except ValueError:
                continue
    
    return sorted(temps)


def load_frame(frame_number, temperature):
    """Load particle data from a CSV file"""
    temp_dir = get_temp_dir(temperature)
    filepath = Path(f"output/{temp_dir}/frame_{frame_number}.csv")
    if not filepath.exists():
        return None
    
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    if data.size == 0:
        return None
    
    return {
        'x': data[:, 0],
        'y': data[:, 1],
        'vx': data[:, 2],
        'vy': data[:, 3]
    }


def find_available_frames(temperature):
    """Find all available frame files for a given temperature"""
    temp_dir = get_temp_dir(temperature)
    output_dir = Path(f"output/{temp_dir}")
    if not output_dir.exists():
        return []
    
    frame_files = sorted(output_dir.glob("frame_*.csv"))
    frame_numbers = []
    for f in frame_files:
        try:
            num = int(f.stem.split('_')[1])
            frame_numbers.append(num)
        except (ValueError, IndexError):
            continue
    
    return sorted(frame_numbers)


def plot_single_frame(frame_number, temperature, domain_size=100, show_velocities=False):
    """Plot a single frame of particle positions"""
    data = load_frame(frame_number, temperature)
    if data is None:
        print(f"Frame {frame_number} not found for T={temperature}!")
        return
    
    temp_dir = get_temp_dir(temperature)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot particles
    scatter = ax.scatter(data['x'], data['y'], s=1, alpha=0.6, c='blue')
    
    # Optionally plot velocity arrows (for small particle counts)
    if show_velocities and len(data['x']) < 500:
        ax.quiver(data['x'], data['y'], data['vx'], data['vy'],
                 alpha=0.3, scale=50, width=0.003, color='red')
    
    ax.set_xlim(0, domain_size)
    ax.set_ylim(0, domain_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Particle System - T={temperature}, Frame {frame_number}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f'output/{temp_dir}/snapshot_frame_{frame_number}.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")
    plt.show()


def create_animation(temperature, domain_size=100, fps=10):
    """Create an animated visualization of all frames"""
    frames = find_available_frames(temperature)
    if not frames:
        print(f"No frames found for T={temperature}!")
        return
    
    temp_dir = get_temp_dir(temperature)
    print(f"Found {len(frames)} frames for T={temperature}: {frames}")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Load first frame to initialize
    first_data = load_frame(frames[0], temperature)
    if first_data is None:
        print(f"Could not load first frame!")
        return
    scatter = ax.scatter(first_data['x'], first_data['y'], s=1, alpha=0.6, c='blue')
    
    ax.set_xlim(0, domain_size)
    ax.set_ylim(0, domain_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes,
                   ha='center', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    def update(frame_idx):
        """Update function for animation"""
        frame_num = frames[frame_idx]
        data = load_frame(frame_num, temperature)
        if data is None:
            return scatter, title
        
        # Update scatter plot
        scatter.set_offsets(np.c_[data['x'], data['y']])
        title.set_text(f'T={temperature} Frame {frame_num} - {len(data["x"])} particles')
        
        return scatter, title
    
    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000//fps, blit=True, repeat=True)
    
    # Save animation
    output_file = f'output/{temp_dir}/particle_animation.gif'
    print(f"Creating animation... (this may take a while)")
    anim.save(output_file, writer='pillow', fps=fps, dpi=100)
    print(f"Animation saved to {output_file}")
    
    plt.show()


def plot_trajectory_sample(temperature, num_particles=10, domain_size=100):
    """Plot trajectories of a subset of particles"""
    frames = find_available_frames(temperature)
    if not frames:
        print(f"No frames found for T={temperature}!")
        return
    
    temp_dir = get_temp_dir(temperature)
    
    # Collect positions over time
    trajectories_x = [[] for _ in range(num_particles)]
    trajectories_y = [[] for _ in range(num_particles)]
    
    for frame_num in frames:
        data = load_frame(frame_num, temperature)
        if data is None:
            continue
        
        for i in range(min(num_particles, len(data['x']))):
            trajectories_x[i].append(data['x'][i])
            trajectories_y[i].append(data['y'][i])
    
    # Plot trajectories
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i in range(num_particles):
        if len(trajectories_x[i]) > 0:
            ax.plot(trajectories_x[i], trajectories_y[i], alpha=0.6, linewidth=0.8)
            # Mark start and end
            ax.plot(trajectories_x[i][0], trajectories_y[i][0], 'go', markersize=8, alpha=0.7)
            ax.plot(trajectories_x[i][-1], trajectories_y[i][-1], 'ro', markersize=8, alpha=0.7)
    
    ax.set_xlim(0, domain_size)
    ax.set_ylim(0, domain_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Particle Trajectories T={temperature} (n={num_particles})\nGreen=Start, Red=End')
    ax.grid(True, alpha=0.3)
    ax.legend(['Trajectories', 'Start', 'End'], loc='upper right')
    
    plt.tight_layout()
    output_file = f'output/{temp_dir}/trajectories.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")
    plt.show()


def plot_velocity_distribution(frame_number, temperature):
    """Plot velocity distribution to verify thermal equilibrium"""
    data = load_frame(frame_number, temperature)
    if data is None:
        print(f"Frame {frame_number} not found for T={temperature}!")
        return
    
    temp_dir = get_temp_dir(temperature)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Velocity components
    axes[0].hist(data['vx'], bins=50, alpha=0.7, density=True, label='vx')
    axes[0].set_xlabel('Velocity X')
    axes[0].set_ylabel('Probability Density')
    axes[0].set_title('X Velocity Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(data['vy'], bins=50, alpha=0.7, density=True, label='vy', color='orange')
    axes[1].set_xlabel('Velocity Y')
    axes[1].set_title('Y Velocity Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Speed distribution (should be Rayleigh for 2D)
    speeds = np.sqrt(data['vx']**2 + data['vy']**2)
    axes[2].hist(speeds, bins=50, alpha=0.7, density=True, color='green')
    axes[2].set_xlabel('Speed |v|')
    axes[2].set_title('Speed Distribution')
    axes[2].grid(True, alpha=0.3)
    
    # Add statistics
    vx_mean, vx_std = np.mean(data['vx']), np.std(data['vx'])
    vy_mean, vy_std = np.mean(data['vy']), np.std(data['vy'])
    speed_mean = np.mean(speeds)
    
    fig.suptitle(f'Velocity Statistics T={temperature} (Frame {frame_number})\n'
                f'⟨vx⟩={vx_mean:.3f}±{vx_std:.3f}, '
                f'⟨vy⟩={vy_mean:.3f}±{vy_std:.3f}, '
                f'⟨|v|⟩={speed_mean:.3f}',
                fontsize=12)
    
    plt.tight_layout()
    output_file = f'output/{temp_dir}/velocity_distribution_frame_{frame_number}.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize StochasticSwarm particles')
    parser.add_argument('-T', '--temperature', type=float, default=None,
                       help='Temperature to visualize (e.g., 0.1, 1.0, 10.0)')
    parser.add_argument('--list-temps', action='store_true',
                       help='List available temperature runs')
    parser.add_argument('--mode', choices=['single', 'animate', 'trajectories', 'velocities'],
                       default='single', help='Visualization mode')
    parser.add_argument('--frame', type=int, default=0, help='Frame number for single frame plot')
    parser.add_argument('--domain', type=float, default=100.0, help='Domain size')
    parser.add_argument('--fps', type=int, default=10, help='Animation frame rate')
    parser.add_argument('--num-particles', type=int, default=10, help='Number of particle trajectories to plot')
    
    args = parser.parse_args()
    
    # List available temperatures
    if args.list_temps:
        temps = list_available_temperatures()
        if not temps:
            print("No temperature runs found in output/")
        else:
            print("Available temperature runs:")
            for t in temps:
                print(f"  T = {t}")
        return
    
    # Check if temperature is provided
    if args.temperature is None:
        temps = list_available_temperatures()
        if not temps:
            print("No temperature runs found. Run the simulation first!")
            print("  ./build/StochasticSwarm -T <temperature>")
            return
        else:
            print("Please specify a temperature with -T <value>")
            print("Available temperatures:")
            for t in temps:
                print(f"  -T {t}")
            return
    
    # Run the appropriate visualization
    if args.mode == 'single':
        plot_single_frame(args.frame, args.temperature, args.domain)
    elif args.mode == 'animate':
        create_animation(args.temperature, args.domain, args.fps)
    elif args.mode == 'trajectories':
        plot_trajectory_sample(args.temperature, args.num_particles, args.domain)
    elif args.mode == 'velocities':
        plot_velocity_distribution(args.frame, args.temperature)
    else:
        print("Invalid mode!")


if __name__ == '__main__':
    main()
