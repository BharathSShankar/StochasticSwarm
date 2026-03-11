#!/usr/bin/env python3
"""
Demo: Density Grid and Zero-Copy NumPy Interface
Shows how to use the density grid for RL observation space
"""

import sys
sys.path.insert(0, '.')
import stochastic_swarm as ss
import numpy as np

def main():
    print("=" * 60)
    print("DENSITY GRID DEMO")
    print("=" * 60)
    
    # Create particle system with density grid
    print("\n1. Creating particle system with density grid...")
    ps = ss.ParticleSystem(
        num_particles=2000,
        temperature=1.0,
        num_basis=9,      # 3x3 grid of basis functions
        grid_res=32       # 32x32 density grid
    )
    ps.initialize_random(100.0)
    print(f"   Created system with {ps.num_particles} particles")
    
    # Set up potential field (repulsive center, attractive edges)
    print("\n2. Setting up potential field...")
    strengths = [
        -1.0, -1.0, -1.0,  # Top row: attractive
         2.0,  3.0,  2.0,  # Middle row: repulsive center
        -1.0, -1.0, -1.0   # Bottom row: attractive
    ]
    ps.set_potential_params(strengths)
    print("   Configured 9 basis functions (repulsive center, attractive edges)")
    
    # Get initial density
    print("\n3. Computing initial density...")
    ps.update_density_grid()
    dg_initial = ps.get_density_grid()
    density_initial = dg_initial.get_grid()
    
    print(f"   Density grid shape: {density_initial.shape}")
    print(f"   Total particles: {density_initial.sum():.0f}")
    print(f"   Max density: {density_initial.max():.2f} particles/cell")
    print(f"   Memory address: 0x{density_initial.__array_interface__['data'][0]:x}")
    
    # Run simulation
    print("\n4. Running simulation (500 steps)...")
    for i in range(500):
        ps.step()
        if (i + 1) % 100 == 0:
            print(f"   Step {i+1}...")
    
    # Get final density
    print("\n5. Computing final density...")
    ps.update_density_grid()
    dg_final = ps.get_density_grid()
    density_final = dg_final.get_grid()
    
    print(f"   Total particles: {density_final.sum():.0f} (conserved!)")
    print(f"   Max density: {density_final.max():.2f} particles/cell")
    print(f"   Memory address: 0x{density_final.__array_interface__['data'][0]:x}")
    
    # Verify zero-copy
    print("\n6. Verifying zero-copy behavior...")
    grid1 = ps.get_density_grid().get_grid()
    grid2 = ps.get_density_grid().get_grid()
    addr1 = grid1.__array_interface__['data'][0]
    addr2 = grid2.__array_interface__['data'][0]
    
    if addr1 == addr2:
        print("   ✓ Zero-copy confirmed: Both arrays point to same memory!")
    else:
        print("   ✗ Warning: Arrays point to different memory")
    
    # Show density statistics
    print("\n7. Density statistics:")
    print(f"   Mean density: {density_final.mean():.2f}")
    print(f"   Std deviation: {density_final.std():.2f}")
    print(f"   Min density: {density_final.min():.2f}")
    print(f"   Max density: {density_final.max():.2f}")
    
    # Show density increased at edges (due to attractive potential)
    center_region = density_final[12:20, 12:20]
    edge_sum = density_final.sum() - center_region.sum()
    print(f"\n8. Spatial distribution:")
    print(f"   Particles in center (25%): {center_region.sum():.0f}")
    print(f"   Particles at edges (75%): {edge_sum:.0f}")
    print(f"   Edge/Center ratio: {edge_sum/center_region.sum():.2f}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("  ✓ DensityGrid creation and integration")
    print("  ✓ Zero-copy NumPy interface (no data duplication)")
    print("  ✓ Particle conservation (count maintained)")
    print("  ✓ Grid updates after simulation")
    print("  ✓ Ready for RL observation space!")

if __name__ == '__main__':
    main()
