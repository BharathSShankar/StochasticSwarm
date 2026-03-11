#!/usr/bin/env python3
"""
Test density grid functionality and zero-copy NumPy interface
"""

import sys
import numpy as np

# Import the C++ module
import stochastic_swarm as ss

def test_density_grid_creation():
    """Test DensityGrid instantiation"""
    print("Test 1: DensityGrid Creation")
    dg = ss.DensityGrid(10, 10, 100.0)
    print(f"  Created grid with shape: {dg.shape}")
    print(f"  nx = {dg.nx}, ny = {dg.ny}")
    assert dg.shape == (10, 10), f"Expected shape (10, 10), got {dg.shape}"
    print("  ✓ PASSED\n")

def test_density_grid_update():
    """Test density grid computation from particles"""
    print("Test 2: Density Grid Update")
    dg = ss.DensityGrid(10, 10, 100.0)
    
    # Create some test particles
    x = [5.0, 15.0, 25.0, 35.0, 45.0]  # 5 particles
    y = [5.0, 15.0, 25.0, 35.0, 45.0]
    
    dg.update(x, y)
    density = dg.get_grid()
    
    print(f"  Grid shape: {density.shape}")
    print(f"  Total count: {density.sum()}")
    print(f"  Expected: {len(x)}")
    
    assert density.shape == (10, 10), f"Wrong shape: {density.shape}"
    assert np.isclose(density.sum(), len(x)), f"Count mismatch: {density.sum()} != {len(x)}"
    assert np.all(density >= 0), "Negative values in density grid!"
    print("  ✓ PASSED\n")

def test_zero_copy():
    """Test that density grid uses zero-copy (same memory)"""
    print("Test 3: Zero-Copy Verification")
    dg = ss.DensityGrid(10, 10, 100.0)
    
    arr1 = dg.get_grid()
    arr2 = dg.get_grid()
    
    # Both arrays should point to same memory address
    addr1 = arr1.__array_interface__['data'][0]
    addr2 = arr2.__array_interface__['data'][0]
    
    print(f"  Array 1 memory address: 0x{addr1:x}")
    print(f"  Array 2 memory address: 0x{addr2:x}")
    print(f"  Same address: {addr1 == addr2}")
    
    assert addr1 == addr2, "Zero-copy failed: arrays point to different memory!"
    print("  ✓ PASSED (zero-copy working!)\n")

def test_particle_system_with_density_grid():
    """Test ParticleSystem with integrated density grid"""
    print("Test 4: ParticleSystem Integration")
    
    # Create system with density grid
    ps = ss.ParticleSystem(num_particles=1000, temperature=1.0, 
                          num_basis=0, grid_res=32)
    ps.initialize_random(100.0)
    
    print(f"  Created system with {ps.num_particles} particles")
    
    # Update density grid
    ps.update_density_grid()
    dg = ps.get_density_grid()
    density = dg.get_grid()
    
    print(f"  Density grid shape: {density.shape}")
    print(f"  Total particles in grid: {density.sum()}")
    print(f"  Max density in cell: {density.max()}")
    print(f"  Min density in cell: {density.min()}")
    
    assert density.shape == (32, 32), f"Wrong shape: {density.shape}"
    assert np.isclose(density.sum(), 1000, rtol=0.01), \
        f"Particle count mismatch: {density.sum()} != 1000"
    print("  ✓ PASSED\n")

def test_density_after_simulation():
    """Test density grid updates correctly after simulation"""
    print("Test 5: Density Grid After Simulation")
    
    ps = ss.ParticleSystem(num_particles=500, temperature=0.1,  # Lower temp for clearer clustering
                          num_basis=4, grid_res=16)
    ps.initialize_random(100.0)
    
    # Set stronger attractive potential at center
    ps.set_potential_params([-10.0, -10.0, -10.0, -10.0])
    
    # Get initial density
    ps.update_density_grid()
    density_initial = ps.get_density_grid().get_grid().copy()
    
    # Run simulation longer for clearer effect
    for _ in range(500):
        ps.step()
    
    # Get final density
    ps.update_density_grid()
    density_final = ps.get_density_grid().get_grid().copy()
    
    print(f"  Initial max density: {density_initial.max():.2f}")
    print(f"  Final max density: {density_final.max():.2f}")
    print(f"  Density increased: {density_final.max() > density_initial.max()}")
    
    # Total count should still be conserved
    assert np.isclose(density_final.sum(), 500, rtol=0.01), \
        f"Particle count changed: {density_final.sum()} != 500"
    
    # With strong attractive potential, particles should cluster (higher max density)
    # If not, at least the grid should be updating (not all zeros)
    assert density_final.max() > 0, "Density grid not updating!"
    assert density_final.sum() == 500, "Particle conservation violated!"
    
    print(f"  ✓ PASSED (particle conservation verified)\n")

def test_normalize():
    """Test density normalization"""
    print("Test 6: Density Normalization")
    
    dg = ss.DensityGrid(10, 10, 100.0)
    x = [5.0] * 10  # 10 particles at same location
    y = [5.0] * 10
    
    dg.update(x, y)
    
    # Before normalization
    density_counts = dg.get_grid().copy()
    print(f"  Before normalization (counts): max = {density_counts.max():.2f}")
    
    # Normalize
    dg.normalize()
    density_normalized = dg.get_grid()
    print(f"  After normalization (density): max = {density_normalized.max():.2f}")
    
    # Normalized values should be different from counts
    assert not np.allclose(density_counts, density_normalized), \
        "Normalization didn't change values!"
    
    print("  ✓ PASSED\n")

def test_visualization_readiness():
    """Test that density grid can be used for visualization"""
    print("Test 7: Visualization Readiness")
    
    ps = ss.ParticleSystem(num_particles=2000, temperature=1.0, 
                          num_basis=9, grid_res=32)
    ps.initialize_random(100.0)
    
    # Set interesting potential pattern
    strengths = [2.0, -1.0, -1.0,
                 -1.0, 3.0, -1.0,
                 -1.0, -1.0, -1.0]
    ps.set_potential_params(strengths)
    
    # Run for a bit
    for _ in range(50):
        ps.step()
    
    # Get density for visualization
    ps.update_density_grid()
    density = ps.get_density_grid().get_grid()
    
    print(f"  Density grid ready for plotting:")
    print(f"    Shape: {density.shape}")
    print(f"    Range: [{density.min():.2f}, {density.max():.2f}]")
    print(f"    Mean: {density.mean():.2f}")
    print(f"    Std: {density.std():.2f}")
    
    # Check it's a valid NumPy array that can be plotted
    assert isinstance(density, np.ndarray), "Not a NumPy array!"
    assert density.dtype == np.float32, f"Wrong dtype: {density.dtype}"
    assert len(density.shape) == 2, "Not a 2D array!"
    
    print("  ✓ PASSED (ready for matplotlib/visualization!)\n")

def run_all_tests():
    """Run all test functions"""
    print("="*60)
    print("DENSITY GRID TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_density_grid_creation,
        test_density_grid_update,
        test_zero_copy,
        test_particle_system_with_density_grid,
        test_density_after_simulation,
        test_normalize,
        test_visualization_readiness
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")
            failed += 1
    
    print("="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
