"""
Comprehensive Python binding tests for StochasticSwarm
Tests all PyBind11 bindings, zero-copy functionality, and RL integration
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import stochastic_swarm as ss
import numpy as np
import pytest


def test_module_import():
    """Test that module imports correctly"""
    assert hasattr(ss, 'ParticleSystem')
    assert hasattr(ss, 'PotentialField')
    assert hasattr(ss, 'DensityGrid')
    print("✓ Module import successful")


def test_particle_system_creation():
    """Test ParticleSystem instantiation"""
    ps = ss.ParticleSystem(num_particles=100, temperature=1.0)
    assert ps.num_particles == 100
    print("✓ ParticleSystem creation successful")


def test_simulation_step():
    """Test that simulation advances"""
    ps = ss.ParticleSystem(100, 1.0)
    ps.initialize_random(100.0)
    
    x0 = ps.get_x().copy()
    
    for _ in range(10):
        ps.step()
    
    x1 = ps.get_x()
    
    # Positions should have changed
    assert not np.allclose(x0, x1)
    print("✓ Simulation step advances correctly")


def test_potential_field():
    """Test potential field control"""
    ps = ss.ParticleSystem(100, 1.0, num_basis=4)
    ps.initialize_random(100.0)
    
    # Set potential
    strengths = [1.0, -1.0, 0.5, -0.5]
    ps.set_potential_params(strengths)
    
    # Get potential field
    pf = ps.get_potential_field()
    assert pf is not None
    assert pf.num_basis == 4
    
    # Check strengths were set
    retrieved = pf.get_strengths()
    assert np.allclose(retrieved, strengths)
    print("✓ Potential field control works")


def test_density_grid():
    """Test density grid computation"""
    ps = ss.ParticleSystem(1000, 1.0, grid_res=10)
    ps.initialize_random(100.0)
    
    ps.update_density_grid()
    density = ps.get_density_grid().get_grid()
    
    # Check shape
    assert density.shape == (10, 10)
    
    # Check total count
    assert np.isclose(density.sum(), 1000, rtol=0.01)
    
    # Check non-negative
    assert np.all(density >= 0)
    print("✓ Density grid computation correct")


def test_zero_copy():
    """Test that density grid uses zero-copy"""
    dg = ss.DensityGrid(10, 10, 100.0)
    
    arr1 = dg.get_grid()
    arr2 = dg.get_grid()
    
    # Should point to same memory
    assert arr1.__array_interface__['data'][0] == \
           arr2.__array_interface__['data'][0]
    print("✓ Zero-copy data sharing verified")


def test_force_computation():
    """Test force field gradient"""
    pf = ss.PotentialField(num_basis=1, domain_size=100.0)
    
    # Zero strength → zero force
    fx, fy = pf.compute_force(50.0, 50.0)
    assert fx == 0.0 and fy == 0.0
    
    # Non-zero strength → non-zero force
    pf.set_strengths([1.0])
    fx2, fy2 = pf.compute_force(60.0, 60.0)
    assert fx2 != 0.0 or fy2 != 0.0
    print("✓ Force computation correct")


def test_gym_environment():
    """Test Gym environment wrapper"""
    try:
        from python.swarm_env import SwarmConcentrationEnv
        
        env = SwarmConcentrationEnv(num_particles=500, num_basis=4)
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (32, 32)
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, (float, np.floating))
        assert obs.shape == (32, 32)
        print("✓ Gym environment integration working")
    except ImportError:
        print("⚠ Gym environment test skipped (gymnasium not installed)")


def test_particle_conservation():
    """Test that particles are conserved"""
    ps = ss.ParticleSystem(1000, 1.0, num_basis=4, grid_res=32)
    ps.initialize_random(100.0)
    
    # Run simulation
    for _ in range(100):
        ps.step()
    
    # Check particle count unchanged
    assert ps.num_particles == 1000
    assert len(ps.get_x()) == 1000
    assert len(ps.get_y()) == 1000
    print("✓ Particle conservation verified")


def test_potential_field_parameters():
    """Test setting all potential field parameters"""
    pf = ss.PotentialField(num_basis=4, domain_size=100.0)
    
    cx = [10.0, 20.0, 30.0, 40.0]
    cy = [15.0, 25.0, 35.0, 45.0]
    strengths = [1.0, -1.0, 2.0, -2.0]
    widths = [5.0, 10.0, 15.0, 20.0]
    
    pf.set_parameters(cx, cy, strengths, widths)
    
    # Verify parameters
    assert np.allclose(pf.get_centers_x(), cx)
    assert np.allclose(pf.get_centers_y(), cy)
    assert np.allclose(pf.get_strengths(), strengths)
    assert np.allclose(pf.get_widths(), widths)
    print("✓ Potential field parameter setting works")


def test_density_grid_normalization():
    """Test density grid normalization"""
    dg = ss.DensityGrid(10, 10, 100.0)
    
    # Add some particles
    x = [5.0, 15.0, 25.0, 35.0]
    y = [5.0, 15.0, 25.0, 35.0]
    dg.update(x, y)
    
    # Get count before normalization
    count_before = dg.get_grid().sum()
    assert np.isclose(count_before, 4.0)
    
    # Normalize to density
    dg.normalize()
    
    # Should be different after normalization
    density_after = dg.get_grid().sum()
    assert density_after != count_before
    print("✓ Density grid normalization works")


def test_boundary_conditions():
    """Test periodic boundary conditions"""
    ps = ss.ParticleSystem(100, 1.0)
    ps.initialize_random(100.0)
    
    # Get positions
    x = ps.get_x()
    y = ps.get_y()
    
    # All particles should be within domain
    assert np.all(x >= 0) and np.all(x <= 100.0)
    assert np.all(y >= 0) and np.all(y <= 100.0)
    
    # Run long simulation
    for _ in range(1000):
        ps.step()
    
    # Check still within bounds
    x = ps.get_x()
    y = ps.get_y()
    assert np.all(x >= 0) and np.all(x <= 100.0)
    assert np.all(y >= 0) and np.all(y <= 100.0)
    print("✓ Periodic boundary conditions working")


def test_attractive_repulsive_forces():
    """Test that potential field forces can be computed and applied"""
    # Create system with potential field
    ps = ss.ParticleSystem(100, 0.1, num_basis=4)
    ps.initialize_random(100.0)
    
    # Get potential field
    pf = ps.get_potential_field()
    assert pf is not None
    
    # Test that forces are zero with zero strengths
    pf.set_strengths([0.0, 0.0, 0.0, 0.0])
    fx, fy = pf.compute_force(50.0, 50.0)
    assert abs(fx) < 1e-6 and abs(fy) < 1e-6
    
    # Test that forces are non-zero with non-zero strengths
    pf.set_strengths([10.0, -10.0, 5.0, -5.0])
    fx, fy = pf.compute_force(50.0, 50.0)
    # Force should be non-zero somewhere
    has_force = abs(fx) > 1e-6 or abs(fy) > 1e-6
    
    # Also test at different location
    fx2, fy2 = pf.compute_force(25.0, 75.0)
    has_force2 = abs(fx2) > 1e-6 or abs(fy2) > 1e-6
    
    assert has_force or has_force2, "Potential field forces not being computed"
    print(f"✓ Potential field force computation working")


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("PYTHON BINDINGS TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_module_import,
        test_particle_system_creation,
        test_simulation_step,
        test_potential_field,
        test_density_grid,
        test_zero_copy,
        test_force_computation,
        test_gym_environment,
        test_particle_conservation,
        test_potential_field_parameters,
        test_density_grid_normalization,
        test_boundary_conditions,
        test_attractive_repulsive_forces,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    # Run with pytest if available, otherwise run directly
    if 'pytest' in sys.modules:
        pytest.main([__file__, '-v'])
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
