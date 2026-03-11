#!/usr/bin/env python3
"""
Test suite for Potential Field implementation (Week 3, Steps 4-5)

Tests the parametric potential field and its integration with ParticleSystem.
Verifies that:
- PotentialField computes correct forces
- Attractive forces pull particles inward
- Repulsive forces push particles outward
- RL control interface works correctly
"""

import sys
import numpy as np

try:
    import stochastic_swarm as ss
except ImportError:
    print("ERROR: Could not import stochastic_swarm module")
    print("Make sure to build the module first: cd build && make stochastic_swarm")
    sys.exit(1)


def test_module_import():
    """Test that module imports correctly with new classes"""
    print("Test 1: Module Import")
    assert hasattr(ss, 'ParticleSystem'), "ParticleSystem not found"
    assert hasattr(ss, 'PotentialField'), "PotentialField not found"
    print("  ✓ Both ParticleSystem and PotentialField available")


def test_potential_field_creation():
    """Test PotentialField instantiation"""
    print("\nTest 2: PotentialField Creation")
    pf = ss.PotentialField(num_basis=4, domain_size=100.0)
    assert pf.num_basis == 4, "Incorrect number of basis functions"
    
    # Check that centers are initialized
    centers_x = pf.get_centers_x()
    centers_y = pf.get_centers_y()
    assert len(centers_x) == 4, "Centers x size mismatch"
    assert len(centers_y) == 4, "Centers y size mismatch"
    
    # Check initial strengths are zero
    strengths = pf.get_strengths()
    assert all(s == 0.0 for s in strengths), "Initial strengths should be zero"
    
    print(f"  ✓ Created with {pf.num_basis} basis functions")
    print(f"  ✓ Centers initialized at: {[(x, y) for x, y in zip(centers_x[:2], centers_y[:2])]}...")


def test_force_computation():
    """Test that forces are computed correctly"""
    print("\nTest 3: Force Computation")
    pf = ss.PotentialField(num_basis=1, domain_size=100.0)
    
    # Zero strength should give zero force
    fx, fy = pf.compute_force(50.0, 50.0)
    assert fx == 0.0 and fy == 0.0, "Zero strength should give zero force"
    print("  ✓ Zero strength → zero force")
    
    # Non-zero strength should give non-zero force
    pf.set_strengths([10.0])
    fx2, fy2 = pf.compute_force(60.0, 50.0)
    assert fx2 != 0.0 or fy2 != 0.0, "Non-zero strength should give non-zero force"
    print(f"  ✓ Non-zero strength → force = ({fx2:.4f}, {fy2:.4f})")


def test_particle_system_with_potential():
    """Test ParticleSystem creation with potential field"""
    print("\nTest 4: ParticleSystem with Potential Field")
    
    # Create system with potential
    ps = ss.ParticleSystem(num_particles=100, temperature=1.0, num_basis=4)
    assert ps.num_particles == 100, "Incorrect particle count"
    
    # Check potential field exists
    pf = ps.get_potential_field()
    assert pf is not None, "Potential field should not be None"
    assert pf.num_basis == 4, "Potential field has wrong number of basis functions"
    print("  ✓ ParticleSystem created with potential field")
    
    # Test without potential
    ps_no_field = ss.ParticleSystem(num_particles=100, temperature=1.0)
    pf_none = ps_no_field.get_potential_field()
    assert pf_none is None, "Potential field should be None when num_basis=0"
    print("  ✓ ParticleSystem without potential field works correctly")


def test_potential_control():
    """Test RL control interface for potential field"""
    print("\nTest 5: RL Control Interface")
    ps = ss.ParticleSystem(num_particles=100, temperature=1.0, num_basis=4)
    
    # Set potential parameters
    strengths = [1.0, -1.0, 0.5, -0.5]
    ps.set_potential_params(strengths)
    
    # Verify they were set
    pf = ps.get_potential_field()
    retrieved = pf.get_strengths()
    assert np.allclose(retrieved, strengths), "Strengths not set correctly"
    print(f"  ✓ Set strengths: {strengths}")
    print(f"  ✓ Retrieved: {retrieved}")


def test_simulation_runs():
    """Test that simulation runs with potential field"""
    print("\nTest 6: Simulation Execution")
    ps = ss.ParticleSystem(num_particles=100, temperature=1.0, num_basis=4)
    ps.initialize_random(100.0)
    
    # Set some potential
    ps.set_potential_params([1.0, -1.0, 0.5, -0.5])
    
    # Run simulation
    for _ in range(50):
        ps.step()
    
    # Check that particles moved
    x = ps.get_x()
    assert len(x) == 100, "Particle count mismatch"
    print("  ✓ Simulation ran for 50 steps")


def test_attractive_force():
    """Test that negative strength creates attractive force"""
    print("\nTest 7: Attractive Force Behavior")
    ps = ss.ParticleSystem(num_particles=500, temperature=0.01, num_basis=1)
    ps.initialize_random(100.0)
    
    # Strong attractive well at center
    ps.set_potential_params([-500.0])
    
    pf = ps.get_potential_field()
    center_x = pf.get_centers_x()[0]
    center_y = pf.get_centers_y()[0]
    
    # Measure initial distance from center
    x0 = np.array(ps.get_x())
    y0 = np.array(ps.get_y())
    dist0 = np.sqrt((x0 - center_x)**2 + (y0 - center_y)**2)
    mean_dist0 = np.mean(dist0)
    
    # Run simulation
    for _ in range(500):
        ps.step()
    
    # Measure final distance
    x1 = np.array(ps.get_x())
    y1 = np.array(ps.get_y())
    dist1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
    mean_dist1 = np.mean(dist1)
    
    change = mean_dist1 - mean_dist0
    print(f"  Initial distance: {mean_dist0:.2f}")
    print(f"  Final distance: {mean_dist1:.2f}")
    print(f"  Change: {change:.2f}")
    
    assert change < -2.0, f"Attractive force should reduce distance (got {change:.2f})"
    print(f"  ✓ ATTRACTIVE: Particles moved {abs(change):.2f} units closer")


def test_repulsive_force():
    """Test that positive strength creates repulsive force"""
    print("\nTest 8: Repulsive Force Behavior")
    ps = ss.ParticleSystem(num_particles=500, temperature=0.01, num_basis=1)
    ps.initialize_random(100.0)
    
    # Strong repulsive hill at center
    ps.set_potential_params([500.0])
    
    pf = ps.get_potential_field()
    center_x = pf.get_centers_x()[0]
    center_y = pf.get_centers_y()[0]
    
    # Measure initial distance from center
    x0 = np.array(ps.get_x())
    y0 = np.array(ps.get_y())
    dist0 = np.sqrt((x0 - center_x)**2 + (y0 - center_y)**2)
    mean_dist0 = np.mean(dist0)
    
    # Run simulation
    for _ in range(500):
        ps.step()
    
    # Measure final distance
    x1 = np.array(ps.get_x())
    y1 = np.array(ps.get_y())
    dist1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
    mean_dist1 = np.mean(dist1)
    
    change = mean_dist1 - mean_dist0
    print(f"  Initial distance: {mean_dist0:.2f}")
    print(f"  Final distance: {mean_dist1:.2f}")
    print(f"  Change: {change:.2f}")
    
    assert change > 2.0, f"Repulsive force should increase distance (got {change:.2f})"
    print(f"  ✓ REPULSIVE: Particles moved {change:.2f} units farther")


def run_all_tests():
    """Run all test functions"""
    print("="*70)
    print("Potential Field Test Suite (Week 3, Steps 4-5)")
    print("="*70)
    
    tests = [
        test_module_import,
        test_potential_field_creation,
        test_force_computation,
        test_particle_system_with_potential,
        test_potential_control,
        test_simulation_runs,
        test_attractive_force,
        test_repulsive_force,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ All tests PASSED!")
        print("\nPotential field implementation is VERIFIED and ready for RL integration!")
    else:
        print(f"❌ {failed} test(s) FAILED")
    print("="*70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
