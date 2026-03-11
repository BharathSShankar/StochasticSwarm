#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cstddef>

/**
 * Configuration parameters for Langevin dynamics simulation
 *
 * Correct Equation: dv = -γv·dt + F/m·dt + √(2γkᵦT/m)·dW
 *
 * The noise coefficient √(2γkᵦT/m) follows from the fluctuation-dissipation theorem,
 * ensuring thermal equilibrium with ⟨v²⟩ = kBT/m.
 */
namespace Config {
    // === Physical Parameters ===
    constexpr float gamma = 1.0f;       // Damping coefficient (viscous friction)
    constexpr float kB = 1.0f;          // Boltzmann constant (normalized to 1)
    constexpr float mass = 2.0f;        // Particle mass
    
    // === Simulation Parameters ===
    constexpr float dt = 0.01f;             // Timestep for integration
    constexpr size_t N_particles = 10000;   // Number of particles (Week 2: scaled to 10k)
    constexpr float domain_size = 100.0f;   // Simulation box size (square domain)
    
    // === Temperature (Experiment with this!) ===
    constexpr float T = 1.0f;           // Temperature (controls thermal noise strength)
                                         // Try: 0.01 (frozen), 1.0 (normal), 10.0 (gas)
    
    // === Analysis Parameters ===
    // NOTE: Relaxation time τ = mass/gamma = 2.0 time units
    // For proper MSD analysis, simulate for at least 10τ = 20 time units (2000 steps)
    // to ensure the system reaches the diffusive regime (t >> τ)
    constexpr int total_steps = 5000;           // Total simulation steps (50 time units)
    constexpr int msd_measurement_interval = 10; // Measure MSD every N steps
    constexpr int output_interval = 500;         // Save CSV/print stats every N steps
}

#endif // CONFIG_HPP
