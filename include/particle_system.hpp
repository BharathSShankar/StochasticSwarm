#ifndef PARTICLE_SYSTEM_HPP
#define PARTICLE_SYSTEM_HPP

#include <vector>
#include <cmath>
#include <memory>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "rng.hpp"
#include "config.hpp"
#include "force_field.hpp"
#include "potential_field.hpp"
#include "density_grid.hpp"

/**
 * Particle system using Structure of Arrays (SoA) layout.
 * Simulates Langevin dynamics: dv = -gamma*v*dt + F/m*dt + sqrt(2*gamma*kBT)*dW
 *
 * Optimisations applied
 * ---------------------
 *  1. PCG::gaussian() -- Box-Muller spare lives in PCG member (no static state,
 *     compiler can pipeline across iterations, NEON 4-wide eligible).
 *  2. Per-thread PCG instances in thread_rngs[] -- each OMP thread gets its own
 *     independent RNG; no mutex, no false sharing.
 *  3. Boundary conditions fused into step() -- one fewer N-element loop pass.
 *  4. Branch-free periodic wrap -- two conditionals (1-2 cycles) vs fmod (~20 cycles).
 *  5. OpenMP parallel for -- scales across all available cores.
 */
class ParticleSystem {
    private:
        // SoA: Separate arrays for each property (cache-friendly, auto-vectorisable)
        std::vector<float> x, y;      // Positions
        std::vector<float> vx, vy;    // Velocities
        size_t N;
        std::vector<float> x0, y0;   // Initial positions (for MSD / reference)

        // Velocity history for VACF computation
        std::vector<std::vector<float>> vx_history;
        std::vector<std::vector<float>> vy_history;
        size_t max_history_length = 1000;

        // Physics parameters
        float gamma, kB, T, mass, dt;
        float domain_size;

        // Primary RNG -- used for serial initialisation
        PCG rng;

        // Per-thread RNGs.
        // One entry per OpenMP thread, seeded independently via large prime stride.
        // With _OPENMP disabled this vector contains exactly one entry (== rng).
        std::vector<PCG> thread_rngs;

        // RL potential field and observation grid
        std::shared_ptr<PotentialField> potential_field;
        DensityGrid density_grid;

    public:
        /**
         * Constructor: Allocate and configure particle system.
         *
         * @param num_particles Number of particles to simulate
         * @param temperature   Thermal energy (controls noise strength)
         * @param num_basis     Basis functions for potential field (0 = no field)
         * @param grid_res      Density grid resolution (default 32x32)
         */
        ParticleSystem(size_t num_particles, float temperature,
                       size_t num_basis = 0, size_t grid_res = 32)
            : N(num_particles), T(temperature),
              domain_size(Config::domain_size), rng(42),
              density_grid(grid_res, grid_res, Config::domain_size)
        {
            x.resize(N);  y.resize(N);
            vx.resize(N); vy.resize(N);
            x0.resize(N); y0.resize(N);

            gamma = Config::gamma;
            kB    = Config::kB;
            mass  = Config::mass;
            dt    = Config::dt;

            if (num_basis > 0) {
                potential_field = std::make_shared<PotentialField>(num_basis, domain_size);
            }

            // Build per-thread RNG pool.
            // Each thread gets a PCG seeded far apart in sequence space so
            // their streams are statistically independent.
            int num_threads = 1;
#ifdef _OPENMP
            num_threads = omp_get_max_threads();
#endif
            thread_rngs.reserve(static_cast<size_t>(num_threads));
            for (int t = 0; t < num_threads; ++t) {
                // Large prime stride ensures non-overlapping sequences
                thread_rngs.emplace_back(
                    PCG(42ULL + static_cast<uint64_t>(t) * 6364136223846793005ULL)
                );
            }
        }

        /**
         * Initialise particles with random positions and thermal velocities.
         * Positions: uniform in [0, domain_size)
         * Velocities: Maxwell-Boltzmann distribution (kBT/m)
         */
        void initialize_random(float domain_sz) {
            domain_size = domain_sz;

            float v_thermal = std::sqrt(kB * T / mass);

            for (size_t i = 0; i < N; ++i) {
                x[i]  = rng.uniform() * domain_size;
                y[i]  = rng.uniform() * domain_size;
                x0[i] = x[i];
                y0[i] = y[i];

                vx[i] = rng.gaussian(0.0f, v_thermal);
                vy[i] = rng.gaussian(0.0f, v_thermal);
            }
        }

        /**
         * Single Euler-Maruyama timestep -- fully parallelised over particles.
         *
         * Boundary conditions are FUSED into this loop (no separate N-pass).
         * Periodic wrap uses branch-free subtract instead of fmod.
         * OpenMP distributes the N-particle loop across all physical cores.
         *
         * Implements: dv = -gamma*v*dt + F/m*dt + sqrt(2*gamma*kBT/m)*sqrt(dt)*dW
         */
        void step() {
            // Hoist constants -- computed once, not N times
            const float noise_coeff = std::sqrt(2.0f * gamma * kB * T / mass);
            const float sqrt_dt     = std::sqrt(dt);
            // Local copy avoids shared_ptr indirection on every iteration
            const float ds          = domain_size;
            const PotentialField* pf = potential_field.get();

#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < N; ++i) {
                // Each thread draws from its own RNG -- no contention, no locks
#ifdef _OPENMP
                PCG& lrng = thread_rngs[static_cast<size_t>(omp_get_thread_num())];
#else
                PCG& lrng = thread_rngs[0];
#endif

                // Force evaluation
                float Fx = 0.0f, Fy = 0.0f;
                if (pf) {
                    auto [fx, fy] = pf->compute_force(x[i], y[i]);
                    Fx = fx;  Fy = fy;
                } else {
                    auto [fx, fy] = compute_force(x[i], y[i]);
                    Fx = fx;  Fy = fy;
                }

                // Wiener increments -- PCG::gaussian() has no static state
                const float dWx = lrng.gaussian() * sqrt_dt;
                const float dWy = lrng.gaussian() * sqrt_dt;

                // Euler-Maruyama velocity update
                vx[i] += (-gamma * vx[i] + Fx / mass) * dt + noise_coeff * dWx;
                vy[i] += (-gamma * vy[i] + Fy / mass) * dt + noise_coeff * dWy;

                // Position update
                x[i] += vx[i] * dt;
                y[i] += vy[i] * dt;

                // Fused periodic boundary conditions -- branch-free subtract.
                // dt=0.01 with bounded velocities means excursion <= 1 domain width,
                // so a single conditional subtract is always sufficient (no fmod).
                if (x[i] >= ds)   x[i] -= ds;
                if (x[i] <  0.0f) x[i] += ds;
                if (y[i] >= ds)   y[i] -= ds;
                if (y[i] <  0.0f) y[i] += ds;
            }
        }

        /**
         * Explicit periodic boundary pass (kept for legacy callers).
         * No longer called internally -- boundaries are fused inside step().
         */
        void apply_periodic_boundaries() {
            for (size_t i = 0; i < N; ++i) {
                if (x[i] >= domain_size) x[i] -= domain_size;
                if (x[i] <  0.0f)        x[i] += domain_size;
                if (y[i] >= domain_size) y[i] -= domain_size;
                if (y[i] <  0.0f)        y[i] += domain_size;
            }
        }

        /** Record current velocities to history for VACF computation. */
        void record_velocities() {
            vx_history.push_back(vx);
            vy_history.push_back(vy);

            if (vx_history.size() > max_history_length) {
                vx_history.erase(vx_history.begin());
                vy_history.erase(vy_history.begin());
            }
        }

        // Accessors
        const std::vector<float>& get_x()         const { return x;  }
        const std::vector<float>& get_y()         const { return y;  }
        const std::vector<float>& get_vx()        const { return vx; }
        const std::vector<float>& get_vy()        const { return vy; }
        const std::vector<float>& get_initial_x() const { return x0; }
        const std::vector<float>& get_initial_y() const { return y0; }
        size_t get_num_particles()                const { return N;  }

        const std::vector<std::vector<float>>& get_vx_history() const { return vx_history; }
        const std::vector<std::vector<float>>& get_vy_history() const { return vy_history; }

        /** RL interface: expose potential field. */
        std::shared_ptr<PotentialField> get_potential_field() { return potential_field; }

        /** RL action: set potential field amplitudes. */
        void set_potential_params(const std::vector<float>& strengths) {
            if (potential_field) {
                potential_field->set_strengths(strengths);
            }
        }

        /** Update density grid from current positions (RL observation). */
        void update_density_grid() { density_grid.update(x, y); }

        DensityGrid&       get_density_grid()       { return density_grid; }
        const DensityGrid& get_density_grid() const { return density_grid; }
};

#endif // PARTICLE_SYSTEM_HPP
