/**
 * @file benchmark_particle_system.cpp
 * @brief End-to-end performance benchmarks for the core C++ simulation engine
 *
 * Measures throughput and latency for:
 *   1. Single physics step (Langevin integrator)            – particles/sec
 *   2. Density grid update (particle binning)               – particles/sec
 *   3. Potential field force evaluation (RBF)               – evaluations/sec
 *   4. Full RL pipeline (step + density + potential field)  – steps/sec
 *   5. Scalability sweep (100 → 50 000 particles)
 *
 * Build type: Release (-O3 -march=native) is assumed for representative numbers.
 * Run from the CMake build directory:
 *   ./benchmark_particle_system
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <algorithm>
#include <numeric>

#include "../include/particle_system.hpp"
#include "../include/potential_field.hpp"
#include "../include/density_grid.hpp"
#include "../include/config.hpp"

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Ms        = std::chrono::duration<double, std::milli>;
using Us        = std::chrono::duration<double, std::micro>;

static inline TimePoint now() { return Clock::now(); }
static inline double elapsed_ms(TimePoint t0) { return Ms(now() - t0).count(); }
static inline double elapsed_us(TimePoint t0) { return Us(now() - t0).count(); }

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------
void print_header(const std::string& title) {
    const int W = 70;
    std::cout << "\n" << std::string(W, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(W, '=') << "\n";
}

void print_row(const std::string& label, double value, const std::string& unit,
               double baseline = 0.0) {
    std::cout << std::left  << std::setw(40) << label
              << std::right << std::setw(14) << std::fixed << std::setprecision(2)
              << value << "  " << std::left << std::setw(12) << unit;
    if (baseline > 0.0) {
        std::cout << "  (" << std::fixed << std::setprecision(2)
                  << (value / baseline) << "× vs baseline)";
    }
    std::cout << "\n";
}

void print_separator() {
    std::cout << std::string(70, '-') << "\n";
}

// ---------------------------------------------------------------------------
// 1.  Single physics step benchmark
// ---------------------------------------------------------------------------
void bench_physics_step() {
    print_header("BENCHMARK 1 · Single Physics Step (Langevin Integrator)");

    const std::vector<size_t> Ns = {500, 1000, 2000, 5000, 10000, 20000, 50000};
    const int warmup      = 20;
    const int iterations  = 500;
    const float domain    = Config::domain_size;
    const float temp      = 1.0f;
    const size_t num_basis = 0;   // No potential field in this sub-benchmark
    const size_t grid_res  = 32;

    std::cout << std::left  << std::setw(10) << "Particles"
              << std::right << std::setw(16) << "Step (µs)"
              << std::setw(20) << "Throughput (M p/s)"
              << "\n";
    print_separator();

    for (size_t N : Ns) {
        ParticleSystem ps(N, temp, num_basis, grid_res);
        ps.initialize_random(domain);

        // Warm-up
        for (int i = 0; i < warmup; ++i) ps.step();

        auto t0 = now();
        for (int i = 0; i < iterations; ++i) ps.step();
        double total_ms = elapsed_ms(t0);

        double step_us   = (total_ms / iterations) * 1e3;
        double throughput = (static_cast<double>(N) * iterations) / (total_ms * 1e-3) / 1e6;

        std::cout << std::left  << std::setw(10) << N
                  << std::right << std::setw(16) << std::fixed << std::setprecision(2)
                  << step_us
                  << std::setw(20) << std::fixed << std::setprecision(3)
                  << throughput
                  << "\n";
    }
}

// ---------------------------------------------------------------------------
// 2.  Density grid update benchmark
// ---------------------------------------------------------------------------
void bench_density_grid() {
    print_header("BENCHMARK 2 · Density Grid Update (Particle Binning)");

    const std::vector<size_t> grid_sizes = {16, 32, 64, 128};
    const std::vector<size_t> Ns         = {1000, 5000, 10000};
    const int iterations = 1000;
    const float domain   = Config::domain_size;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> pos(0.0f, domain);

    std::cout << std::left  << std::setw(10) << "Particles"
              << std::setw(10) << "Grid"
              << std::right << std::setw(14) << "Update (µs)"
              << std::setw(20) << "Throughput (M p/s)"
              << "\n";
    print_separator();

    for (size_t N : Ns) {
        std::vector<float> x(N), y(N);
        for (size_t i = 0; i < N; ++i) { x[i] = pos(rng); y[i] = pos(rng); }

        for (size_t g : grid_sizes) {
            DensityGrid grid(g, g, domain);

            // Warm-up
            for (int i = 0; i < 50; ++i) { grid.clear(); grid.update(x, y); }

            auto t0 = now();
            for (int i = 0; i < iterations; ++i) {
                grid.clear();
                grid.update(x, y);
            }
            double total_ms = elapsed_ms(t0);

            double update_us  = (total_ms / iterations) * 1e3;
            double throughput = (static_cast<double>(N) * iterations) / (total_ms * 1e-3) / 1e6;

            std::cout << std::left  << std::setw(10) << N
                      << std::setw(10) << (std::to_string(g) + "×" + std::to_string(g))
                      << std::right << std::setw(14) << std::fixed << std::setprecision(2)
                      << update_us
                      << std::setw(20) << std::fixed << std::setprecision(3)
                      << throughput
                      << "\n";
        }
        print_separator();
    }
}

// ---------------------------------------------------------------------------
// 3.  Potential field force evaluation benchmark
// ---------------------------------------------------------------------------
void bench_potential_field() {
    print_header("BENCHMARK 3 · Potential Field Force Evaluation (RBF)");

    const std::vector<size_t> basis_counts = {4, 9, 16, 25, 36, 49, 64};
    const std::vector<size_t> Ns           = {1000, 5000, 10000};
    const int iterations = 200;
    const float domain   = Config::domain_size;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos(0.0f, domain);
    std::uniform_real_distribution<float> strength(-1000.0f, 1000.0f);

    std::cout << std::left  << std::setw(10) << "Particles"
              << std::setw(10) << "Basis"
              << std::right << std::setw(14) << "Step (µs)"
              << std::setw(22) << "Force evals/sec (M)"
              << "\n";
    print_separator();

    for (size_t N : Ns) {
        std::vector<float> x(N), y(N);
        for (size_t i = 0; i < N; ++i) { x[i] = pos(rng); y[i] = pos(rng); }

        for (size_t nb : basis_counts) {
            // Build particle system with potential field
            ParticleSystem ps(N, 1.0f, nb, 32);
            ps.initialize_random(domain);

            // Assign random strengths (simulates RL action)
            std::vector<float> strengths(nb);
            for (size_t i = 0; i < nb; ++i) strengths[i] = strength(rng);
            ps.set_potential_params(strengths);

            // Warm-up
            for (int i = 0; i < 20; ++i) ps.step();

            auto t0 = now();
            for (int i = 0; i < iterations; ++i) ps.step();
            double total_ms = elapsed_ms(t0);

            double step_us   = (total_ms / iterations) * 1e3;
            double evals_per_sec = (static_cast<double>(N) * nb * iterations)
                                   / (total_ms * 1e-3) / 1e6;

            std::cout << std::left  << std::setw(10) << N
                      << std::setw(10) << nb
                      << std::right << std::setw(14) << std::fixed << std::setprecision(2)
                      << step_us
                      << std::setw(22) << std::fixed << std::setprecision(3)
                      << evals_per_sec
                      << "\n";
        }
        print_separator();
    }
}

// ---------------------------------------------------------------------------
// 4.  Full RL pipeline benchmark
//     Mirrors what SwarmEnv does each gym step:
//       set_potential_params → N × step → update_density_grid
// ---------------------------------------------------------------------------
void bench_full_rl_pipeline() {
    print_header("BENCHMARK 4 · Full RL Pipeline (set_params → step×N → density update)");

    struct Config_ {
        size_t n_particles;
        int    physics_steps;
        size_t num_basis;
        size_t grid_res;
    };

    const std::vector<Config_> configs = {
        { 500,  5,  16, 32},
        { 500, 10,  16, 32},
        {2000,  5,  25, 32},
        {2000, 10,  25, 32},
        {2000, 10,  25, 64},
        {5000, 10,  25, 32},
        {5000, 10,  36, 64},
    };

    const int rl_steps   = 300;   // gym episodes length comparable to training
    const int iterations = 10;
    const float domain   = Config::domain_size;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> strength(-500.0f, 500.0f);

    std::cout << std::left
              << std::setw(8)  << "N"
              << std::setw(8)  << "Phys"
              << std::setw(8)  << "Basis"
              << std::setw(8)  << "Grid"
              << std::right
              << std::setw(14) << "Step (ms)"
              << std::setw(18) << "Steps/sec"
              << std::setw(18) << "Sim-steps/sec"
              << "\n";
    print_separator();

    for (const auto& cfg : configs) {
        ParticleSystem ps(cfg.n_particles, 1.0f, cfg.num_basis, cfg.grid_res);
        ps.initialize_random(domain);

        std::vector<float> strengths(cfg.num_basis);
        for (size_t i = 0; i < cfg.num_basis; ++i) strengths[i] = strength(rng);

        // Warm-up
        for (int step = 0; step < 10; ++step) {
            ps.set_potential_params(strengths);
            for (int p = 0; p < cfg.physics_steps; ++p) ps.step();
            ps.update_density_grid();
        }

        double total_ms = 0.0;
        for (int iter = 0; iter < iterations; ++iter) {
            auto t0 = now();
            for (int step = 0; step < rl_steps; ++step) {
                ps.set_potential_params(strengths);
                for (int p = 0; p < cfg.physics_steps; ++p) ps.step();
                ps.update_density_grid();
            }
            total_ms += elapsed_ms(t0);
        }

        double avg_episode_ms = total_ms / iterations;
        double step_ms        = avg_episode_ms / rl_steps;
        double steps_per_sec  = 1000.0 / step_ms;
        double sim_steps_sec  = steps_per_sec * cfg.physics_steps;

        std::cout << std::left
                  << std::setw(8) << cfg.n_particles
                  << std::setw(8) << cfg.physics_steps
                  << std::setw(8) << cfg.num_basis
                  << std::setw(8) << (std::to_string(cfg.grid_res) + "²")
                  << std::right
                  << std::setw(14) << std::fixed << std::setprecision(3) << step_ms
                  << std::setw(18) << std::fixed << std::setprecision(1) << steps_per_sec
                  << std::setw(18) << std::fixed << std::setprecision(1) << sim_steps_sec
                  << "\n";
    }
}

// ---------------------------------------------------------------------------
// 5.  Scalability sweep
// ---------------------------------------------------------------------------
void bench_scalability() {
    print_header("BENCHMARK 5 · Particle Count Scalability (step + density)");

    const std::vector<size_t> Ns = {100, 500, 1000, 2000, 5000, 10000, 20000, 50000};
    const int iterations  = 200;
    const size_t grid_res = 32;
    const size_t num_basis = 16;
    const float domain    = Config::domain_size;

    std::cout << std::left  << std::setw(12) << "Particles"
              << std::right
              << std::setw(16) << "Step+Dens (µs)"
              << std::setw(20) << "Throughput (M p/s)"
              << std::setw(16) << "Scaling factor"
              << "\n";
    print_separator();

    double baseline_throughput = 0.0;

    for (size_t N : Ns) {
        ParticleSystem ps(N, 1.0f, num_basis, grid_res);
        ps.initialize_random(domain);

        std::vector<float> strengths(num_basis, 100.0f);
        ps.set_potential_params(strengths);

        // Warm-up
        for (int i = 0; i < 20; ++i) { ps.step(); ps.update_density_grid(); }

        auto t0 = now();
        for (int i = 0; i < iterations; ++i) {
            ps.step();
            ps.update_density_grid();
        }
        double total_ms = elapsed_ms(t0);

        double step_us    = (total_ms / iterations) * 1e3;
        double throughput = (static_cast<double>(N) * iterations) / (total_ms * 1e-3) / 1e6;

        if (baseline_throughput == 0.0) baseline_throughput = throughput;
        double scale = throughput / baseline_throughput;

        std::cout << std::left  << std::setw(12) << N
                  << std::right
                  << std::setw(16) << std::fixed << std::setprecision(2) << step_us
                  << std::setw(20) << std::fixed << std::setprecision(3) << throughput
                  << std::setw(16) << std::fixed << std::setprecision(3) << scale
                  << "\n";
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          StochasticSwarm · C++ Performance Benchmarks           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nDomain size : " << Config::domain_size << "\n";
    std::cout << "γ           : " << Config::gamma       << "\n";
    std::cout << "dt          : " << Config::dt          << "\n\n";

    bench_physics_step();
    bench_density_grid();
    bench_potential_field();
    bench_full_rl_pipeline();
    bench_scalability();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  All benchmarks complete.\n";
    std::cout << std::string(70, '=') << "\n";
    return 0;
}
