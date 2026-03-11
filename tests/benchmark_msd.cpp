#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include "../include/analysis.hpp"
#include "../include/analysis_simd.hpp"
#include "../include/config.hpp"

/**
 * @brief Benchmark comparing scalar vs ARM NEON SIMD MSD computation
 * 
 * Tests both implementations for correctness and performance.
 * Expected speedup: 2.5-3.5x on M1 Mac.
 */

int main() {
    std::cout << "=== MSD Computation Benchmark: Scalar vs ARM NEON ===" << std::endl;
    std::cout << std::endl;
    
    const size_t N = 10000;
    const int iterations = 1000;
    const float domain_size = Config::domain_size;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Particles: " << N << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Domain size: " << domain_size << std::endl;
    std::cout << std::endl;
    
    // Generate test data with realistic particle distributions
    std::vector<float> x(N), y(N), x0(N), y0(N);
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> pos_dist(0.0f, domain_size);
    std::uniform_real_distribution<float> disp_dist(-5.0f, 5.0f);
    
    std::cout << "Generating test data..." << std::endl;
    for (size_t i = 0; i < N; ++i) {
        x0[i] = pos_dist(rng);
        y0[i] = pos_dist(rng);
        
        // Simulate some displacement
        x[i] = x0[i] + disp_dist(rng);
        y[i] = y0[i] + disp_dist(rng);
        
        // Wrap to domain (simulate periodic boundaries)
        if (x[i] < 0.0f) x[i] += domain_size;
        if (x[i] >= domain_size) x[i] -= domain_size;
        if (y[i] < 0.0f) y[i] += domain_size;
        if (y[i] >= domain_size) y[i] -= domain_size;
    }
    
    std::cout << "Done." << std::endl;
    std::cout << std::endl;
    
    // Warm-up run
    float warmup_scalar = compute_msd(x, y, x0, y0, domain_size);
    float warmup_neon = compute_msd_neon(x, y, x0, y0, domain_size);
    
    // Verify correctness (results should match within floating-point tolerance)
    float diff = std::abs(warmup_scalar - warmup_neon);
    float relative_error = diff / warmup_scalar;
    
    std::cout << "Correctness Check:" << std::endl;
    std::cout << "  Scalar MSD: " << warmup_scalar << std::endl;
    std::cout << "  NEON MSD:   " << warmup_neon << std::endl;
    std::cout << "  Difference: " << diff << std::endl;
    std::cout << "  Relative error: " << (relative_error * 100) << "%" << std::endl;
    
    if (relative_error > 1e-5) {
        std::cerr << "  WARNING: Results differ by more than 1e-5!" << std::endl;
        std::cerr << "  NEON implementation may have bugs." << std::endl;
        return 1;
    } else {
        std::cout << "  ✓ Results match within tolerance" << std::endl;
    }
    std::cout << std::endl;
    
    // Benchmark scalar implementation
    std::cout << "Benchmarking scalar implementation..." << std::flush;
    auto start_scalar = std::chrono::high_resolution_clock::now();
    
    volatile float result_scalar = 0.0f;  // volatile prevents optimization away
    for (int iter = 0; iter < iterations; ++iter) {
        result_scalar = compute_msd(x, y, x0, y0, domain_size);
    }
    
    auto end_scalar = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration<double, std::milli>(end_scalar - start_scalar).count();
    
    std::cout << " Done." << std::endl;
    
    // Benchmark NEON implementation
    std::cout << "Benchmarking NEON implementation..." << std::flush;
    auto start_neon = std::chrono::high_resolution_clock::now();
    
    volatile float result_neon = 0.0f;
    for (int iter = 0; iter < iterations; ++iter) {
        result_neon = compute_msd_neon(x, y, x0, y0, domain_size);
    }
    
    auto end_neon = std::chrono::high_resolution_clock::now();
    auto neon_time = std::chrono::duration<double, std::milli>(end_neon - start_neon).count();
    
    std::cout << " Done." << std::endl;
    std::cout << std::endl;
    
    // Report results
    std::cout << "=== Performance Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Scalar implementation:" << std::endl;
    std::cout << "  Total time:     " << scalar_time << " ms" << std::endl;
    std::cout << "  Time per iter:  " << (scalar_time / iterations) << " ms" << std::endl;
    std::cout << std::endl;
    
    std::cout << "NEON implementation:" << std::endl;
    std::cout << "  Total time:     " << neon_time << " ms" << std::endl;
    std::cout << "  Time per iter:  " << (neon_time / iterations) << " ms" << std::endl;
    std::cout << std::endl;
    
    double speedup = scalar_time / neon_time;
    std::cout << std::setprecision(2);
    std::cout << "Speedup: " << speedup << "x";
    
    if (speedup >= 2.5) {
        std::cout << " ✓ EXCELLENT (>2.5x)" << std::endl;
    } else if (speedup >= 2.0) {
        std::cout << " ✓ GOOD (>2.0x)" << std::endl;
    } else if (speedup >= 1.5) {
        std::cout << " ⚠ MODERATE (>1.5x)" << std::endl;
    } else {
        std::cout << " ✗ POOR (<1.5x)" << std::endl;
        std::cout << "  Check compiler flags: -O3 -march=armv8-a+simd" << std::endl;
    }
    std::cout << std::endl;
    
    // Theoretical analysis
    std::cout << "=== Theoretical Analysis ===" << std::endl;
    std::cout << "NEON processes 4 floats per instruction (128-bit vectors)" << std::endl;
    std::cout << "Theoretical maximum speedup: 4.0x" << std::endl;
    std::cout << "Actual speedup: " << speedup << "x (" 
              << std::setprecision(1) << (speedup / 4.0 * 100) << "% of theoretical)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Speedup factors:" << std::endl;
    std::cout << "  - Vectorization: ~4x (process 4 particles simultaneously)" << std::endl;
    std::cout << "  - Overhead: branch prediction, memory alignment, tail loop" << std::endl;
    std::cout << "  - Realistic speedup on M1: 2.5-3.5x" << std::endl;
    std::cout << std::endl;
    
    if (speedup >= 2.0) {
        std::cout << "✓ Benchmark PASSED: NEON optimization is effective!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Benchmark FAILED: NEON speedup below 2.0x threshold" << std::endl;
        return 1;
    }
}
