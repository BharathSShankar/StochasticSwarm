/**
 * Unit tests for RNG module
 * Tests PCG random number generator and Gaussian distribution
 */

#include "../include/rng.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

bool test_pcg_range() {
    std::cout << "Test: PCG generates values in [0, 1)... ";
    PCG rng(42);
    
    for (int i = 0; i < 10000; ++i) {
        float val = rng.uniform();
        if (val < 0.0f || val >= 1.0f) {
            std::cout << "FAILED (value out of range: " << val << ")\n";
            return false;
        }
    }
    
    std::cout << "PASSED\n";
    return true;
}

bool test_gaussian_mean_stddev() {
    std::cout << "Test: Gaussian distribution mean≈0, stddev≈1... ";
    PCG rng(42);
    
    const int N = 10000;
    std::vector<float> samples;
    samples.reserve(N);
    
    for (int i = 0; i < N; ++i) {
        samples.push_back(gaussian_random(rng));
    }
    
    // Calculate mean
    float sum = 0.0f;
    for (float val : samples) {
        sum += val;
    }
    float mean = sum / N;
    
    // Calculate stddev
    float sq_sum = 0.0f;
    for (float val : samples) {
        sq_sum += (val - mean) * (val - mean);
    }
    float stddev = std::sqrt(sq_sum / N);
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "(mean=" << mean << ", stddev=" << stddev << ") ";
    
    // Check if within acceptable bounds
    if (std::abs(mean) > 0.05f || std::abs(stddev - 1.0f) > 0.05f) {
        std::cout << "FAILED\n";
        return false;
    }
    
    std::cout << "PASSED\n";
    return true;
}

bool test_gaussian_with_parameters() {
    std::cout << "Test: Gaussian with custom mean/stddev... ";
    PCG rng(123);
    
    float target_mean = 5.0f;
    float target_stddev = 2.0f;
    const int N = 10000;
    
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        float val = gaussian_random(rng, target_mean, target_stddev);
        sum += val;
    }
    float mean = sum / N;
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "(mean=" << mean << ", expected=" << target_mean << ") ";
    
    if (std::abs(mean - target_mean) > 0.1f) {
        std::cout << "FAILED\n";
        return false;
    }
    
    std::cout << "PASSED\n";
    return true;
}

bool test_pcg_deterministic() {
    std::cout << "Test: PCG is deterministic with same seed... ";
    PCG rng1(42);
    PCG rng2(42);
    
    for (int i = 0; i < 100; ++i) {
        if (rng1.uniform() != rng2.uniform()) {
            std::cout << "FAILED\n";
            return false;
        }
    }
    
    std::cout << "PASSED\n";
    return true;
}

bool test_pcg_different_seeds() {
    std::cout << "Test: PCG produces different sequences with different seeds... ";
    PCG rng1(42);
    PCG rng2(123);
    
    int different_count = 0;
    for (int i = 0; i < 100; ++i) {
        if (rng1.uniform() != rng2.uniform()) {
            different_count++;
        }
    }
    
    // Should be mostly different (allow a few collisions)
    if (different_count < 95) {
        std::cout << "FAILED (only " << different_count << " different values)\n";
        return false;
    }
    
    std::cout << "PASSED\n";
    return true;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║        RNG Module Unit Tests                   ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n\n";
    
    int passed = 0;
    int total = 0;
    
    if (test_pcg_range()) passed++;
    total++;
    
    if (test_gaussian_mean_stddev()) passed++;
    total++;
    
    if (test_gaussian_with_parameters()) passed++;
    total++;
    
    if (test_pcg_deterministic()) passed++;
    total++;
    
    if (test_pcg_different_seeds()) passed++;
    total++;
    
    std::cout << "\n╔════════════════════════════════════════════════╗\n";
    std::cout << "║  Test Summary: " << passed << "/" << total << " tests passed";
    std::cout << std::string(21, ' ') << "║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";
    
    return (passed == total) ? 0 : 1;
}
