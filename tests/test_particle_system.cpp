/**
 * Unit tests for ParticleSystem
 * Tests particle initialization, boundary conditions, and integration
 */

#include "../include/particle_system.hpp"
#include "../include/config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

bool test_particle_count() {
    std::cout << "Test: Particle count remains constant... ";
    
    ParticleSystem system(1000, 1.0f);
    system.initialize_random(100.0f);
    
    size_t initial_count = system.get_num_particles();
    
    // Run several steps
    for (int i = 0; i < 100; ++i) {
        system.step();
    }
    
    size_t final_count = system.get_num_particles();
    
    if (initial_count != final_count || final_count != 1000) {
        std::cout << "FAILED (initial=" << initial_count 
                  << ", final=" << final_count << ")\n";
        return false;
    }
    
    std::cout << "PASSED\n";
    return true;
}

bool test_periodic_boundaries() {
    std::cout << "Test: Periodic boundaries keep particles in domain... ";
    
    ParticleSystem system(100, 1.0f);
    system.initialize_random(100.0f);
    
    // Run many steps
    for (int i = 0; i < 1000; ++i) {
        system.step();
    }
    
    // Check all particles are within bounds
    auto& x = system.get_x();
    auto& y = system.get_y();
    
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] < 0.0f || x[i] >= 100.0f || 
            y[i] < 0.0f || y[i] >= 100.0f) {
            std::cout << "FAILED (particle " << i 
                      << " at (" << x[i] << ", " << y[i] << "))\n";
            return false;
        }
    }
    
    std::cout << "PASSED\n";
    return true;
}

bool test_thermal_initialization() {
    std::cout << "Test: Thermal velocity initialization... ";
    
    float T = 1.0f;
    // Use the same kB and mass as Config so expected matches ParticleSystem
    float expected_stddev = std::sqrt(Config::kB * T / Config::mass);
    
    ParticleSystem system(5000, T);
    system.initialize_random(100.0f);
    
    auto& vx = system.get_vx();
    auto& vy = system.get_vy();
    
    // Calculate mean and stddev of velocities
    float sum_vx = 0.0f, sum_vy = 0.0f;
    for (size_t i = 0; i < vx.size(); ++i) {
        sum_vx += vx[i];
        sum_vy += vy[i];
    }
    float mean_vx = sum_vx / vx.size();
    float mean_vy = sum_vy / vy.size();
    
    float sq_sum_vx = 0.0f, sq_sum_vy = 0.0f;
    for (size_t i = 0; i < vx.size(); ++i) {
        sq_sum_vx += (vx[i] - mean_vx) * (vx[i] - mean_vx);
        sq_sum_vy += (vy[i] - mean_vy) * (vy[i] - mean_vy);
    }
    float stddev_vx = std::sqrt(sq_sum_vx / vx.size());
    float stddev_vy = std::sqrt(sq_sum_vy / vy.size());
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "(stddev_vx=" << stddev_vx 
              << ", stddev_vy=" << stddev_vy 
              << ", expected=" << expected_stddev << ") ";
    
    // Check if close to expected thermal velocity
    if (std::abs(stddev_vx - expected_stddev) > 0.1f || 
        std::abs(stddev_vy - expected_stddev) > 0.1f) {
        std::cout << "FAILED\n";
        return false;
    }
    
    std::cout << "PASSED\n";
    return true;
}

bool test_position_initialized_in_domain() {
    std::cout << "Test: Initial positions within domain... ";
    
    float domain_size = 100.0f;
    ParticleSystem system(1000, 1.0f);
    system.initialize_random(domain_size);
    
    auto& x = system.get_x();
    auto& y = system.get_y();
    
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] < 0.0f || x[i] >= domain_size || 
            y[i] < 0.0f || y[i] >= domain_size) {
            std::cout << "FAILED (particle " << i 
                      << " at (" << x[i] << ", " << y[i] << "))\n";
            return false;
        }
    }
    
    std::cout << "PASSED\n";
    return true;
}

bool test_step_runs_without_crash() {
    std::cout << "Test: Step function runs without crash... ";
    
    ParticleSystem system(5000, 1.0f);
    system.initialize_random(100.0f);
    
    try {
        for (int i = 0; i < 100; ++i) {
            system.step();
        }
    } catch (...) {
        std::cout << "FAILED (exception thrown)\n";
        return false;
    }
    
    std::cout << "PASSED\n";
    return true;
}

bool test_temperature_effect() {
    std::cout << "Test: Higher temperature → higher velocities... ";
    
    // Low temperature system
    ParticleSystem system_low(1000, 0.1f);
    system_low.initialize_random(100.0f);
    
    // High temperature system
    ParticleSystem system_high(1000, 10.0f);
    system_high.initialize_random(100.0f);
    
    // Run both
    for (int i = 0; i < 100; ++i) {
        system_low.step();
        system_high.step();
    }
    
    // Calculate average speeds
    auto calc_avg_speed = [](const ParticleSystem& sys) {
        auto& vx = sys.get_vx();
        auto& vy = sys.get_vy();
        float sum = 0.0f;
        for (size_t i = 0; i < vx.size(); ++i) {
            sum += std::sqrt(vx[i]*vx[i] + vy[i]*vy[i]);
        }
        return sum / vx.size();
    };
    
    float speed_low = calc_avg_speed(system_low);
    float speed_high = calc_avg_speed(system_high);
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "(low=" << speed_low << ", high=" << speed_high << ") ";
    
    if (speed_high <= speed_low) {
        std::cout << "FAILED (high temp should have higher speed)\n";
        return false;
    }
    
    std::cout << "PASSED\n";
    return true;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║     ParticleSystem Unit Tests                  ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n\n";
    
    int passed = 0;
    int total = 0;
    
    if (test_particle_count()) passed++;
    total++;
    
    if (test_periodic_boundaries()) passed++;
    total++;
    
    if (test_thermal_initialization()) passed++;
    total++;
    
    if (test_position_initialized_in_domain()) passed++;
    total++;
    
    if (test_step_runs_without_crash()) passed++;
    total++;
    
    if (test_temperature_effect()) passed++;
    total++;
    
    std::cout << "\n╔════════════════════════════════════════════════╗\n";
    std::cout << "║  Test Summary: " << passed << "/" << total << " tests passed";
    std::cout << std::string(21, ' ') << "║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";
    
    return (passed == total) ? 0 : 1;
}
