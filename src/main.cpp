#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <filesystem>
#include <chrono>
#include "config.hpp"
#include "particle_system.hpp"
#include "analysis.hpp"
#include "spatial_hash.hpp"

namespace fs = std::filesystem;

/**
 * Print particle system statistics
 */
void print_particle_stats(const ParticleSystem& sys) {
    auto& x = sys.get_x();
    auto& y = sys.get_y();
    auto& vx = sys.get_vx();
    auto& vy = sys.get_vy();
    
    Stats x_stats = calculate_stats(x);
    Stats y_stats = calculate_stats(y);
    Stats vx_stats = calculate_stats(vx);
    Stats vy_stats = calculate_stats(vy);
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Position X: mean=" << x_stats.mean
              << " ±" << x_stats.stddev
              << " [" << x_stats.min_val << ", " << x_stats.max_val << "]\n";
    std::cout << "  Position Y: mean=" << y_stats.mean
              << " ±" << y_stats.stddev
              << " [" << y_stats.min_val << ", " << y_stats.max_val << "]\n";
    std::cout << "  Velocity X: mean=" << vx_stats.mean
              << " ±" << vx_stats.stddev << "\n";
    std::cout << "  Velocity Y: mean=" << vy_stats.mean
              << " ±" << vy_stats.stddev << "\n";
    
    // Average speed
    float avg_speed = 0.0f;
    for (size_t i = 0; i < vx.size(); ++i) {
        avg_speed += sqrt(vx[i]*vx[i] + vy[i]*vy[i]);
    }
    avg_speed /= vx.size();
    std::cout << "  Average speed: " << avg_speed << "\n";
}

/**
 * Format temperature for directory name (e.g., 1.5 -> "T_1.5")
 */
std::string get_temp_dir_name(float temperature) {
    std::ostringstream oss;
    oss << "T_" << std::fixed << std::setprecision(2) << temperature;
    return oss.str();
}

/**
 * Create output directory for temperature if it doesn't exist
 * Returns the directory path
 */
std::string ensure_output_dir(float temperature) {
    std::string temp_dir = get_temp_dir_name(temperature);
    fs::path output_path = fs::path("../output") / temp_dir;
    
    if (!fs::exists(output_path)) {
        if (!fs::create_directories(output_path)) {
            std::cerr << "Warning: Could not create output directory: " << output_path << "\n";
        }
    }
    
    return output_path.string();
}

/**
 * Save particle positions to CSV file for visualization
 */
void save_to_csv(const ParticleSystem& sys, int frame, const std::string& output_dir) {
    std::string filepath = output_dir + "/frame_" + std::to_string(frame) + ".csv";
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open output file: " << filepath << "\n";
        return;
    }
    
    file << "x,y,vx,vy\n";
    auto& x = sys.get_x();
    auto& y = sys.get_y();
    auto& vx = sys.get_vx();
    auto& vy = sys.get_vy();
    
    for (size_t i = 0; i < x.size(); ++i) {
        file << x[i] << "," << y[i] << "," << vx[i] << "," << vy[i] << "\n";
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  -T, --temperature <value>      Set temperature (default: " << Config::T << ")\n";
    std::cout << "  -s, --steps <value>            Number of simulation steps (default: " << Config::total_steps << ")\n";
    std::cout << "  -m, --msd-interval <value>     MSD measurement interval (default: " << Config::msd_measurement_interval << ")\n";
    std::cout << "  -o, --output-interval <value>  Output/print interval (default: " << Config::output_interval << ")\n";
    std::cout << "  -h, --help                     Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " -T 0.01                    # Run with frozen particles\n";
    std::cout << "  " << program_name << " -T 1.0 -s 5000             # 5000 steps at T=1.0\n";
    std::cout << "  " << program_name << " -T 10.0 -m 5 -o 50         # High temp, frequent MSD, less output\n";
}

int main(int argc, char* argv[]) {
    using namespace Config;
    
    // Parse command-line arguments (with defaults from config)
    float temperature = T;
    int num_steps = total_steps;
    int msd_interval = msd_measurement_interval;
    int output_interval = output_interval;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-T") == 0 || strcmp(argv[i], "--temperature") == 0) {
            if (i + 1 < argc) {
                temperature = static_cast<float>(std::atof(argv[++i]));
                if (temperature < 0.0f) {
                    std::cerr << "Error: Temperature must be non-negative\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: -T requires a value\n";
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--steps") == 0) {
            if (i + 1 < argc) {
                num_steps = std::atoi(argv[++i]);
                if (num_steps <= 0) {
                    std::cerr << "Error: Number of steps must be positive\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: -s requires a value\n";
                return 1;
            }
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--msd-interval") == 0) {
            if (i + 1 < argc) {
                msd_interval = std::atoi(argv[++i]);
                if (msd_interval <= 0) {
                    std::cerr << "Error: MSD interval must be positive\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: -m requires a value\n";
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output-interval") == 0) {
            if (i + 1 < argc) {
                output_interval = std::atoi(argv[++i]);
                if (output_interval <= 0) {
                    std::cerr << "Error: Output interval must be positive\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: -o requires a value\n";
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "╔════════════════════════════════════════════════════╗\n";
    std::cout << "║   StochasticSwarm - Week 1: Langevin Dynamics     ║\n";
    std::cout << "╚════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Simulation Parameters:\n";
    std::cout << "  Particles:       " << N_particles << "\n";
    std::cout << "  Domain:          " << domain_size << " x " << domain_size << "\n";
    std::cout << "  Timestep dt:     " << dt << "\n";
    std::cout << "  Total steps:     " << num_steps << "\n";
    std::cout << "  MSD interval:    " << msd_interval << "\n";
    std::cout << "  Output interval: " << output_interval << "\n";
    std::cout << "\nPhysics Parameters:\n";
    std::cout << "  Damping γ:       " << gamma << "\n";
    std::cout << "  Temperature T:   " << temperature << "\n";
    std::cout << "  Mass m:          " << mass << "\n";
    std::cout << "  Boltzmann kB:    " << kB << "\n";
    
    std::cout << "\n" << std::string(52, '-') << "\n";
    std::cout << "✓ Steps 1-2: Configuration & RNG initialized\n";
    
    // Step 3: Create particle system
    std::cout << "\n[Step 3] Creating particle system...\n";
    ParticleSystem particles(N_particles, temperature);
    std::cout << "✓ ParticleSystem created with " << particles.get_num_particles() << " particles\n";
    
    // Step 4: Initialize particles
    std::cout << "\n[Step 4] Initializing particle positions & velocities...\n";
    particles.initialize_random(domain_size);
    std::cout << "✓ Particles initialized with thermal distribution\n";
    std::cout << "\nInitial State (t=0):\n";
    print_particle_stats(particles);
    
    // Create output directory for this temperature
    std::string output_dir = ensure_output_dir(temperature);
    std::cout << "\n✓ Output directory: " << output_dir << "\n";
    
    // Save initial state
    save_to_csv(particles, 0, output_dir);
    
    // Step 3: Initialize MSD tracking
    std::vector<float> time_points;
    std::vector<float> msd_values;
    
    // Record initial MSD (should be 0.0)
    float initial_msd = compute_msd(particles.get_x(), particles.get_y(),
                                    particles.get_initial_x(), particles.get_initial_y(),
                                    domain_size);
    time_points.push_back(0.0f);
    msd_values.push_back(initial_msd);
    
    // Record initial velocities for VACF
    particles.record_velocities();
    
    // Step 7: Initialize spatial hash (Week 2)
    // Grid resolution chosen from benchmark: 50×50 provides 68× speedup
    const int grid_resolution = 50;
    SpatialHash spatial_hash(domain_size, grid_resolution);
    spatial_hash.reserve_capacity(N_particles / (grid_resolution * grid_resolution) * 2);
    
    std::cout << "\n✓ Spatial hash initialized: " << grid_resolution << "×" << grid_resolution
              << " grid (cell size: " << std::fixed << std::setprecision(2)
              << (domain_size / grid_resolution) << ")\n";
    
    // Run simulation for several timesteps
    std::cout << "\n" << std::string(52, '-') << "\n";
    std::cout << "[Step 5-6] Running Euler-Maruyama integration...\n";
    std::cout << "[Step 3] Tracking Mean Squared Displacement (MSD)...\n";
    std::cout << "[Step 5] Recording velocities for VACF...\n";
    std::cout << "[Step 7-8] Spatial hashing for neighbor queries...\n\n";
    
    std::cout << "Initial MSD (t=0): " << initial_msd << " (expected: ~0.0)\n\n";
    
    // Timing for spatial hash performance monitoring
    double total_hash_build_time = 0.0;
    int hash_rebuild_count = 0;
    
    for (int step = 1; step <= num_steps; ++step) {
        particles.step();
        
        // Rebuild spatial hash periodically (every 10 steps for demonstration)
        // In practice, rebuild every frame for dynamic interactions
        if (step % 10 == 0) {
            auto hash_start = std::chrono::high_resolution_clock::now();
            
            spatial_hash.clear();
            auto& x = particles.get_x();
            auto& y = particles.get_y();
            for (size_t i = 0; i < N_particles; ++i) {
                spatial_hash.insert(i, x[i], y[i]);
            }
            
            auto hash_end = std::chrono::high_resolution_clock::now();
            total_hash_build_time += std::chrono::duration<double, std::milli>(hash_end - hash_start).count();
            hash_rebuild_count++;
        }
        
        // Compute MSD periodically
        if (step % msd_interval == 0) {
            float msd = compute_msd(particles.get_x(), particles.get_y(),
                                   particles.get_initial_x(), particles.get_initial_y(),
                                   domain_size);
            time_points.push_back(step * dt);
            msd_values.push_back(msd);
            
            // Record velocities for VACF at same intervals as MSD
            particles.record_velocities();
        }
        
        if (step % output_interval == 0) {
            std::cout << "Step " << std::setw(4) << step
                      << " (t=" << std::setw(5) << std::setprecision(2) << step * dt << ")";
            
            // Show current MSD if available
            if (!msd_values.empty()) {
                std::cout << " | MSD=" << std::setw(8) << std::setprecision(4) << msd_values.back();
            }
            std::cout << "\n";
            
            print_particle_stats(particles);
            save_to_csv(particles, step, output_dir);
            std::cout << "\n";
        }
    }
    
    // Save MSD data to CSV
    std::string msd_filename = output_dir + "/msd_data.csv";
    if (save_msd_to_csv(time_points, msd_values, msd_filename)) {
        std::cout << "✓ MSD data saved to " << msd_filename << "\n";
    }
    
    // Print MSD summary
    std::cout << "\nMSD Analysis:\n";
    std::cout << "  Initial MSD (t=0):       " << std::setprecision(6) << msd_values.front() << "\n";
    std::cout << "  Final MSD (t=" << time_points.back() << "):    " << msd_values.back() << "\n";
    std::cout << "  Number of measurements:  " << msd_values.size() << "\n";
    
    // Compute and save VACF
    std::cout << "\n" << std::string(52, '-') << "\n";
    std::cout << "[Step 5] Computing Velocity Autocorrelation Function (VACF)...\n";
    
    // Determine max_lag (up to 200 time intervals or all available data)
    int max_lag = std::min(200, static_cast<int>(particles.get_vx_history().size()));
    std::vector<float> vacf_values = compute_vacf(particles.get_vx_history(),
                                                   particles.get_vy_history(),
                                                   max_lag);
    
    // Save VACF to CSV
    std::string vacf_filename = output_dir + "/vacf_data.csv";
    if (save_vacf_to_csv(vacf_values, dt, msd_interval, vacf_filename)) {
        std::cout << "✓ VACF data saved to " << vacf_filename << "\n";
    }
    
    // Print VACF summary
    std::cout << "\nVACF Analysis:\n";
    std::cout << "  VACF(0):                 " << std::setprecision(6) << vacf_values[0] << " (expected: 1.0)\n";
    std::cout << "  Number of lags:          " << vacf_values.size() << "\n";
    std::cout << "  Velocity history size:   " << particles.get_vx_history().size() << "\n";
    
    // Find decay time (where VACF drops below 1/e)
    float threshold = 1.0f / std::exp(1.0f);  // 1/e ≈ 0.368
    int decay_index = -1;
    for (size_t i = 0; i < vacf_values.size(); ++i) {
        if (vacf_values[i] < threshold) {
            decay_index = i;
            break;
        }
    }
    if (decay_index >= 0) {
        float decay_time = decay_index * dt * msd_interval;
        std::cout << "  Decay time (1/e):        " << decay_time << " (expected: ~" << (1.0f/gamma) << " for γ=" << gamma << ")\n";
    }
    
    // Spatial hash performance summary
    std::cout << "\n" << std::string(52, '-') << "\n";
    std::cout << "Spatial Hash Performance (Week 2, Steps 7-8):\n";
    std::cout << "  Grid resolution:         " << grid_resolution << "×" << grid_resolution << "\n";
    std::cout << "  Cell size:               " << std::setprecision(2) << (domain_size / grid_resolution) << "\n";
    std::cout << "  Particles in hash:       " << spatial_hash.get_particle_count() << "\n";
    std::cout << "  Average load per cell:   " << std::setprecision(1) << spatial_hash.get_average_load() << "\n";
    if (hash_rebuild_count > 0) {
        std::cout << "  Hash rebuilds:           " << hash_rebuild_count << "\n";
        std::cout << "  Avg rebuild time:        " << std::setprecision(3)
                  << (total_hash_build_time / hash_rebuild_count) << " ms\n";
        std::cout << "  Total rebuild time:      " << std::setprecision(2)
                  << total_hash_build_time << " ms\n";
    }
    std::cout << "  Note: Benchmark shows " << grid_resolution << "×" << grid_resolution
              << " grid provides ~70× speedup vs O(N²) search\n";
    
    std::cout << "\n" << std::string(52, '-') << "\n";
    std::cout << "✓ Simulation complete! " << num_steps << " timesteps\n";
    std::cout << "\nVisualization:\n";
    std::cout << "  Trajectory CSV files: " << output_dir << "/frame_*.csv\n";
    std::cout << "  MSD data:             " << msd_filename << "\n";
    std::cout << "  VACF data:            " << vacf_filename << "\n";
    std::cout << "\nNext Steps:\n";
    std::cout << "  1. Plot MSD:  python plot_msd.py " << msd_filename << "\n";
    std::cout << "  2. Plot VACF: python plot_vacf.py " << vacf_filename << "\n";
    std::cout << "  3. Run spatial hash benchmarks: ./build/benchmark_spatial_hash_optimized\n";
    std::cout << "  4. Expected: MSD ∝ t (slope ≈ 1 in log-log plot for normal diffusion)\n";
    std::cout << "  5. Expected: VACF decay time ≈ 1/γ ≈ " << (1.0f/gamma) << "\n";
    std::cout << "  6. Try different temperatures: ./StochasticSwarm -T <value>\n";
    std::cout << "     Examples: -T 0.01 (frozen), -T 1.0 (normal), -T 10.0 (gas)\n";
    std::cout << "\nWeek 2 Complete:\n";
    std::cout << "  ✓ Steps 1-6: MSD and VACF analysis implemented\n";
    std::cout << "  ✓ Steps 7-8: Spatial hashing for O(k) neighbor queries (70× speedup!)\n";
    std::cout << "  → Ready for Week 3: Python bindings + RL control with particle interactions\n";
    
    return 0;
}
