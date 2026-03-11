#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include "../include/spatial_hash.hpp"
#include "../include/config.hpp"

/**
 * Benchmark: Spatial Hash vs Naive O(N²) Neighbor Search
 * 
 * Goal: Demonstrate that spatial hashing provides significant speedup
 * for neighbor queries in large particle systems.
 * 
 * Expected result: 10-50× speedup for N=10,000 particles
 */

// Naive O(N²) neighbor search
std::vector<size_t> naive_neighbor_search(
    const std::vector<float>& x,
    const std::vector<float>& y,
    size_t query_particle_id,
    float radius,
    float domain_size)
{
    std::vector<size_t> neighbors;
    float radius_sq = radius * radius;
    float half_domain = domain_size * 0.5f;
    
    float qx = x[query_particle_id];
    float qy = y[query_particle_id];
    
    // Check every particle (O(N) per query → O(N²) for all particles)
    for (size_t i = 0; i < x.size(); ++i) {
        float dx = x[i] - qx;
        float dy = y[i] - qy;
        
        // Minimum image convention for periodic boundaries
        if (dx > half_domain) dx -= domain_size;
        if (dx < -half_domain) dx += domain_size;
        if (dy > half_domain) dy -= domain_size;
        if (dy < -half_domain) dy += domain_size;
        
        float dist_sq = dx*dx + dy*dy;
        
        if (dist_sq < radius_sq) {
            neighbors.push_back(i);
        }
    }
    
    return neighbors;
}

int main() {
    // Test parameters
    const size_t N = 10000;
    const float domain_size = Config::domain_size;
    const float search_radius = 10.0f;
    const int num_queries = 1000;  // Number of neighbor queries to benchmark
    
    std::cout << "=== Spatial Hash Benchmark ===" << std::endl;
    std::cout << "Particles: " << N << std::endl;
    std::cout << "Domain size: " << domain_size << std::endl;
    std::cout << "Search radius: " << search_radius << std::endl;
    std::cout << "Number of queries: " << num_queries << std::endl;
    std::cout << std::endl;
    
    // Generate random particle positions
    std::vector<float> x(N), y(N);
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, domain_size);
    
    for (size_t i = 0; i < N; ++i) {
        x[i] = dist(gen);
        y[i] = dist(gen);
    }
    
    // Random query particle indices
    std::uniform_int_distribution<size_t> particle_dist(0, N-1);
    std::vector<size_t> query_particles(num_queries);
    for (int i = 0; i < num_queries; ++i) {
        query_particles[i] = particle_dist(gen);
    }
    
    std::cout << "Generated " << N << " random particle positions" << std::endl;
    std::cout << std::endl;
    
    // ========================================
    // METHOD 1: Naive O(N²) Search
    // ========================================
    std::cout << "Running naive O(N²) neighbor search..." << std::endl;
    
    auto start_naive = std::chrono::high_resolution_clock::now();
    
    size_t total_neighbors_naive = 0;
    for (int q = 0; q < num_queries; ++q) {
        size_t pid = query_particles[q];
        auto neighbors = naive_neighbor_search(x, y, pid, search_radius, domain_size);
        total_neighbors_naive += neighbors.size();
    }
    
    auto end_naive = std::chrono::high_resolution_clock::now();
    auto naive_duration = std::chrono::duration<double, std::milli>(end_naive - start_naive).count();
    
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << naive_duration << " ms" << std::endl;
    std::cout << "  Average neighbors found: " << total_neighbors_naive / num_queries << std::endl;
    std::cout << std::endl;
    
    // ========================================
    // METHOD 2: Spatial Hash
    // ========================================
    std::cout << "Running spatial hash neighbor search..." << std::endl;
    
    // Build spatial hash
    int grid_resolution = 10;  // 10×10 grid
    SpatialHash hash(domain_size, grid_resolution);
    
    auto start_insert = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        hash.insert(i, x[i], y[i]);
    }
    auto end_insert = std::chrono::high_resolution_clock::now();
    auto insert_duration = std::chrono::duration<double, std::milli>(end_insert - start_insert).count();
    
    std::cout << "  Hash construction time: " << insert_duration << " ms" << std::endl;
    std::cout << "  Particles inserted: " << hash.get_particle_count() << std::endl;
    std::cout << "  Average load per cell: " << std::fixed << std::setprecision(1) 
              << hash.get_average_load() << std::endl;
    
    // Query neighbors using spatial hash
    auto start_hash = std::chrono::high_resolution_clock::now();
    
    size_t total_neighbors_hash = 0;
    float radius_sq = search_radius * search_radius;
    float half_domain = domain_size * 0.5f;
    
    for (int q = 0; q < num_queries; ++q) {
        size_t pid = query_particles[q];
        float qx = x[pid];
        float qy = y[pid];
        
        // Get candidate neighbors from hash
        auto candidates = hash.query_neighbors(qx, qy, search_radius);
        
        // Filter by actual distance (hash returns all particles in 3×3 cells)
        size_t neighbor_count = 0;
        for (size_t i : candidates) {
            float dx = x[i] - qx;
            float dy = y[i] - qy;
            
            // Minimum image convention
            if (dx > half_domain) dx -= domain_size;
            if (dx < -half_domain) dx += domain_size;
            if (dy > half_domain) dy -= domain_size;
            if (dy < -half_domain) dy += domain_size;
            
            float dist_sq = dx*dx + dy*dy;
            
            if (dist_sq < radius_sq) {
                neighbor_count++;
            }
        }
        
        total_neighbors_hash += neighbor_count;
    }
    
    auto end_hash = std::chrono::high_resolution_clock::now();
    auto hash_duration = std::chrono::duration<double, std::milli>(end_hash - start_hash).count();
    
    std::cout << "  Query time: " << hash_duration << " ms" << std::endl;
    std::cout << "  Average neighbors found: " << total_neighbors_hash / num_queries << std::endl;
    std::cout << std::endl;
    
    // ========================================
    // Results Comparison
    // ========================================
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Naive O(N²) time:    " << std::setw(8) << naive_duration << " ms" << std::endl;
    std::cout << "Spatial hash time:   " << std::setw(8) << hash_duration << " ms" << std::endl;
    std::cout << "Hash construction:   " << std::setw(8) << insert_duration << " ms" << std::endl;
    std::cout << "Total hash time:     " << std::setw(8) << (hash_duration + insert_duration) << " ms" << std::endl;
    std::cout << std::endl;
    
    double speedup_query_only = naive_duration / hash_duration;
    double speedup_total = naive_duration / (hash_duration + insert_duration);
    
    std::cout << "Speedup (query only):     " << std::setprecision(2) << speedup_query_only << "×" << std::endl;
    std::cout << "Speedup (including build): " << std::setprecision(2) << speedup_total << "×" << std::endl;
    std::cout << std::endl;
    
    // Verification: Check that both methods found similar number of neighbors
    float neighbor_diff = std::abs(static_cast<float>(total_neighbors_naive - total_neighbors_hash)) 
                         / total_neighbors_naive;
    
    std::cout << "=== Verification ===" << std::endl;
    std::cout << "Neighbor count difference: " << std::setprecision(2) << (neighbor_diff * 100) << "%" << std::endl;
    
    if (neighbor_diff < 0.01f) {
        std::cout << "✓ Both methods found same neighbors (< 1% difference)" << std::endl;
    } else {
        std::cout << "✗ WARNING: Neighbor counts differ significantly!" << std::endl;
    }
    
    if (speedup_query_only > 5.0) {
        std::cout << "✓ Spatial hash provides significant speedup (>5×)" << std::endl;
    } else {
        std::cout << "✗ WARNING: Speedup lower than expected" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Benchmark complete!" << std::endl;
    
    return 0;
}
