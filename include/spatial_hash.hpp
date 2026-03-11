#ifndef SPATIAL_HASH_HPP
#define SPATIAL_HASH_HPP

#include <vector>
#include <cmath>

/**
 * SpatialHash: Uniform grid-based spatial partitioning for fast neighbor queries.
 * 
 * Purpose: Reduce neighbor search from O(N²) to O(k) where k = average particles per cell.
 * 
 * Core Concept:
 * - Divide space into uniform grid cells
 * - Each particle hashed to a cell based on position
 * - Neighbor queries only check 3×3 = 9 cells (current + 8 neighbors) in 2D
 * 
 * Use case: Collision detection, flocking, interaction forces (Week 3)
 */
class SpatialHash {
private:
    float cell_size;        // Width of each grid cell
    int grid_width;         // Number of cells along one dimension (assumes square domain)
    float domain_size;      // Total domain size (for periodic wrapping)
    std::vector<std::vector<size_t>> cells;  // cells[cell_id] = list of particle indices
    
    /**
     * Convert 2D position to 1D cell ID
     * Handles periodic boundary wrapping
     * Inlined for performance
     */
    inline int get_cell_id(float x, float y) const {
        int cell_x = static_cast<int>(std::floor(x / cell_size));
        int cell_y = static_cast<int>(std::floor(y / cell_size));
        
        // Wrap for periodic boundaries (handles negative indices correctly)
        cell_x = (cell_x % grid_width + grid_width) % grid_width;
        cell_y = (cell_y % grid_width + grid_width) % grid_width;
        
        // Map 2D cell coordinate to 1D index
        return cell_y * grid_width + cell_x;
    }
    
public:
    /**
     * Constructor
     * @param domain_size Total size of simulation domain (assumes square: [0, domain_size]²)
     * @param grid_resolution Number of cells along each dimension (e.g., 10 → 10×10 = 100 cells)
     */
    SpatialHash(float domain_size, int grid_resolution) 
        : domain_size(domain_size), grid_width(grid_resolution) {
        cell_size = domain_size / grid_resolution;
        cells.resize(grid_resolution * grid_resolution);
    }
    
    /**
     * Clear all cells (call before rebuilding hash each frame)
     */
    void clear() {
        for (auto& cell : cells) {
            cell.clear();
        }
    }
    
    /**
     * Reserve capacity in cells to reduce allocations
     * Call once with expected average particles per cell
     */
    void reserve_capacity(size_t avg_particles_per_cell) {
        for (auto& cell : cells) {
            cell.reserve(avg_particles_per_cell * 2);  // 2× buffer for variance
        }
    }
    
    /**
     * Insert a particle into the spatial hash
     * @param particle_id Index of the particle in the main particle array
     * @param x X-coordinate of particle
     * @param y Y-coordinate of particle
     */
    inline void insert(size_t particle_id, float x, float y) {
        int cell_id = get_cell_id(x, y);
        cells[cell_id].push_back(particle_id);
    }
    
    /**
     * Query all particles in neighboring cells (including current cell)
     * Returns particle IDs, NOT filtered by distance - caller must check actual distances
     * 
     * @param x X-coordinate of query point
     * @param y Y-coordinate of query point
     * @param radius Interaction radius (currently unused, reserved for future optimization)
     * @return Vector of particle IDs in the 3×3 cell neighborhood
     */
    std::vector<size_t> query_neighbors(float x, float y, float radius) const {
        std::vector<size_t> neighbors;
        
        int center_cell_x = static_cast<int>(std::floor(x / cell_size));
        int center_cell_y = static_cast<int>(std::floor(y / cell_size));
        
        // Check 3×3 neighborhood (current cell + 8 neighbors)
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int cell_x = center_cell_x + dx;
                int cell_y = center_cell_y + dy;
                
                // Wrap for periodic boundaries
                cell_x = (cell_x % grid_width + grid_width) % grid_width;
                cell_y = (cell_y % grid_width + grid_width) % grid_width;
                
                int cell_id = cell_y * grid_width + cell_x;
                
                // Add all particles from this cell to neighbor list
                for (size_t pid : cells[cell_id]) {
                    neighbors.push_back(pid);
                }
            }
        }
        
        return neighbors;
    }
    
    /**
     * Get total number of particles stored in hash (for verification)
     */
    size_t get_particle_count() const {
        size_t count = 0;
        for (const auto& cell : cells) {
            count += cell.size();
        }
        return count;
    }
    
    /**
     * Get number of cells
     */
    size_t get_num_cells() const {
        return cells.size();
    }
    
    /**
     * Get particles in a specific cell (for debugging/visualization)
     */
    const std::vector<size_t>& get_cell(int cell_id) const {
        return cells[cell_id];
    }
    
    /**
     * Get average particles per cell (load balancing metric)
     */
    float get_average_load() const {
        return static_cast<float>(get_particle_count()) / cells.size();
    }
};

#endif // SPATIAL_HASH_HPP
