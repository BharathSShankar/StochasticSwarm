#ifndef DENSITY_GRID_HPP
#define DENSITY_GRID_HPP

#include <vector>
#include <algorithm>

/**
 * Spatial density grid (heatmap) for RL observation.
 * Bins particles into a 2-D grid, counting particles per cell.
 *
 * Optimisation: raw-pointer update overload avoids std::vector copy
 * when called from Python via NumPy arrays (zero-copy path).
 */
class DensityGrid {
private:
    size_t nx, ny;              // Grid dimensions
    float domain_size;          // Physical domain size
    std::vector<float> grid;    // Flattened 2-D array (row-major)
    float cell_size_x, cell_size_y;

public:
    /**
     * Constructor: Initialise grid with given dimensions.
     * @param grid_nx Number of cells in x direction
     * @param grid_ny Number of cells in y direction
     * @param domain  Physical domain size (assumes square domain)
     */
    DensityGrid(size_t grid_nx, size_t grid_ny, float domain)
        : nx(grid_nx), ny(grid_ny), domain_size(domain)
    {
        grid.resize(nx * ny, 0.0f);
        cell_size_x = domain_size / nx;
        cell_size_y = domain_size / ny;
    }

    /** Clear grid (reset all counts to zero). */
    void clear() {
        std::fill(grid.begin(), grid.end(), 0.0f);
    }

    /**
     * Increment the bin that contains particle at (x, y).
     * @param x X position of particle
     * @param y Y position of particle
     */
    void add_particle(float x, float y) {
        int ix = static_cast<int>(x / cell_size_x);
        int iy = static_cast<int>(y / cell_size_y);

        ix = std::clamp(ix, 0, static_cast<int>(nx) - 1);
        iy = std::clamp(iy, 0, static_cast<int>(ny) - 1);

        grid[iy * nx + ix] += 1.0f;
    }

    /**
     * Update grid from std::vector arrays (C++ internal path).
     * @param x Vector of x positions
     * @param y Vector of y positions
     */
    void update(const std::vector<float>& x, const std::vector<float>& y) {
        clear();
        for (size_t i = 0; i < x.size(); ++i) {
            add_particle(x[i], y[i]);
        }
    }

    /**
     * Update grid from raw pointer arrays (zero-copy Python/NumPy path).
     * The Python bindings pass numpy .data() pointers directly here,
     * eliminating the std::vector copy that cost ~20× overhead.
     *
     * @param xdata Pointer to x position array
     * @param ydata Pointer to y position array
     * @param n     Number of particles
     */
    void update(const float* xdata, const float* ydata, size_t n) {
        clear();
        for (size_t i = 0; i < n; ++i) {
            add_particle(xdata[i], ydata[i]);
        }
    }

    /** Normalise grid (convert counts to density: particles per unit area). */
    void normalize() {
        float cell_area = cell_size_x * cell_size_y;
        for (float& val : grid) {
            val /= cell_area;
        }
    }

    // Accessors
    size_t get_nx() const { return nx; }
    size_t get_ny() const { return ny; }
    const std::vector<float>& get_grid()         const { return grid; }
    std::vector<float>&       get_grid_mutable()       { return grid; }
};

#endif // DENSITY_GRID_HPP
